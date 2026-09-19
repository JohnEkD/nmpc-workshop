"""Library for the quadrotor NMPC notebook.

The notebook holds the equations and the tasks; this file holds the
simulation, the plots and the checks.

    import drone_lab as lab
    lab.fetch_model()
    log = lab.run()
    lab.plot_run(log); lab.animate_line(log); lab.animate_mujoco(log)
"""
import os
import time

# common comes before MuJoCo is imported: it sets up 3-D rendering on Colab
from common import (
    NotYet, _figure_frames, _friendly_check, _safe_render, _stitch,
    _stored_video, _video_html, is_path, project_root, require_finished_in,
    stop_unless, video,
)

import casadi as ca
import matplotlib.pyplot as plt
import mujoco
import numpy as np

MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie"
_x2_cache = [None]

# the vehicle, measured
MASS = 1.325                  # kg
G = 9.81                      # m/s^2
IYY = 0.0254117               # kg m^2, about the pitch axis, in the body frame
L = 0.14                      # m, rotor offset along body x
TMAX_ROTOR = 13.0             # N, one rotor
TMAX_PAIR = 2 * TMAX_ROTOR    # N, a pair
ROTOR_R = 0.13                # m, radius of each rotor disc in x2.xml
WEIGHT = MASS * G
REAR, FRONT = [0, 1], [2, 3]  # rotor indices, paired by their x offset

# the task
START = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
GOAL = np.array([4.0, 3.0, 0.0, 0.0, 0.0, 0.0])
OBS_C = np.array([2.0, 2.0])
OBS_R, CLEAR = 0.60, 0.35
# the trunk the obstruction stands on, modelled as two small discs
TRUNK_R = 0.10
TRUNK = (np.array([2.0, 0.5]), np.array([2.0, 1.0]))
PITCH_MAX = 1.0               # rad

DT, N = 0.05, 20
T_TOTAL = 6.0

WEIGHTS = dict(pos=12.0, ang=1.5, vel=1.5, rate=0.2, thrust=0.02, term=8.0)

BLUE, GREEN, GREY, RED, ORANGE = "#1a5fb4", "#2e7d32", "#6b6b6b", "#b3261e", "#c46210"
BODY = "#2c3e50"


# model
def fetch_model(dest=None, quiet=False):
    """Sparse-clone just the Skydio X2 folder from the DeepMind Menagerie."""
    if dest is None:
        dest = project_root()              # beside the notebooks, not in lib/
    root = os.path.join(dest, "mujoco_menagerie")
    if not os.path.isdir(os.path.join(root, "skydio_x2")):
        if not os.path.isdir(root):
            os.system("git clone --depth 1 --filter=blob:none --sparse -q %s %s"
                      % (MENAGERIE, root))
        os.system("cd %s && git sparse-checkout add skydio_x2" % root)
    p = os.path.join(root, "skydio_x2", "x2.xml")
    if not os.path.exists(p):
        raise RuntimeError("could not fetch the X2 model into %s" % root)
    _x2_cache[0] = p
    if not quiet:
        print("vehicle model ready:", os.path.relpath(p))
    return p


def _x2():
    if _x2_cache[0] and os.path.exists(_x2_cache[0]):
        return _x2_cache[0]
    for base in (project_root(), os.getcwd(), "/content"):
        p = os.path.join(base, "mujoco_menagerie", "skydio_x2", "x2.xml")
        if os.path.exists(p):
            _x2_cache[0] = p
            return p
    raise FileNotFoundError(
        "the X2 model is missing. Run  lab.fetch_model()  first.")


def build_plant(drag=False):
    """The X2 as MuJoCo simulates it. Drag off by default.

    The vehicle file has no floor, so a falling vehicle keeps falling."""
    m = mujoco.MjModel.from_xml_path(_x2())
    m.opt.density = 1.225 if drag else 0.0
    return m


def vehicle_facts():
    """Read the numbers off the model rather than trusting the notebook."""
    m = mujoco.MjModel.from_xml_path(_x2())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.body_iquat[1])
    Ib = R.reshape(3, 3) @ np.diag(m.body_inertia[1]) @ R.reshape(3, 3).T
    sites = np.array([m.site_pos[int(m.actuator_trnid[i, 0])]
                      for i in range(m.nu)])
    return dict(mass=float(mujoco.mj_getTotalmass(m)),
                weight=float(mujoco.mj_getTotalmass(m) * G),
                Iyy=float(Ib[1, 1]),
                arm=float(abs(sites[0][0])),
                tmax=float(m.actuator_ctrlrange[0, 1]),
                sites=sites, density=float(m.opt.density))


def pitch_of(d):
    """Pitch angle about the world y axis, from the body quaternion."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, d.qpos[3:7])
    R = R.reshape(3, 3)
    return float(np.arctan2(R[0, 2], R[2, 2]))


def state_of(d):
    return np.array([d.qpos[0], d.qpos[2], pitch_of(d),
                     d.qvel[0], d.qvel[2], d.qvel[4]])


def apply_pair(d, u_rear, u_front):
    """Split each pair total equally between its two rotors."""
    for i in REAR:
        d.ctrl[i] = u_rear / 2.0
    for i in FRONT:
        d.ctrl[i] = u_front / 2.0


# dynamics
def f_sym(s, u):
    """Planar quadrotor, as CasADi expressions."""
    T = u[0] + u[1]
    return ca.vertcat(s[3], s[4], s[5],
                      T * ca.sin(s[2]) / MASS,
                      T * ca.cos(s[2]) / MASS - G,
                      L * (u[0] - u[1]) / IYY)


def f(s, u):
    """The same, on plain numbers."""
    s, u = np.asarray(s, float), np.asarray(u, float)
    T = u[0] + u[1]
    return np.array([s[3], s[4], s[5],
                     T * np.sin(s[2]) / MASS,
                     T * np.cos(s[2]) / MASS - G,
                     L * (u[0] - u[1]) / IYY])


def rk4(s, u, dt, rhs=None):
    rhs = rhs or f
    k1 = rhs(s, u)
    k2 = rhs(s + dt / 2 * k1, u)
    k3 = rhs(s + dt / 2 * k2, u)
    k4 = rhs(s + dt * k3, u)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def hover_thrust():
    """The thrust each pair needs just to hold the vehicle up."""
    return WEIGHT / 2.0


def reference_build(N=N, dt=DT, w=None):
    """The notebook's build(), with the task answers written in.

    Used only when no build is given: to remake the opening video and the
    maintainer's figures. Students' runs use their own build(). The decision
    variables are the thrusts only; the states follow from them through Euler
    steps of the model.
    """
    w = w or dict(WEIGHTS)
    opti = ca.Opti()
    U = opti.variable(2, N)
    s0 = opti.parameter(6)
    goal = ca.DM(GOAL)
    X = [s0]
    for k in range(N):
        X.append(X[-1] + dt * f_sym(X[-1], U[:, k]))
    X = ca.horzcat(*X)
    Q = [w["pos"], w["pos"], w["ang"], w["vel"], w["vel"], w["rate"]]

    def state_cost(e):
        return sum(Q[i] * e[i] ** 2 for i in range(6))

    J = 0
    for k in range(N):
        du = U[:, k] - hover_thrust()
        J = J + state_cost(X[:, k] - goal) + w["thrust"] * ca.dot(du, du)
    J = J + w["term"] * state_cost(X[:, N] - goal)
    opti.minimize(J)
    for k in range(2, N + 1):          # steps 0 and 1 are fixed by the measurement
        d = X[:2, k] - ca.DM(OBS_C)
        opti.subject_to(ca.dot(d, d) - (OBS_R + CLEAR) ** 2 >= 0)
        for tc in TRUNK:
            d = X[:2, k] - ca.DM(tc)
            opti.subject_to(ca.dot(d, d) - (TRUNK_R + CLEAR) ** 2 >= 0)
        opti.subject_to(opti.bounded(-PITCH_MAX, X[2, k], PITCH_MAX))
    opti.subject_to(opti.bounded(0, ca.vec(U), TMAX_PAIR))
    opti.set_initial(U, hover_thrust())
    opti.solver("ipopt", {"print_time": 0, "ipopt.print_level": 0,
                          "ipopt.sb": "yes"})
    return opti, X, U, s0


def run(build=None, N=N, T=T_TOTAL, drag=False, report=True, plant="mujoco"):
    """Run a controller on the vehicle: plan, apply the first pair, repeat.

    build is the notebook's build() (the student's own controller); if it is
    left out, reference_build is used.
    plant="mujoco"  the vehicle is the MuJoCo simulation (the default)
    plant="model"   the vehicle is the prediction model itself, one Euler step
                    per control period, as in the notebook's own loop
    """
    build = reference_build if build is None else build
    opti, X, U, s0p = build(N=N)
    if plant == "mujoco":
        m = build_plant(drag=drag)
        d = mujoco.MjData(m)
        d.qpos[0], d.qpos[2] = START[0], START[1]
        mujoco.mj_forward(m, d)
        sub = int(round(DT / m.opt.timestep))
    else:
        x_model = START.copy()

    states, inputs, plans, ms, fails, fail_at = [], [], [], [], 0, []
    for _ in range(int(T / DT)):
        s0 = state_of(d) if plant == "mujoco" else x_model.copy()
        states.append(s0)
        opti.set_value(s0p, s0)
        t0 = time.time()
        try:
            sol = opti.solve()
            Uopt = np.array(sol.value(U))
            Xopt = np.array(sol.value(X))
        except RuntimeError:
            # the solver stopped without meeting its own test for a best,
            # allowed answer; use its last attempt, as a real controller must
            fails += 1
            fail_at.append(len(states) - 1)
            Uopt = np.array(opti.debug.value(U))
            Xopt = np.array(opti.debug.value(X))
        ms.append((time.time() - t0) * 1e3)
        u = np.clip(Uopt[:, 0], 0.0, TMAX_PAIR)
        inputs.append(u.copy())
        plans.append(Xopt[:2, :].T.copy())
        opti.set_initial(U, np.hstack([Uopt[:, 1:], Uopt[:, -1:]]))

        if plant == "mujoco":
            apply_pair(d, u[0], u[1])
            for _ in range(sub):
                mujoco.mj_step(m, d)
        else:
            x_model = x_model + DT * f(x_model, u)

    log = dict(state=np.array(states), u=np.array(inputs),
               plan=np.array(plans), ms=np.array(ms), fails=fails,
               N=N, drag=drag, plant=plant, fail_at=fail_at)
    log["clearance"] = np.linalg.norm(log["state"][:, :2] - OBS_C, axis=1)
    log["trunk"] = trunk_distance(log["state"])
    if report:
        summary(log)
    return log


def draw_trunk(ax, rings=True, color="#8a6d3b"):
    """The trunk under the obstruction, and its two keep-out rings."""
    top = OBS_C[1] - OBS_R
    ax.add_patch(plt.Rectangle((OBS_C[0] - TRUNK_R, 0.0), 2 * TRUNK_R, top,
                               color=color, zorder=2))
    if rings:
        for c in TRUNK:
            ax.add_patch(plt.Circle(c, TRUNK_R + CLEAR, fc="none", ec=GREEN,
                                    lw=1.2, ls=(0, (2, 3)), zorder=3))


def trunk_distance(states):
    """Distance from the vehicle centre to the nearer trunk disc, per step."""
    S = np.asarray(states, float)
    return np.min([np.hypot(S[:, 0] - c[0], S[:, 1] - c[1]) for c in TRUNK], axis=0)


def arrival_time(log, tol=0.05):
    """First time the vehicle is within `tol` metres of the goal, or None."""
    d = np.linalg.norm(log["state"][:, :2] - GOAL[:2], axis=1)
    hit = np.nonzero(d < tol)[0]
    return None if len(hit) == 0 else float(hit[0] * DT)


def summary(log):
    s, u = log["state"], log["u"]
    near = log["clearance"].min()
    t_arr = arrival_time(log)
    print("window %d steps = %.2f s ahead,  drag %s,  vehicle: %s"
          % (log["N"], log["N"] * DT, "on" if log["drag"] else "off",
             "MuJoCo" if log.get("plant", "mujoco") == "mujoco" else "the model"))
    if t_arr is None:
        print("  flight time        did NOT get within 5 cm of the goal in %.0f s"
              % (len(s) * DT))
    else:
        print("  flight time        within 5 cm of the goal after %.2f s" % t_arr)
    print("  final error        %.3f m" % np.linalg.norm(s[-1, :2] - GOAL[:2]))
    ring = near - (OBS_R + CLEAR)
    print("  closest approach   %.3f m from the centre: %s the keep-out ring by %.3f m,"
          % (near, "outside" if ring >= -5e-4 else "INSIDE", abs(ring)))
    if near >= OBS_R:
        print("                     %.3f m clear of the obstruction itself" % (near - OBS_R))
    else:
        print("                     %.3f m INTO the obstruction (the simulator does not "
              "stop it: nothing solid is there)" % (OBS_R - near))
    tip = near - OBS_R - (L + ROTOR_R)
    print("                     rotor tips (%.2f m from the centre of the vehicle): %s"
          % (L + ROTOR_R, ("clear by %.3f m" % tip) if tip >= 0
             else ("would strike it, %.3f m deep" % -tip)))
    trunk_gap = log["trunk"].min() - (TRUNK_R + CLEAR)
    print("  trunk              %s its keep-out rings by %.3f m"
          % ("outside" if trunk_gap >= -5e-4 else "INSIDE", abs(trunk_gap)))
    print("  largest pitch      %.2f rad (%.0f deg), limit %.2f rad"
          % (np.abs(s[:, 2]).max(), np.degrees(np.abs(s[:, 2]).max()), PITCH_MAX))
    print("  thrust per pair    %.1f to %.1f N   (hover %.2f, limits 0 and %.0f)"
          % (max(u.min(), 0.0) + 0.0, u.max(), hover_thrust(), TMAX_PAIR))
    print("  solver gave up     %d of %d solves" % (log["fails"], len(s)))
    ms = log["ms"]
    print("  computing time     typically %.0f ms per solve, slowest %.0f ms "
          "(a new solve is due every %.0f ms)"
          % (np.median(ms[1:]), np.max(ms[1:]), DT * 1e3))
    print("                     the very first solve took %.0f ms: it starts from "
          "a rough guess" % ms[0])


# plots
OBS_FC = "#c0504d"


def plot_run(log, title=""):
    s, u = log["state"], log["u"]
    t = np.arange(len(s)) * DT
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    draw_trunk(ax[0])
    ax[0].add_patch(plt.Circle(OBS_C, OBS_R, color=OBS_FC))
    ax[0].add_patch(plt.Circle(OBS_C, OBS_R + CLEAR, fill=False, ls=(0, (2, 3)),
                               color=GREEN, lw=1.6))
    ax[0].plot(s[:, 0], s[:, 1], lw=2, color=BLUE)
    ax[0].plot(START[0], START[1], "o", ms=9, color="black", label="start")
    ax[0].plot(GOAL[0], GOAL[1], "*", ms=16, color=GREEN, label="goal")
    ax[0].set_aspect("equal"); ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("z (m)")
    t_arr = arrival_time(log)
    ax[0].set_title(("the path: at the goal after %.2f s" % t_arr) if t_arr is not None
                    else "the path: did not reach the goal")

    ax[1].plot(t, s[:, 2], color=BLUE)
    ax[1].axhline(PITCH_MAX, ls="dashed", color="0.6")
    ax[1].axhline(-PITCH_MAX, ls="dashed", color="0.6")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("pitch $\\theta$ (rad)")
    ax[1].set_title("pitch; dashed = limit $\\pm%.2f$ rad" % PITCH_MAX)

    ax[2].step(t, u[:, 0], where="post", label="rear pair $T_r$")
    ax[2].step(t, u[:, 1], where="post", label="front pair $T_f$")
    ax[2].axhline(TMAX_PAIR, ls="dashed", color="0.6")
    ax[2].axhline(0, ls="dashed", color="0.6")
    ax[2].axhline(hover_thrust(), ls=":", color=GREEN, label="hover")
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("thrust per pair (N)")
    ax[2].legend(fontsize=8); ax[2].set_title("commanded thrust; dashed = limits")

    ax[3].plot(t, log["clearance"], color=BLUE)
    ax[3].axhline(OBS_R + CLEAR, ls=(0, (2, 3)), color=GREEN, label="keep-out ring")
    ax[3].axhline(OBS_R, ls="dashed", color=OBS_FC, label="obstruction surface")
    ax[3].set_ylim(0, max(2.6, log["clearance"].max() * 1.05))
    ax[3].set_xlabel("time (s)"); ax[3].set_ylabel("distance from centre (m)")
    ax[3].legend(fontsize=8); ax[3].set_title("distance from the obstruction centre")
    if title:
        fig.suptitle(title)
    for a in ax[1:]:
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# animations
def _ensure_ffmpeg():
    """matplotlib needs ffmpeg on PATH. imageio-ffmpeg ships one; use it."""
    from matplotlib import animation
    if animation.writers.is_available("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return animation.writers.is_available("ffmpeg")
    except Exception:
        return False


def _clip(fig, frame_fn, n, fps=20, label=""):
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    _ensure_ffmpeg()
    plt.rcParams["animation.embed_limit"] = 100
    anim = FuncAnimation(fig, frame_fn, frames=n, interval=1000 // fps)
    try:
        html = anim.to_html5_video(embed_limit=100)
        html = html.replace(" controls autoplay loop>", " controls>")
    except Exception:
        html = anim.to_jshtml(fps=fps)
    plt.close(fig)
    print("%s: %d frames, %.2f MB" % (label or "clip", n, len(html) / 1e6))
    return HTML(html)


def animate_line(log, fps=20, show_plan=True, frames_only=False):
    """A stick figure: the airframe seen edge on, with the plan ahead of it."""
    s = log["state"]
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    def frame(i):
        ax.clear()
        draw_trunk(ax, rings=False, color="0.55")
        ax.add_patch(plt.Circle(OBS_C, OBS_R, color="0.55"))
        ax.add_patch(plt.Circle(OBS_C, OBS_R + CLEAR, fill=False, ls="dashed",
                                color="0.55"))
        ax.plot(GOAL[0], GOAL[1], "*", ms=16, color=GREEN, label="goal")
        if show_plan:
            pl = log["plan"][i]
            ax.plot(pl[:, 0], pl[:, 1], "o", ms=3.5, color=BLUE, alpha=0.8,
                    label="the plan")
        ax.plot(s[:i + 1, 0], s[:i + 1, 1], "-", lw=2, color=RED, alpha=0.8,
                label="flown so far")
        x, z, th = s[i, 0], s[i, 1], s[i, 2]
        dx, dz = L * np.cos(th), L * np.sin(th)
        ax.plot([x - dx, x + dx], [z - dz, z + dz], lw=6, color=BODY,
                solid_capstyle="round", zorder=4, label="the vehicle")
        for sgn, thrust in ((-1, log["u"][i][0]), (1, log["u"][i][1])):
            bx, bz = x + sgn * dx, z + sgn * dz
            sc = 0.30 * thrust / TMAX_PAIR
            ax.arrow(bx, bz, -sc * np.sin(th), sc * np.cos(th),
                     head_width=0.05, color=ORANGE, zorder=5)
        ax.set_xlim(-0.6, 5.0); ax.set_ylim(-0.2, 4.2)
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("t = %.2f s     clearance %.2f m     pitch %+.0f deg"
                     % (i * DT, log["clearance"][i], np.degrees(th)),
                     fontsize=10)

    if frames_only:
        return fig, frame, len(s)
    return _clip(fig, frame, len(s), fps=fps, label="line animation")


def animate_mujoco(log, fps=20, width=768, height=624, bright=True,
                   frames_only=False, at=None, caption=True):
    """The real airframe, rendered by MuJoCo."""
    from PIL import Image, ImageDraw
    import base64, io
    import imageio.v3 as iio
    from IPython.display import HTML

    spec = mujoco.MjSpec.from_file(
        os.path.join(os.path.dirname(_x2()), "scene.xml"))
    wb = spec.worldbody
    ob = wb.add_body(name="obs", pos=[OBS_C[0], 0.0, OBS_C[1]])
    g = ob.add_geom()
    g.name = "obs_g"; g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [OBS_R, 0.35, 0.0]
    g.quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    g.rgba = [0.80, 0.15, 0.12, 1.0]; g.contype, g.conaffinity = 0, 0
    top = OBS_C[1] - OBS_R
    tb = wb.add_body(name="trunk", pos=[OBS_C[0], 0.0, top / 2])
    tk = tb.add_geom()
    tk.name = "trunk_g"; tk.type = mujoco.mjtGeom.mjGEOM_BOX
    tk.size = [TRUNK_R, 0.10, top / 2]
    tk.rgba = [0.45, 0.32, 0.20, 1.0]; tk.contype, tk.conaffinity = 0, 0
    tg = wb.add_body(name="goal", pos=[GOAL[0], 0.0, GOAL[1]])
    t2 = tg.add_geom()
    t2.name = "goal_g"; t2.type = mujoco.mjtGeom.mjGEOM_SPHERE
    t2.size = [0.13, 0, 0]; t2.rgba = [0.05, 0.50, 0.20, 1.0]
    t2.contype, t2.conaffinity = 0, 0
    if bright:
        h = spec.visual.headlight
        h.diffuse = [0.80] * 3; h.ambient = [0.32] * 3; h.specular = [0.05] * 3
        spec.visual.rgba.haze = [0.72, 0.78, 0.85, 1.0]
        for tx in spec.textures:
            if tx.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                tx.rgb1 = [0.66, 0.76, 0.89]; tx.rgb2 = [0.44, 0.58, 0.76]
            elif tx.name == "groundplane":
                tx.rgb1 = [0.74, 0.76, 0.79]; tx.rgb2 = [0.55, 0.58, 0.62]
                tx.markrgb = [0.28, 0.30, 0.33]
        for mat in spec.materials:
            if mat.name == "groundplane":
                mat.reflectance = 0.05
        for lt in spec.lights:
            lt.castshadow = False
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    m = spec.compile()

    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 90.0, -6.0, 5.8
    cam.lookat[:] = [2.0, 0.0, 2.0]
    r = mujoco.Renderer(m, height=height, width=width)
    frames = []
    try:
        for k in (range(len(log["state"])) if at is None else at):
            s = log["state"][k]
            d.qpos[:3] = [s[0], 0.0, s[1]]
            d.qpos[3:7] = [np.cos(s[2] / 2), 0.0, np.sin(s[2] / 2), 0.0]
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cam)
            im = Image.fromarray(r.render())
            dr = ImageDraw.Draw(im)
            if not caption:
                frames.append(np.array(im))
                continue
            dr.rectangle([0, 0, im.width, 74], fill=(255, 255, 255))
            dr.text((14, 8), "t = %.2f s      pitch %+.0f deg"
                    % (k * DT, np.degrees(s[2])), fill=(15, 15, 15))
            dr.text((14, 30), "red disc = obstruction (a cylinder seen end-on)      "
                    "green ball = goal      the vehicle is the small dark shape",
                    fill=(90, 90, 90))
            dr.text((14, 52), "distance from the obstruction centre %.3f m   "
                    "(keep-out ring %.2f m)"
                    % (log["clearance"][k], OBS_R + CLEAR),
                    fill=((170, 20, 20) if log["clearance"][k] < OBS_R
                          else (15, 15, 15)))
            frames.append(np.array(im))
    finally:
        r.close()
    if frames_only:
        return frames
    buf = io.BytesIO()
    iio.imwrite(buf, np.array(frames), extension=".mp4", fps=fps,
                codec="libx264")
    b64 = base64.b64encode(buf.getvalue()).decode()
    print("mujoco animation: %d frames, %.2f MB" % (len(frames), len(b64) / 1e6))
    return HTML('<video width="%d" controls>'
                '<source src="data:video/mp4;base64,%s" type="video/mp4">'
                '</video>' % (width, b64))


def check_against_mujoco(rhs=None, T=1.5, report=True):
    """Integrate the planar model beside MuJoCo and compare. Drag off."""
    rhs = rhs or f
    m = build_plant(drag=False)
    d = mujoco.MjData(m)
    d.qpos[2] = START[1]                 # start at the same height as the task
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    s = np.array([0.0, float(d.qpos[2]), 0.0, 0.0, 0.0, 0.0])
    hov = hover_thrust()
    th_prev = th_un = 0.0
    for k in range(int(T / dt)):
        dT = 0.20 * np.sin(2 * np.pi * (k * dt) / T)
        u = [hov + dT, hov - dT]
        apply_pair(d, u[0], u[1])
        mujoco.mj_step(m, d)
        s = rk4(s, u, dt, rhs)
        th = pitch_of(d)
        th_un += np.arctan2(np.sin(th - th_prev), np.cos(th - th_prev))
        th_prev = th
    mj = np.array([d.qpos[0], d.qpos[2], th_un, d.qvel[0], d.qvel[2], d.qvel[4]])
    if report:
        print("  %-7s %11s %11s %12s" % ("state", "MuJoCo", "our model", "diff"))
        for i, n in enumerate(["x", "z", "pitch", "vx", "vz", "rate"]):
            print("  %-7s %11.5f %11.5f %12.2e" % (n, mj[i], s[i], mj[i] - s[i]))
    return mj, s


# ================================================================ diagrams


# ================================================= the pieces, for reference
def ref_f(s, u):
    """The equations of motion: the rate of change of every state."""
    s = np.asarray(s, float)
    T = u[0] + u[1]
    return np.array([s[3], s[4], s[5],
                     T * np.sin(s[2]) / MASS,
                     T * np.cos(s[2]) / MASS - G,
                     L * (u[0] - u[1]) / IYY])


def ref_step(s, u, dt=DT):
    """One step forward in time, the simple way."""
    return np.asarray(s, float) + dt * ref_f(s, u)


def ref_state_cost(e, w):
    """What it costs to be away from the goal state."""
    return (w["pos"] * (e[0] ** 2 + e[1] ** 2) + w["ang"] * e[2] ** 2
            + w["vel"] * (e[3] ** 2 + e[4] ** 2) + w["rate"] * e[5] ** 2)


def ref_stage_cost(e, du, w):
    """Cost at one step: being away from the goal, plus using thrust."""
    return ref_state_cost(e, w) + w["thrust"] * float(np.dot(du, du))


def ref_terminal_cost(e, w):
    """Cost at the last step, weighted more heavily."""
    return w["term"] * ref_state_cost(e, w)


def ref_obstacle(p, centre, radius, margin=CLEAR):
    """Positive when the vehicle is clear of the obstacle."""
    d = np.asarray(p, float) - np.asarray(centre, float)
    return float(np.dot(d, d) - (radius + margin) ** 2)


def ref_rollout(s0, U, dt=DT):
    """Where the vehicle would go for a whole sequence of thrusts."""
    X = [np.asarray(s0, float)]
    for k in range(U.shape[1]):
        X.append(ref_step(X[-1], U[:, k], dt))
    return np.array(X)


def ref_total_cost(X, U, goal, w):
    """The cost of a whole plan."""
    N = U.shape[1]
    hov = hover_thrust()
    J = sum(ref_stage_cost(X[k] - goal, U[:, k] - hov, w) for k in range(N))
    return J + ref_terminal_cost(X[N] - goal, w)


def _report(name, worst, tol):
    ok = worst < tol
    print("%s: worst difference from the reference %.2e" % (name, worst))
    print("   %s" % ("PASS" if ok else "not correct yet"))
    return ok


def _rand_state(rng):
    return np.r_[rng.uniform(-3, 3, 2), rng.uniform(-1, 1),
                 rng.uniform(-2, 2, 3)]


def check_f(fn, tol=1e-9):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        s, u = _rand_state(rng), rng.uniform(0, TMAX_PAIR, 2)
        got = np.asarray(fn(s, u), float).ravel()
        if got.shape != (6,):
            print("expected six numbers, received shape", got.shape)
            return False
        worst = max(worst, np.abs(got - ref_f(s, u)).max())
    return _report("equations of motion", worst, tol)


def check_step(fn, tol=1e-9):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(400):
        s, u = _rand_state(rng), rng.uniform(0, TMAX_PAIR, 2)
        got = np.asarray(fn(s, u, DT), float).ravel()
        if got.shape != (6,):
            print("expected six numbers, received shape", got.shape)
            return False
        worst = max(worst, np.abs(got - ref_step(s, u, DT)).max())
    return _report("one step", worst, tol)


def check_state_cost(fn, tol=1e-9):
    rng = np.random.default_rng(2)
    worst = max(abs(float(fn(e, WEIGHTS)) - ref_state_cost(e, WEIGHTS))
                for e in (rng.normal(size=6) for _ in range(400)))
    return _report("state cost", worst, tol)


def check_stage_cost(fn, tol=1e-9):
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(400):
        e, du = rng.normal(size=6), rng.normal(size=2)
        worst = max(worst, abs(float(fn(e, du, WEIGHTS))
                               - ref_stage_cost(e, du, WEIGHTS)))
    return _report("stage cost", worst, tol)


def check_terminal_cost(fn, tol=1e-9):
    rng = np.random.default_rng(4)
    worst = max(abs(float(fn(e, WEIGHTS)) - ref_terminal_cost(e, WEIGHTS))
                for e in (rng.normal(size=6) for _ in range(400)))
    return _report("terminal cost", worst, tol)


def check_obstacle(fn, tol=1e-9):
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(400):
        p = rng.uniform(0, 4, 2)
        worst = max(worst, abs(float(fn(p, OBS_C, OBS_R, CLEAR))
                               - ref_obstacle(p, OBS_C, OBS_R, CLEAR)))
    return _report("obstacle constraint", worst, tol)


def check_rollout(fn, tol=1e-9):
    """X must be 6 rows by N+1 columns: one column per step."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(50):
        s0 = _rand_state(rng)
        U = rng.uniform(0, TMAX_PAIR, (2, 8))
        got = np.asarray(ca.DM(fn(s0, U, DT)))
        if got.shape != (6, 9):
            print("expected a 6 by 9 matrix, received", got.shape)
            return False
        worst = max(worst, np.abs(got - ref_rollout(s0, U, DT).T).max())
    return _report("rollout", worst, tol)


animate_mujoco = _safe_render(animate_mujoco)


_vehicle_facts_live = vehicle_facts


def vehicle_facts():
    """Parameters of the airframe. Falls back to stored values if MuJoCo
    cannot be imported, which happens when no graphics backend is present."""
    try:
        return _vehicle_facts_live()
    except Exception:
        return dict(mass=MASS, weight=WEIGHT, Iyy=IYY, arm=L,
                    tmax=TMAX_ROTOR, sites=None, density=1.225)


def check_total_cost(fn, tol=1e-6):
    """X is 6 by N+1, U is 2 by N."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(50):
        s0 = _rand_state(rng)
        U = rng.uniform(0, TMAX_PAIR, (2, 8))
        X = ref_rollout(s0, U, DT)
        worst = max(worst, abs(float(ca.DM(fn(X.T, U, GOAL, WEIGHTS)))
                               - ref_total_cost(X, U, GOAL, WEIGHTS)))
    return _report("total cost", worst, tol)


def check_loop(path, tol_goal=0.02):
    """The closed loop in the notebook's last task: arrived, and kept out?"""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 6:
        print("expected one row of six numbers per step, received shape", path.shape)
        return False
    err = float(np.linalg.norm(path[-1, :2] - GOAL[:2]))
    near = float(np.hypot(path[:, 0] - OBS_C[0], path[:, 1] - OBS_C[1]).min())
    ring = near - (OBS_R + CLEAR)
    trunk = float(trunk_distance(path).min()) - (TRUNK_R + CLEAR)
    over = path[np.argmin(np.abs(path[:, 0] - OBS_C[0])), 1] > OBS_C[1]
    print("ended %.1f mm from the goal; closest approach %.3f m from the centre "
          "(keep-out ring %.2f m)" % (1000 * err, near, OBS_R + CLEAR))
    print("passed %s the obstruction; trunk keep-out rings %s"
          % ("over" if over else "UNDER", "clear" if trunk > -1e-3 else "ENTERED"))
    ok = err < tol_goal and ring > -1e-3 and trunk > -1e-3
    print("   PASS" if ok else "   not yet: the vehicle should end within 2 cm and "
          "never enter the ring")
    return ok


for _name in [n for n in list(globals())
              if n.startswith("check_") and n != "check_against_mujoco"]:
    globals()[_name] = _friendly_check(globals()[_name])


_check_against_mujoco_live = check_against_mujoco


def check_against_mujoco(rhs=None, T=1.5, report=True):
    """Integrate the planar model beside MuJoCo and compare. Drag off.

    If the model passed in is not finished yet, say so plainly and return
    (None, None) instead of stopping with an error.
    """
    try:
        return _check_against_mujoco_live(rhs=rhs, T=T, report=report)
    except (NotImplementedError, TypeError) as err:
        if isinstance(err, TypeError) and "ellipsis" not in str(err):
            raise
        print("not yet: finish Task 1 (my_f) so that its check prints PASS, "
              "then run this cell again.")
        return None, None


def preview(render=False):
    """The finished controller, shown before you build it: MuJoCo beside Python.

    Plays the stored video in videos/ if it can be found. render=True makes a
    fresh one instead, which takes about 25 seconds.
    """
    name = "drone_preview.mp4"
    stored = None if render else _stored_video(name)
    if stored:
        return _video_html(open(stored, "rb").read())
    print("making the video, about 25 seconds ...")
    fetch_model(quiet=True)            # the MuJoCo model, downloaded once
    log = run(report=False)
    fig, frame, n = animate_line(log, frames_only=True)
    left = animate_mujoco(log, frames_only=True, width=640, height=520)
    data = _stitch(left, _figure_frames(fig, frame, n), 20,
                   DRONE_HEADLINE, DRONE_LEFT, DRONE_RIGHT)
    return _video_html(data)


DRONE_HEADLINE = ("Build a controller similar to this: it flies the quadrotor "
                  "over the obstruction to the treatment point")
DRONE_LEFT = "left: the quadrotor, simulated in MuJoCo"
DRONE_RIGHT = "right: the controller's view; the dots are its current plan"


_TASK_CHECKS = {
    "my_f": ("check_f", "Task 1"),
    "my_state_cost": ("check_state_cost", "Task 2"),
    "my_stage_cost": ("check_stage_cost", "Task 2"),
    "my_terminal_cost": ("check_terminal_cost", "Task 2"),
    "my_step": ("check_step", "the cell of parts written for you"),
    "my_rollout": ("check_rollout", "the cell of parts written for you"),
    "my_total_cost": ("check_total_cost", "the cell of parts written for you"),
    "my_obstacle": ("check_obstacle", "the cell of parts written for you"),
}


# used by the notebooks as lab.is_path, lab.stop_unless, lab.video, lab.NotYet
NotYet, is_path, stop_unless, video = NotYet, is_path, stop_unless, video


def require_finished(*functions):
    """Stop with a plain message if any of these task functions fails its check."""
    require_finished_in(globals(), _TASK_CHECKS, *functions)




class MuJoCoVehicle:
    """The X2 as a simple object: reset it, then step it with a pair of thrusts.

    Mirrors robot_lab.MuJoCoBase, so a notebook loop reads the same way:

        vehicle = lab.MuJoCoVehicle()
        s = vehicle.reset()
        s = vehicle.step(rear, front, lab.DT)
    """

    name = "MuJoCo X2"

    def __init__(self, drag=False):
        self.model = build_plant(drag=drag)
        self.data = mujoco.MjData(self.model)
        self.sub = int(round(DT / self.model.opt.timestep))

    def reset(self, start=None):
        start = START if start is None else np.asarray(start, float)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0], self.data.qpos[2] = start[0], start[1]
        mujoco.mj_forward(self.model, self.data)
        return state_of(self.data)

    def step(self, rear, front, dt=DT):
        apply_pair(self.data, float(rear), float(front))
        for _ in range(int(round(dt / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        return state_of(self.data)


def rates_of_change(x, z, th, vx, vz, om, rear, front):
    """How fast each of the six states is changing, from section 2.2.

    Works with plain numbers and with the symbols a controller uses.
    """
    thrust = rear + front
    return (vx,
            vz,
            om,
            thrust * ca.sin(th) / MASS,
            thrust * ca.cos(th) / MASS - G,
            L * (rear - front) / IYY)


def keep_out_rule(x, z, cx, cz, safe):
    """At or below zero while the vehicle is at least `safe` from (cx, cz)."""
    return safe ** 2 - ((x - cx) ** 2 + (z - cz) ** 2)


def draw_scene(ax, goal=None, margin=CLEAR):
    """The side view: the obstruction, its trunk, the keep-out rings, start and goal."""
    goal = GOAL if goal is None else goal
    draw_trunk(ax, rings=False)
    ax.add_patch(plt.Circle(OBS_C, OBS_R, color=OBS_FC, zorder=3))
    ax.add_patch(plt.Circle(OBS_C, OBS_R + margin, fc="none", ec=GREEN, lw=1.6,
                            ls=(0, (2, 3)), zorder=3))
    for c in TRUNK:
        ax.add_patch(plt.Circle(c, TRUNK_R + margin, fc="none", ec=GREEN, lw=1.2,
                                ls=(0, (2, 3)), zorder=3))
    ax.plot(START[0], START[1], "o", color="black", ms=8)
    ax.plot(goal[0], goal[1], "*", color=GREEN, ms=16)
    ax.set_xlim(-0.6, 5.2); ax.set_ylim(-0.3, 4.2)
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")


def flight_report(states, goal=None, margin=CLEAR):
    """Say when the vehicle arrived and how close it came, in plain words."""
    S = np.asarray(states, float)
    goal = GOAL if goal is None else np.asarray(goal, float)
    gone = np.linalg.norm(S[:, :2] - goal[:2], axis=1)
    there = np.nonzero(gone < 0.05)[0]
    if len(there):
        print("reached the goal after %.2f s" % (there[0] * DT))
    else:
        print("did NOT reach the goal: stopped %.2f m away" % gone[-1])
    near = float(np.hypot(S[:, 0] - OBS_C[0], S[:, 1] - OBS_C[1]).min())
    trunk = float(trunk_distance(S).min())
    print("closest approach %.3f m from the obstruction's centre: %s its keep-out "
          "ring by %.3f m" % (near, "outside" if near >= OBS_R + margin - 5e-4
                              else "INSIDE", abs(near - OBS_R - margin)))
    tips = near - OBS_R - (L + ROTOR_R)
    print("rotor tips %s the obstruction by %.3f m"
          % ("clear of" if tips >= 0 else "INTO", abs(tips)))
    gap = trunk - TRUNK_R - margin
    print("trunk %s its keep-out rings by %.3f m"
          % ("outside" if gap >= -5e-4 else "INSIDE", abs(gap)))


def filmstrip(log, shots=4, width=420, height=340):
    """A row of MuJoCo pictures across a flight, so the vehicle can be seen."""
    S = log["state"]
    moving = np.linalg.norm(S[:, 3:5], axis=1) > 0.05
    last = int(np.nonzero(moving)[0][-1]) if moving.any() else len(S) - 2
    n = min(last + 8, len(S) - 1)                 # a moment after it settles
    at = [int(round(k * n / (shots - 1))) for k in range(shots)]
    frames = animate_mujoco(log, at=at, frames_only=True, width=width,
                            height=height, caption=False)
    if frames is None:                            # no 3-D rendering here
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(3.4 * len(frames), 2.9))
    for ax, frame, k in zip(np.atleast_1d(axes), frames, at):
        ax.imshow(frame); ax.axis("off")
        ax.set_title("t = %.2f s" % (k * DT), fontsize=9)
    plt.tight_layout()
    plt.show()


def compare_runs(log, reference=None, labels=("this flight", "the first flight"),
                 goal=None, title=""):
    """The path, the thrusts and the tilt, and the distance to the goal."""
    goal = GOAL if goal is None else np.asarray(goal, float)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.4),
                                     gridspec_kw={"width_ratios": [1.15, 1, 1]})
    tilt_axis = a2.twinx()                       # the tilt has its own scale
    thrust_colour, tilt_colour = "#e07b39", "#7a4fa3"
    for lg, colour, name, wide in ((reference, "0.62", labels[1], 3.4),
                                   (log, BLUE, labels[0], 2.2)):
        if lg is None:
            continue
        S, U = lg["state"], np.asarray(lg["u"], float)
        T = np.asarray(lg["t"], float) if "t" in lg else np.arange(len(U)) * DT
        a1.plot(S[:, 0], S[:, 1], lw=wide, color=colour, label=name)
        first = lg is reference
        a2.plot(T, U[:, 0], lw=wide, color="0.62" if first else thrust_colour,
                label=("the first flight" if first else name) + ": rear")
        a2.plot(T, U[:, 1], lw=wide, color="0.62" if first else thrust_colour,
                ls=(0, (4, 2)),
                label=("the first flight" if first else name) + ": front")
        tilt_axis.plot(T, S[:len(T), 2], lw=wide * 0.8,
                       color="0.62" if first else tilt_colour, ls=(0, (1, 2)),
                       label=("the first flight" if first else name) + ": tilt")
        gap = np.linalg.norm(S[:len(T), :2] - goal[:2], axis=1)
        a3.plot(T, gap, lw=wide, color=colour, label=name)
    draw_scene(a1, goal=goal)
    a1.set_title("where it flew", fontsize=10)
    a1.legend(loc="lower left", fontsize=8)

    a2.axhline(TMAX_PAIR, color=thrust_colour, lw=1, ls=(0, (2, 3)))
    a2.axhline(0, color=thrust_colour, lw=1, ls=(0, (2, 3)))
    a2.axhline(hover_thrust(), color="0.4", lw=1, ls=(0, (1, 3)))
    a2.text(0.05, hover_thrust() + 0.4, "hover", fontsize=8, color="0.4")
    a2.set_xlabel("time (s)")
    a2.set_ylabel("thrust per pair (N): rear solid, front dashed",
                  color=thrust_colour)
    a2.tick_params(axis="y", labelcolor=thrust_colour)
    a2.set_ylim(-TMAX_PAIR * 0.1, TMAX_PAIR * 1.15)
    a2.set_title("what it was told to do, and how it tilted (limits dotted)",
                 fontsize=10)
    tilt_axis.set_ylabel("tilt (rad), dotted line", color=tilt_colour)
    tilt_axis.tick_params(axis="y", labelcolor=tilt_colour)
    tilt_axis.axhline(PITCH_MAX, color=tilt_colour, lw=1, ls=(0, (2, 3)))
    tilt_axis.axhline(-PITCH_MAX, color=tilt_colour, lw=1, ls=(0, (2, 3)))
    tilt_axis.set_ylim(-PITCH_MAX * 1.3, PITCH_MAX * 1.3)
    lines = a2.get_lines() + tilt_axis.get_lines()
    keep = [ln for ln in lines if ln.get_label() and not ln.get_label().startswith("_")]
    a2.legend(keep, [ln.get_label() for ln in keep], fontsize=7.5, loc="upper right")
    a2.grid(alpha=0.25)

    a3.axhline(0.05, color=GREEN, lw=1, ls=(0, (2, 3)))
    a3.text(0.05, 0.057, "at the goal", fontsize=8, color=GREEN)
    a3.set_xlabel("time (s)"); a3.set_ylabel("distance to the goal (m)")
    a3.set_ylim(0, None)
    a3.set_title("how far from the goal", fontsize=10)
    a3.legend(fontsize=8, loc="lower left"); a3.grid(alpha=0.25, which="both")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()
