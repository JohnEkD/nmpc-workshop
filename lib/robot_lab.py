"""Library for the mobile-robot NMPC notebook.

A kinematic formulation: the state is the earth-fixed pose of the robot and
the input is the body-fixed forward speed together with the turn rate. The
notebook holds the equations and the tasks; this file holds the simulation,
the plots and the checks.
"""
import os
import pathlib

# common comes before MuJoCo is imported: it sets up 3-D rendering on Colab
from common import (
    NotYet, _figure_frames, _friendly_check, _safe_render, _stitch,
    _stored_video, _video_html, is_path, project_root, require_finished_in,
    stop_unless, video,
)

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

R_ROBOT = 0.30                       # swept radius used for keep-out, m
TRACK = 0.42                          # wheel track, for the drawing only

V_MAX = 1.00                         # m/s
W_MAX = 1.50                         # rad/s
MARGIN = 0.10                        # safety margin outside robot + obstacle, m

# the floor
ROOM = (0.0, 5.0, 0.0, 4.0)
OBSTACLES = [(1.6, 1.2, 0.30, "machine base"),
             (3.2, 2.6, 0.25, "pallet stack"),
             (1.2, 3.0, 0.20, "support column")]
START, GOAL = (0.5, 0.5), (4.3, 3.4)


MENAGERIE = "https://github.com/google-deepmind/mujoco_menagerie"
_xml_cache = [None]


def fetch_model(dest=None, quiet=False):
    """Sparse-clone just the robot folder from the DeepMind Menagerie."""
    if dest is None:
        dest = project_root()              # beside the notebooks, not in lib/
    root = os.path.join(dest, "mujoco_menagerie")
    if not os.path.isdir(os.path.join(root, "stanford_tidybot")):
        if not os.path.isdir(root):
            os.system("git clone --depth 1 --filter=blob:none --sparse -q %s %s"
                      % (MENAGERIE, root))
        os.system("cd %s && git sparse-checkout add stanford_tidybot" % root)
    p = os.path.join(root, "stanford_tidybot", "scene_base.xml")
    if not os.path.exists(p):
        raise RuntimeError("could not fetch the robot base into %s" % root)
    _xml_cache[0] = p
    if not quiet:
        print("robot model ready:", os.path.relpath(p))
    return p


def _xml():
    if _xml_cache[0] and os.path.exists(_xml_cache[0]):
        return _xml_cache[0]
    for base in (pathlib.Path(project_root()), pathlib.Path.cwd(),
                 pathlib.Path("/content")):
        p = base / "mujoco_menagerie" / "stanford_tidybot" / "scene_base.xml"
        if p.exists():
            _xml_cache[0] = str(p)
            return str(p)
    raise FileNotFoundError("the robot model is missing. Run lab.fetch_model() first.")


# the plants
class PlanningModel:
    """The prediction model used by the controller.

    Kinematics only: no mass, no actuator dynamics and no contact.
    """

    name = "planning model"
    w = None

    def reset(self, pose):
        self.pose = np.array(pose, float)
        return self.pose

    def step(self, v, om, dt):
        x, y, th = self.pose
        self.pose = np.array([x + dt * v * np.cos(th),
                              y + dt * v * np.sin(th), th + dt * om])
        return self.pose


class MuJoCoBase:
    """The simulated vehicle, commanded through its velocity servos.

    The chassis as modelled is omnidirectional. The nonholonomic restriction is
    imposed by the controller rather than by the hardware: the body-fixed
    forward speed is transformed into earth-fixed axes through the heading, and
    no lateral velocity is ever commanded.
    """

    name = "MuJoCo"
    w = None

    def __init__(self, kv=800.0):
        import mujoco
        spec = mujoco.MjSpec.from_file(_xml())
        for a in spec.actuators:
            if a.trntype == mujoco.mjtTrn.mjTRN_JOINT:
                a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
                a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
                a.gainprm[0] = kv
                a.biasprm[:] = 0.0
                a.biasprm[2] = -kv
                a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        self._mj = mujoco
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

    def reset(self, pose):
        self._mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = pose
        self._mj.mj_forward(self.model, self.data)
        return np.array(self.data.qpos[:3])

    def step(self, v, om, dt):
        th = float(self.data.qpos[2])
        ctrl = [v * np.cos(th), v * np.sin(th), om]
        for _ in range(int(round(dt / self.model.opt.timestep))):
            self.data.ctrl[:3] = ctrl
            self._mj.mj_step(self.model, self.data)
        return np.array(self.data.qpos[:3])


def circles():
    return [(x, y, r) for x, y, r, _ in OBSTACLES]


def reference_build(N=25, dt=0.1, w=None, margin=MARGIN):
    """The notebook's build(), with the task answers written in.

    Used only when no build is given: to remake the opening video and the
    maintainer's figures. Students' runs use their own build().
    """
    w = w or dict(WEIGHTS)

    def step(s, u):
        return ca.vertcat(s[0] + dt * u[0] * ca.cos(s[2]),
                          s[1] + dt * u[0] * ca.sin(s[2]),
                          s[2] + dt * u[1])

    opti = ca.Opti()
    U = opti.variable(2, N)
    s0 = opti.parameter(3)
    goal = opti.parameter(2)
    X = [s0]
    for k in range(N):
        X.append(step(X[-1], U[:, k]))
    X = ca.horzcat(*X)
    J = 0
    for k in range(N):
        e = X[:2, k] - goal
        J = J + w["q"] * ca.dot(e, e) + w["r"] * ca.dot(U[:, k], U[:, k])
    e = X[:2, N] - goal
    J = J + w["qN"] * ca.dot(e, e)
    opti.minimize(J)
    for k in range(1, N + 1):
        for c in OBSTACLES:
            d = X[:2, k] - ca.DM([c[0], c[1]])
            opti.subject_to(ca.dot(d, d) - (c[2] + R_ROBOT + margin) ** 2 >= 0)
    opti.subject_to(opti.bounded(-V_MAX, U[0, :], V_MAX))
    opti.subject_to(opti.bounded(-W_MAX, U[1, :], W_MAX))
    opti.solver("ipopt", {"print_time": 0, "ipopt.print_level": 0,
                          "ipopt.sb": "yes"})
    return opti, X, U, s0, goal


def _solve(controller, pose, previous=None):
    """One solve of a built controller from `pose`.

    Returns the command plan (2 by N), the predicted states (3 by N+1, or
    None) and whether the solver succeeded. If it did not, the previous plan
    is used, moved on by one step, as a real controller would.
    """
    opti, X, U, ps, pg = controller
    opti.set_value(ps, np.asarray(pose, float))
    opti.set_value(pg, np.array(GOAL))
    try:
        sol = opti.solve()
        return np.atleast_2d(sol.value(U)), np.array(sol.value(X)), True
    except RuntimeError:
        if previous is None:
            return np.zeros((2, U.shape[1])), None, False
        return np.hstack([previous[:, 1:], previous[:, -1:]]), None, False


# the plants
def run(build=None, plant=None, theta0=0.3, N=25, q=10.0, r=0.01, qN=100.0,
        margin=MARGIN, dt=0.1, T=40.0, quiet=False):
    """Run a controller on a plant. Returns a dict of everything worth plotting.

    build is the notebook's build() (the student's own controller); if it is
    left out, reference_build is used. plant is PlanningModel() or
    MuJoCoBase().
    """
    build = reference_build if build is None else build
    plant = PlanningModel() if plant is None else plant
    controller = build(N=N, dt=dt, w={"q": q, "r": r, "qN": qN}, margin=margin)
    pose = np.array([START[0], START[1], theta0]); plant.reset(pose)
    log = {k: [] for k in ["pose", "cmd", "t", "pred", "pred_u"]}
    goal = np.array(GOAL)
    plan = None
    for k in range(int(T / dt)):
        plan, X, ok = _solve(controller, pose, plan)     # the controller measures the pose
        u = plan[:, 0]
        log["pose"].append(pose.copy()); log["cmd"].append(u.copy())
        log["t"].append(k * dt); log["pred"].append(None if X is None else X.T.copy())
        log["pred_u"].append(plan.copy())
        controller[0].set_initial(controller[2], plan)   # start the next search here
        pose = plant.step(u[0], u[1], dt)
        if np.linalg.norm(pose[:2] - goal) < 0.02:
            break
    log["pose"].append(pose.copy())
    for key in ["pose", "cmd", "t"]:
        log[key] = np.array(log[key])
    P = log["pose"]
    log["error"] = np.linalg.norm(P[-1, :2] - goal)
    log["d_goal"] = np.linalg.norm(P[:, :2] - goal, axis=1)
    log["clearance"] = np.array([min(np.linalg.norm(p - np.array(o[:2])) - o[2] - R_ROBOT
                                     for o in circles()) for p in P[:, :2]])
    log["time"] = log["t"][-1] if len(log["t"]) else 0.0
    log["ring"] = log["clearance"] - margin
    log["reached"] = bool(log["error"] < 0.02)
    if not quiet:
        report(log)
    log["plant"] = getattr(plant, "name", "?")
    return log


# plotting
def draw_room(ax, label=True, obstacles=None, goal=None, margin=MARGIN):
    """The floor, the equipment and its keep-out rings, the start and the goal.

    obstacles and goal default to the ones in the workshop's scene; pass your
    own to draw a different one.
    """
    obstacles = OBSTACLES if obstacles is None else obstacles
    goal = GOAL if goal is None else goal
    # the floor area only: there are no walls in the problem, so none are drawn
    ax.set_xlim(ROOM[0] - 0.2, ROOM[1] + 0.2)
    ax.set_ylim(ROOM[2] - 0.2, ROOM[3] + 0.2)
    for cx, cy, rad, nm in obstacles:
        ax.add_patch(Circle((cx, cy), rad, fc="#c0392b", alpha=0.7))
        ax.add_patch(Circle((cx, cy), rad + R_ROBOT + margin, fc="none",
                            ec="#c0392b", ls=":", lw=1))
        if label:
            ax.text(cx, cy - rad - 0.18, nm, ha="center", fontsize=8, color="#7b241c")
    ax.plot(*START, "ko", ms=6); ax.plot(*goal, "gx", ms=13, mew=3)
    if label:
        ax.text(START[0], START[1] - 0.42, "start", fontsize=8, ha="center",
                va="top")
        ax.text(goal[0] - 0.05, goal[1] + 0.20, "goal", fontsize=8, color="#2e7d32",
                ha="center")
    ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")


def animate(log, step=2, dpi=90, fps=12, trim=2.0, frames_only=False):
    """Top view with the robot, its trail and the plan it is currently following.

    A clock and a speed readout are shown, so the pace of the run can be read
    off rather than guessed from how long the clip lasts.
    """
    P, PR, C, T = log["pose"], log["pred"], log["cmd"], log["t"]
    # If the robot stops early and simply sits there, cut the clip a couple of
    # seconds afterwards rather than filming a stationary robot.
    last = len(P) - 1
    if trim:
        moving = np.abs(C[:, 0]) > 0.01
        if moving.any():
            last = min(last, int(np.max(np.nonzero(moving))) + int(trim / 0.1))
    idx = list(range(0, max(last, 2), step))
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    draw_room(ax, label=False)
    trail, = ax.plot([], [], color="0.45", lw=1.6)
    plan, = ax.plot([], [], color="#1f77b4", lw=2, ls="dashed")
    arts = robot_artists(ax)
    ax.set_title("dashed line is the plan for the next 2.5 s")
    clock = ax.text(0.02, 0.97, "", transform=ax.transAxes, ha="left", va="top",
                    fontsize=11, family="monospace",
                    bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=3))

    def frame(i):
        j = idx[i]; x, y, th = P[j]
        trail.set_data(P[:j + 1, 0], P[:j + 1, 1])
        if PR[j] is not None:
            plan.set_data(PR[j][:, 0], PR[j][:, 1])
        place_robot(arts, x, y, th)
        jc = min(j, len(C) - 1)
        clock.set_text("t = %5.1f s\nv = %+5.2f m/s" % (T[jc], C[jc, 0]))
        return []

    if frames_only:
        return fig, frame, idx
    return _clip(fig, frame, len(idx), dpi=dpi, fps=fps, label="animation")


def _clip(fig, frame_fn, n, dpi=85, fps=12, label=""):
    """Render to an inline video with play, pause and a scrub bar."""
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    plt.rcParams["animation.embed_limit"] = 80
    anim = FuncAnimation(fig, frame_fn, frames=n, interval=1000 // fps)
    try:
        html = anim.to_html5_video(embed_limit=80)      # needs ffmpeg, has controls
        # matplotlib emits "controls autoplay loop"; drop both so the clip plays
        # once, on demand, and stops at the last frame
        html = html.replace(" controls autoplay loop>", " controls>")
    except Exception:
        html = anim.to_jshtml(fps=fps)                  # fallback, also has controls
    plt.close(fig)
    print("%s: %d frames, %.2f MB" % (label or "clip", n, len(html) / 1e6))
    return HTML(html)


def robot_artists(ax):
    """The robot drawn as a body, two wheels and a nose marker."""
    body = Circle((0, 0), R_ROBOT, fc="#2a5db0", ec="k", alpha=0.35, zorder=3)
    ax.add_patch(body)
    wheels = [Rectangle((0, 0), 2 * 0.065, 0.09, fc="k", zorder=6)
              for _ in range(2)]
    for w in wheels:
        ax.add_patch(w)
    nose, = ax.plot([], [], color="gold", lw=3, zorder=5)
    return body, wheels, nose


def place_robot(arts, x, y, th):
    body, wheels, nose = arts
    body.center = (x, y)
    nose.set_data([x, x + R_ROBOT * np.cos(th)], [y, y + R_ROBOT * np.sin(th)])
    for w, side in zip(wheels, (+1, -1)):
        cx = x - side * TRACK / 2 * np.sin(th)
        cy = y + side * TRACK / 2 * np.cos(th)
        w.set_xy((cx - 0.065 * np.cos(th) + 0.045 * np.sin(th),
                  cy - 0.065 * np.sin(th) - 0.045 * np.cos(th)))
        w.angle = np.degrees(th)


def plot_plan(build, pose, N=25, dt=0.1):
    """Solve the NMPC once and show everything that one solve produced.

    Left: the planned path. Right, in order: the commanded speed, the commanded
    turn rate, the predicted position and the predicted heading, all across the
    full horizon. The single command that is actually applied is marked in red.
    """
    v_max, w_max = V_MAX, W_MAX
    build = reference_build if build is None else build
    plan, X, ok = _solve(build(N=N, dt=dt), pose)
    if X is None:
        print("the solver found no plan from this pose")
        return None
    P = X.T
    t_state = np.arange(N + 1) * dt
    t_cmd = np.arange(N) * dt

    fig = plt.figure(figsize=(13.4, 6.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.30, 1], hspace=0.72, wspace=0.24,
                          left=0.05, right=0.98, top=0.92, bottom=0.09)

    ax = fig.add_subplot(gs[:, 0])
    draw_room(ax)
    place_robot(robot_artists(ax), *pose)
    ax.plot(P[:, 0], P[:, 1], color="#1f77b4", lw=2, ls="dashed", label="planned path, %.1f s" % (N * dt))
    ax.plot(P[::5, 0], P[::5, 1], "o", color="#1f77b4", ms=5)
    ax.plot(P[1, 0], P[1, 1], "o", color="#c0392b", ms=10, label="where one step lands")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("one solve: %d steps planned, %.1f s ahead" % (N, N * dt), fontsize=10)

    cmds = plan.T

    sp = fig.add_subplot(gs[0, 1])
    sp.step(t_cmd, cmds[:, 0], where="post", lw=1.8, color="#1f77b4")
    for y in (v_max, -v_max):
        sp.axhline(y, ls="dashed", c="r", lw=0.9)
    sp.plot(0, cmds[0, 0], "o", color="#c0392b", ms=8, zorder=5)
    sp.set_ylim(-1.15 * v_max, 1.15 * v_max)
    sp.set_title("commanded speed $v$ (m/s), limit $\\pm%.1f$" % v_max, fontsize=9)

    tr = fig.add_subplot(gs[1, 1])
    tr.step(t_cmd, cmds[:, 1], where="post", lw=1.8, color="#e07b39")
    for y in (w_max, -w_max):
        tr.axhline(y, ls="dashed", c="r", lw=0.9)
    tr.plot(0, cmds[0, 1], "o", color="#c0392b", ms=8, zorder=5)
    tr.set_ylim(-1.15 * w_max, 1.15 * w_max)
    tr.set_title("commanded turn rate $\\omega$ (rad/s), limit $\\pm%.1f$" % w_max, fontsize=9)

    ps = fig.add_subplot(gs[2, 1])
    ps.plot(t_state, P[:, 0], lw=1.7, label="$p_x$")
    ps.plot(t_state, P[:, 1], lw=1.7, label="$p_y$")
    ps.plot(dt, P[1, 0], "o", color="#c0392b", ms=8)
    ps.plot(dt, P[1, 1], "o", color="#c0392b", ms=8)
    ps.legend(fontsize=8, ncol=2, loc="upper left")
    ps.set_title("predicted position (m)", fontsize=9)

    th = fig.add_subplot(gs[3, 1])
    th.plot(t_state, P[:, 2], lw=1.7, color="#5b3a8e")
    th.plot(dt, P[1, 2], "o", color="#c0392b", ms=8)
    th.set_xlabel("time from now (s)", fontsize=9)
    th.set_title("predicted heading $\\theta$ (rad)", fontsize=9)

    for p in (sp, tr, ps, th):
        p.axvspan(0, dt, color="#c0392b", alpha=0.12, zorder=0)
        p.grid(alpha=0.25); p.tick_params(labelsize=8); p.set_xlim(0, N * dt)

    print("first command applied:  v = %+.3f m/s,  omega = %+.3f rad/s" % (plan[0, 0], plan[1, 0]))
    print("the other %d commands are computed, then discarded" % (N - 1))
    print("shaded band on the right = the one step that is actually taken")
    plt.show()


def plot_horizon_vs_obstacle(build=None, Ns=(4, 5, 25, 50), dt=0.1, plant="mujoco"):
    """Run the closed loop with several horizons and show where each one went.

    A short window cannot see past the machine base. Every short plan that
    goes round it first moves away from the goal, which the cost punishes,
    so the robot stops at the keep-out ring. A long enough window sees the
    way round. Beyond that, a longer window changes little but costs more
    computing time per solve.
    """
    import time
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    draw_room(ax)
    cols = ["#c0392b", "#2e7d32", "#1a5fb4", "#e07b39"]
    for N, c in zip(Ns, cols):
        t0 = time.perf_counter()
        vehicle = MuJoCoBase() if plant == "mujoco" else PlanningModel()
        log = run(build, vehicle, N=N, dt=dt, quiet=True)
        per = 1000 * (time.perf_counter() - t0) / max(1, len(log["t"]))
        P = log["pose"]
        if log["reached"]:
            note = "reached the goal in %.1f s" % log["time"]
        else:
            note = "stuck, %.1f m from the goal" % log["error"]
        ax.plot(P[:, 0], P[:, 1], color=c, lw=2.4 if N == Ns[0] else 2.0,
                ls="solid" if N != Ns[-1] else "dashed",
                label="N = %d (%.1f s ahead): %s, about %.0f ms per solve"
                      % (N, N * dt, note, per))
        ax.plot(P[-1, 0], P[-1, 1], "o", color=c, ms=8)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_ylim(-0.2, 5.4)
    ax.set_title("the same controller with %d window lengths, on %s"
                 % (len(Ns), "MuJoCo" if plant == "mujoco" else "the model"), fontsize=10)
    plt.tight_layout()
    plt.show()


def report(log):
    """Print the outcome of one run in plain words."""
    if log["reached"]:
        print("reached the goal (within 2 cm) after %.1f s, final error %.0f mm"
              % (log["time"], 1000 * log["error"]))
    else:
        print("did NOT reach the goal: stopped after %.1f s, still %.0f mm away"
              % (log["time"], 1000 * log["error"]))
    ring, touch = log["ring"].min(), log["clearance"].min()
    if abs(ring) < 5e-4:
        where = "exactly on the keep-out ring"
    else:
        where = "%.0f mm %s the keep-out ring" % (abs(1000 * ring),
                                                 "outside" if ring > 0 else "INSIDE")
    print("closest approach: %s, %.0f mm from touching" % (where, 1000 * touch))


def base_facts():
    """Parameters of the simulated chassis, read from the model itself."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(_xml())
    JT = {2: "slide", 3: "hinge"}
    joints = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
               JT.get(int(m.jnt_type[j]), "?"),
               np.round(m.jnt_axis[j], 2)) for j in range(m.njnt)]
    return dict(mass=float(mujoco.mj_getTotalmass(m)),
                joints=joints, nq=int(m.nq), nu=int(m.nu),
                bodies=int(m.nbody),
                footprint=float(2 * m.geom_size[:, 1].max()),
                deck=float(max(m.geom_pos[i][2] + m.geom_size[i][0]
                               for i in range(m.ngeom))))


# ================================================= the pieces, for reference
def ref_step(s, u, dt=0.1):
    """One step of the model."""
    s = np.asarray(s, float)
    return np.array([s[0] + dt * u[0] * np.cos(s[2]),
                     s[1] + dt * u[0] * np.sin(s[2]),
                     s[2] + dt * u[1]])


def ref_stage_cost(e, u, w):
    """Cost charged at one step of the horizon."""
    return w["q"] * float(np.dot(e, e)) + w["r"] * float(np.dot(u, u))


def ref_terminal_cost(e, w):
    """Cost charged at the last step of the horizon."""
    return w["qN"] * float(np.dot(e, e))


def ref_obstacle(p, centre, radius, margin=0.10):
    """Positive when the robot is clear of the obstacle, negative when not."""
    d = np.asarray(p, float) - np.asarray(centre, float)
    return float(np.dot(d, d) - (radius + R_ROBOT + margin) ** 2)


def ref_rollout(s0, U, dt=0.1):
    """Predicted states for a whole sequence of commands."""
    X = [np.asarray(s0, float)]
    for k in range(U.shape[1]):
        X.append(ref_step(X[-1], U[:, k], dt))
    return np.array(X)


def ref_total_cost(X, U, goal, w):
    """The cost of a whole plan."""
    N = U.shape[1]
    J = 0.0
    for k in range(N):
        J += ref_stage_cost(X[k, :2] - np.asarray(goal, float), U[:, k], w)
    J += ref_terminal_cost(X[N, :2] - np.asarray(goal, float), w)
    return J


WEIGHTS = {"q": 10.0, "r": 0.01, "qN": 100.0}


def _report(name, worst, tol=1e-9):
    ok = worst < tol
    print("%s: worst difference from the reference %.2e" % (name, worst))
    print("   %s" % ("PASS" if ok else "not correct yet"))
    return ok


def check_step(fn, tol=1e-9):
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(400):
        s = np.r_[rng.uniform(-5, 5, 2), rng.uniform(-np.pi, np.pi)]
        u = np.array([rng.uniform(-V_MAX, V_MAX), rng.uniform(-W_MAX, W_MAX)])
        got = np.asarray(fn(s, u, 0.1), float).ravel()
        if got.shape != (3,):
            print("expected three numbers, received shape", got.shape)
            return False
        worst = max(worst, np.abs(got - ref_step(s, u, 0.1)).max())
    return _report("model step", worst, tol)


def check_stage_cost(fn, tol=1e-9):
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(400):
        e, u = rng.normal(size=2), rng.normal(size=2)
        worst = max(worst, abs(float(fn(e, u, WEIGHTS))
                               - ref_stage_cost(e, u, WEIGHTS)))
    return _report("stage cost", worst, tol)


def check_terminal_cost(fn, tol=1e-9):
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(400):
        e = rng.normal(size=2)
        worst = max(worst, abs(float(fn(e, WEIGHTS))
                               - ref_terminal_cost(e, WEIGHTS)))
    return _report("terminal cost", worst, tol)


def check_obstacle(fn, tol=1e-9):
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(400):
        p = rng.uniform(0, 5, 2)
        cx, cy, rad, _ = OBSTACLES[rng.integers(len(OBSTACLES))]
        worst = max(worst, abs(float(fn(p, (cx, cy), rad, 0.10))
                               - ref_obstacle(p, (cx, cy), rad, 0.10)))
    return _report("obstacle constraint", worst, tol)


def check_rollout(fn, tol=1e-9):
    """X must be 3 rows by N+1 columns: one column per step."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(50):
        s0 = np.r_[rng.uniform(0, 4, 2), rng.uniform(-np.pi, np.pi)]
        U = np.vstack([rng.uniform(-V_MAX, V_MAX, 8),
                       rng.uniform(-W_MAX, W_MAX, 8)])
        got = np.asarray(ca.DM(fn(s0, U, 0.1)))
        if got.shape != (3, 9):
            print("expected a 3 by 9 matrix, received", got.shape)
            return False
        worst = max(worst, np.abs(got - ref_rollout(s0, U, 0.1).T).max())
    return _report("rollout", worst, tol)

def check_total_cost(fn, tol=1e-6):
    """X is 3 by N+1, U is 2 by N."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(50):
        s0 = np.r_[rng.uniform(0, 4, 2), rng.uniform(-np.pi, np.pi)]
        U = np.vstack([rng.uniform(-V_MAX, V_MAX, 8),
                       rng.uniform(-W_MAX, W_MAX, 8)])
        X = ref_rollout(s0, U, 0.1)
        g = np.array(GOAL, float)
        worst = max(worst, abs(float(ca.DM(fn(X.T, U, g, WEIGHTS)))
                               - ref_total_cost(X, U, g, WEIGHTS)))
    return _report("total cost", worst, tol)

def animate_mujoco(log, fps=12, step=2, width=768, height=528, at=None,
                   frames_only=False, caption=True):
    """Replay a run inside MuJoCo, so the simulated robot itself is seen."""
    from PIL import Image, ImageDraw
    import base64, io
    import imageio.v3 as iio
    from IPython.display import HTML
    import mujoco

    spec = mujoco.MjSpec.from_file(_xml())
    wb = spec.worldbody
    for k, (cx, cy, rad, _) in enumerate(OBSTACLES):
        b = wb.add_body(name="obs%d" % k, pos=[cx, cy, 0.35])
        g = b.add_geom()
        g.name = "obs%d_g" % k
        g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        g.size = [rad, 0.35, 0.0]
        g.rgba = [0.75, 0.23, 0.17, 1.0]
        g.contype, g.conaffinity = 0, 0
        r2 = wb.add_body(name="ring%d" % k, pos=[cx, cy, 0.008])
        rg = r2.add_geom()
        rg.name = "ring%d_g" % k
        rg.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        rg.size = [rad + R_ROBOT + 0.10, 0.002, 0.0]
        rg.rgba = [0.80, 0.35, 0.30, 0.30]
        rg.contype, rg.conaffinity = 0, 0
    tg = wb.add_body(name="goal", pos=[GOAL[0], GOAL[1], 0.10])
    t2 = tg.add_geom()
    t2.name = "goal_g"
    t2.type = mujoco.mjtGeom.mjGEOM_SPHERE
    t2.size = [0.14, 0, 0]
    t2.rgba = [0.05, 0.55, 0.20, 1.0]
    t2.contype, t2.conaffinity = 0, 0
    h = spec.visual.headlight
    h.diffuse = [0.85] * 3
    h.ambient = [0.40] * 3
    h.specular = [0.05] * 3
    for lt in spec.lights:
        lt.castshadow = False
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    m = spec.compile()

    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 55.0, -48.0, 8.6
    cam.lookat[:] = [2.5, 2.0, 0.2]
    rend = mujoco.Renderer(m, height, width)
    P, T = log["pose"], log["t"]
    frames = []
    try:
        for k in (range(0, len(P) - 1, step) if at is None else at):
            d.qpos[:3] = P[k]
            mujoco.mj_forward(m, d)
            rend.update_scene(d, cam)
            im = Image.fromarray(rend.render())
            if caption:
                dr = ImageDraw.Draw(im)
                dr.rectangle([0, 0, im.width, 52], fill=(255, 255, 255))
                dr.text((14, 8), "the robot inside MuJoCo      t = %.1f s"
                        % T[min(k, len(T) - 1)], fill=(15, 15, 15))
                dr.text((14, 28), "red = equipment    pink ring = keep-out    "
                                  "green = target", fill=(90, 90, 90))
            frames.append(np.array(im))
    finally:
        rend.close()
    if frames_only:
        return frames
    buf = io.BytesIO()
    iio.imwrite(buf, np.array(frames), extension=".mp4", fps=fps,
                codec="libx264")
    b64 = base64.b64encode(buf.getvalue()).decode()
    print("MuJoCo animation: %d frames, %.2f MB" % (len(frames), len(b64) / 1e6))
    return HTML('<video width="%d" controls>'
                '<source src="data:video/mp4;base64,%s" type="video/mp4">'
                '</video>' % (width, b64))


# rendering, made optional


animate_mujoco = _safe_render(animate_mujoco)


_base_facts_live = base_facts


def base_facts():
    """Parameters of the chassis. Falls back to stored values if MuJoCo
    cannot be imported, which happens when no graphics backend is present."""
    try:
        return _base_facts_live()
    except Exception:
        return dict(mass=60.0, nq=3, nu=3, bodies=2, footprint=0.567,
                    deck=0.335,
                    joints=[("joint_x", "slide", np.array([1., 0., 0.])),
                            ("joint_y", "slide", np.array([0., 1., 0.])),
                            ("joint_th", "hinge", np.array([0., 0., 1.]))])


def ring_words(gap):
    """Describe a distance to the keep-out ring in words, in metres."""
    if abs(gap) < 0.0005:
        return "exactly on the keep-out ring"
    return "%s the keep-out ring by %.3f m" % ("outside" if gap > 0 else "inside", abs(gap))


def plot_paths(results, title=""):
    """Several paths on one map.

    `results` maps a label to either a run log or a plain path: one row of
    x, y, theta per step, as your own loop collects.
    """
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    draw_room(ax)
    cols = ["#1a5fb4", "#c0392b", "#2e7d32", "#e07b39"]
    styles = ["dashed", "solid", "solid", "solid"]
    for (name, r), c, ls in zip(results.items(), cols, styles):
        P = np.asarray(r["pose"] if isinstance(r, dict) else r, float)
        ax.plot(P[:, 0], P[:, 1], color=c, lw=2.2, ls=ls, label=name)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.show()


def path_report(path, goal=None, obstacles=None, margin=MARGIN):
    """Say where a path ended and how close it came, without passing judgement.

    goal and obstacles default to the workshop's scene; pass your own if you
    have changed them.
    """
    path = np.asarray(path, float)
    goal = GOAL if goal is None else goal
    obstacles = OBSTACLES if obstacles is None else obstacles
    err = float(np.linalg.norm(path[-1, :2] - np.array(goal)))
    ring = min((np.hypot(path[:, 0] - c[0], path[:, 1] - c[1])
                - (c[2] + R_ROBOT + margin)).min() for c in obstacles)
    touch = min((np.hypot(path[:, 0] - c[0], path[:, 1] - c[1])
                 - (c[2] + R_ROBOT)).min() for c in obstacles)
    print("ended %.0f mm from the goal" % (1000 * err))
    print("closest approach: %s, %.0f mm from touching"
          % (ring_words(ring), 1000 * touch))


def check_loop(path, tol_goal=0.02):
    """The closed loop in Task 5: did it arrive, and did it keep out?"""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 3:
        print("expected one row of three numbers per step, received shape", path.shape)
        return False
    err = float(np.linalg.norm(path[-1, :2] - np.array(GOAL)))
    ring = min((np.hypot(path[:, 0] - c[0], path[:, 1] - c[1])
                - (c[2] + R_ROBOT + MARGIN)).min() for c in OBSTACLES)
    print("ended %.1f mm from the goal; closest approach: %s"
          % (1000 * err, ring_words(ring)))
    ok = err < tol_goal and ring > -1e-3
    print("   PASS" if ok else "   not yet: the robot should end within 2 cm and "
          "never enter a ring")
    return ok


for _name in [n for n in list(globals())
              if n.startswith("check_") and n != "check_against_mujoco"]:
    globals()[_name] = _friendly_check(globals()[_name])


def preview(render=False):
    """The finished controller, shown before you build it: MuJoCo beside Python.

    Plays the stored video in videos/ if it can be found. render=True makes a
    fresh one instead, which takes about 15 seconds.
    """
    name = "robot_preview.mp4"
    stored = None if render else _stored_video(name)
    if stored:
        return _video_html(open(stored, "rb").read())
    print("making the video, about 15 seconds ...")
    fetch_model(quiet=True)            # the MuJoCo model, downloaded once
    log = run(plant=MuJoCoBase(), quiet=True)
    fig, frame, idx = animate(log, step=1, frames_only=True)
    left = animate_mujoco(log, at=idx, frames_only=True, width=640, height=480)
    data = _stitch(left, _figure_frames(fig, frame, len(idx)), 10,
                   "Build a controller similar to this: it drives the robot to "
                   "the goal and keeps it clear of the equipment",
                   "left: the robot, simulated in MuJoCo",
                   "right: the controller's view; the dashed line is its current plan")
    return _video_html(data)


_TASK_CHECKS = {
    "my_step": ("check_step", "Task 1"),
    "my_stage_cost": ("check_stage_cost", "Task 2"),
    "my_terminal_cost": ("check_terminal_cost", "Task 2"),
    "my_total_cost": ("check_total_cost", "Task 2"),
    "my_obstacle": ("check_obstacle", "Task 3"),
    "my_rollout": ("check_rollout", "Task 4"),
}


# used by the notebooks as lab.is_path, lab.stop_unless, lab.video, lab.NotYet
NotYet, is_path, stop_unless, video = NotYet, is_path, stop_unless, video


def require_finished(*functions):
    """Stop with a plain message if any of these task functions fails its check."""
    require_finished_in(globals(), _TASK_CHECKS, *functions)




def filmstrip(log, shots=4, width=420, height=300):
    """A row of MuJoCo pictures across a run, so the robot itself can be seen."""
    moving = np.abs(np.asarray(log["cmd"])[:, 0]) > 0.02
    last = int(np.nonzero(moving)[0][-1]) if moving.any() else len(log["pose"]) - 2
    n = min(last + 5, len(log["pose"]) - 2)          # a moment after it settles
    at = [int(round(k * n / (shots - 1))) for k in range(shots)]
    frames = animate_mujoco(log, at=at, frames_only=True, width=width,
                            height=height, caption=False)
    if frames is None:                       # no 3-D rendering on this machine
        return
    fig, axes = plt.subplots(1, len(frames), figsize=(3.4 * len(frames), 2.7))
    for ax, frame, k in zip(np.atleast_1d(axes), frames, at):
        ax.imshow(frame); ax.axis("off")
        ax.set_title("t = %.1f s" % (k * 0.1), fontsize=9)
    plt.tight_layout()
    plt.show()


def compare_runs(log, reference=None, labels=("this run", "the first run"),
                 title=""):
    """The path, the commands and the distance to the goal, against a first run."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.4),
                                     gridspec_kw={"width_ratios": [1.15, 1, 1]})
    for lg, colour, name, wide in ((reference, "0.62", labels[1], 3.4),
                                   (log, "#c0392b", labels[0], 2.2)):
        if lg is None:
            continue
        P, C, T = lg["pose"], lg["cmd"], np.asarray(lg["t"], float)
        a1.plot(P[:, 0], P[:, 1], lw=wide, color=colour, label=name)
        a2.plot(T, C[:, 0], lw=wide, color=colour, label=name + ": speed")
        a2.plot(T, C[:, 1], lw=wide, color=colour, ls=(0, (4, 2)),
                label=name + ": turn rate")
        gap = np.linalg.norm(P[:len(T), :2] - np.array(GOAL), axis=1)
        a3.plot(T, gap, lw=wide, color=colour, label=name)
    draw_room(a1, label=False)
    a1.set_title("where it went", fontsize=10)
    a1.legend(loc="lower right", fontsize=8)

    for limit, style in ((V_MAX, "-"), (W_MAX, "-")):
        for sign in (1, -1):
            a2.axhline(sign * limit, color="#c0392b", lw=1, ls=(0, (2, 3)))
    a2.set_xlabel("time (s)"); a2.set_ylabel("command")
    a2.set_title("what it was told to do (limits dotted)", fontsize=10)
    a2.legend(fontsize=8, loc="lower right"); a2.grid(alpha=0.25)

    a3.axhline(0.02, color="#2e7d32", lw=1, ls=(0, (2, 3)))
    a3.text(0.02, 0.024, "at the goal", fontsize=8, color="#2e7d32")
    a3.set_xlabel("time (s)"); a3.set_ylabel("distance to the goal (m)")
    a3.set_ylim(0, None); a3.set_title("how far from the goal", fontsize=10)
    a3.legend(fontsize=8); a3.grid(alpha=0.25, which="both")

    # stop the time axes shortly after the last run settles, not at the full 40 s
    ends = []
    for lg in (reference, log):
        if lg is None:
            continue
        moving = np.abs(np.asarray(lg["cmd"])[:, 0]) > 0.02
        T = np.asarray(lg["t"], float)
        ends.append(T[np.nonzero(moving)[0][-1]] if moving.any() else T[-1])
    stop = min(max(ends) + 1.5, float(np.asarray(log["t"])[-1]))
    a2.set_xlim(0, stop); a3.set_xlim(0, stop)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()
