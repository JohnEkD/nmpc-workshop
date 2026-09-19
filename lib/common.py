"""Code shared by robot_lab.py and drone_lab.py.

Importing this module first sets up MuJoCo's off-screen 3-D rendering on
Colab, which must happen before MuJoCo itself is imported. It also holds the
plain "not yet" messages, the video helpers and the check wrapper.

Students never need to read or change this file.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


def _prepare_rendering(force=False):
    """Pick an off-screen renderer that works, before MuJoCo is imported.

    Only needed on Colab (or when force=True). Tries the fast GPU route (EGL)
    first, then the software route (OSMesa), and keeps the first that can
    draw a tiny test scene.
    """
    if not (force or os.path.isdir("/content")):
        return os.environ.get("MUJOCO_GL")
    if not os.environ.get("_LAB_GL_READY"):
        import subprocess
        subprocess.run("apt-get -qq install -y libosmesa6 libgl1-mesa-glx "
                       "> /dev/null 2>&1 || true", shell=True)
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            import mujoco
            m = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><body><geom size='.1'/></body>"
                "</worldbody></mujoco>")
            mujoco.Renderer(m, 64, 64).close()
            os.environ["_LAB_GL_READY"] = backend
            return backend
        except Exception:
            for k in list(sys.modules):
                if k.startswith("mujoco"):
                    del sys.modules[k]
    return None


_prepare_rendering()


class NotYet(Exception):
    """Stops a cell with one plain sentence and no error listing."""

    def _render_traceback_(self):
        return []


def stop_unless(condition, message):
    """Stop this cell with a plain message unless the condition holds."""
    if not condition:
        print("not yet: " + message)
        raise NotYet(message)


def is_path(path):
    """True when path is a finished result: one row of numbers per step."""
    return (isinstance(path, np.ndarray) and path.ndim == 2 and len(path) > 1
            and np.issubdtype(path.dtype, np.number))


def _friendly_check(check):
    """Turn the error from an unfinished task into one plain sentence."""
    import functools

    @functools.wraps(check)
    def wrapper(*args, **kwargs):
        try:
            result = check(*args, **kwargs)
            # a pass or fail is already printed; returning it would also show
            # a stray True or False under the cell
            return None if isinstance(result, (bool, np.bool_)) else result
        except NotImplementedError:
            print("not yet: this task, or an earlier task it uses, has not been "
                  "written. Replace its `raise NotImplementedError` line, then run "
                  "this cell again.")
        except TypeError as err:
            if "ellipsis" not in str(err):
                raise
            print("not yet: a blank (...) is still in this task, or in an earlier "
                  "task it uses. Fill it in, then run this cell again.")
        return None
    return wrapper


def _safe_render(fn):
    """3-D rendering can be unavailable. Say so and carry on."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as err:
            print("3-D rendering is not available on this machine (%s)."
                  % type(err).__name__)
            print("Everything else in the notebook works. Try setting")
            print('    import os; os.environ["MUJOCO_GL"] = "osmesa"')
            print("in the first cell, then restarting the runtime.")
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn          # the original, for working out what failed
    return wrapper


# where a missing video can be fetched from: the exercises live in two
# repositories, and the same library serves both
VIDEO_URLS = ("https://raw.githubusercontent.com/JohnEkD/nmpc-intro/main/videos/",
              "https://raw.githubusercontent.com/JohnEkD/nmpc-workshop/main/videos/")


def _stitch(left, right, fps, headline, left_note, right_note):
    """Join two lists of frames side by side, under a heading. Returns mp4 bytes."""
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import font_manager
    import io
    import imageio.v3 as iio

    try:
        path = font_manager.findfont("DejaVu Sans")
        big, small = ImageFont.truetype(path, 22), ImageFont.truetype(path, 16)
    except Exception:
        big = small = ImageFont.load_default()
    rows = []
    for i in range(len(right)):
        rt = Image.fromarray(np.asarray(right[i])[..., :3])
        h = rt.height
        if left is not None:
            lf = Image.fromarray(np.asarray(left[i])[..., :3])
            lf = lf.resize((int(lf.width * h / lf.height), h))
            body = Image.new("RGB", (lf.width + rt.width, h), "white")
            body.paste(lf, (0, 0)); body.paste(rt, (lf.width, 0))
        else:
            body = rt
        top = 78
        W = body.width + (-body.width) % 16
        H = h + top + (-(h + top)) % 16
        im = Image.new("RGB", (W, H), "white")
        im.paste(body, (0, top))
        dr = ImageDraw.Draw(im)
        dr.text((14, 10), headline, fill=(15, 15, 15), font=big)
        if left is not None:
            dr.text((14, 46), left_note, fill=(70, 70, 70), font=small)
            dr.text((body.width - rt.width + 14, 46), right_note,
                    fill=(70, 70, 70), font=small)
        else:
            dr.text((14, 46), right_note + "   (3-D view not available here)",
                    fill=(70, 70, 70), font=small)
        rows.append(np.array(im))
    buf = io.BytesIO()
    iio.imwrite(buf, np.array(rows), extension=".mp4", fps=fps, codec="libx264")
    return buf.getvalue()


def _video_html(data):
    """Show mp4 bytes as a video player, at the video's own size."""
    import base64
    from IPython.display import HTML
    b64 = base64.b64encode(data).decode()
    return HTML('<video style="max-width:100%%" controls>'
                '<source src="data:video/mp4;base64,%s" type="video/mp4">'
                '</video>' % b64)


def _figure_frames(fig, frame, n):
    """Draw a matplotlib animation frame by frame into image arrays."""
    out = []
    for i in range(n):
        frame(i)
        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return out


def project_root():
    """The folder that holds the notebooks: the one above lib/."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stored_video(name):
    """A pre-made video in videos/, if it can be found or downloaded."""
    import urllib.request
    for base in (project_root(), os.getcwd(), "/content"):
        p = os.path.join(base, "videos", name)
        if os.path.exists(p):
            return p
    folder = os.path.join(project_root(), "videos")
    for url in VIDEO_URLS:
        try:
            os.makedirs(folder, exist_ok=True)
            p = os.path.join(folder, name)
            urllib.request.urlretrieve(url + name, p)
            if os.path.getsize(p) > 1000:
                return p
            os.remove(p)
        except Exception:
            continue
    return None


def video(name):
    """Play one of the workshop's pre-made videos, for example video("bowl_free")."""
    path = _stored_video(name + ".mp4")
    if path is None:
        print("the video %s.mp4 was not found. Put the videos folder beside this "
              "notebook, then run this cell again." % name)
        return None
    with open(path, "rb") as fh:
        return _video_html(fh.read())


def require_finished_in(namespace, table, *functions):
    """Stop with a plain message if any of these task functions fails its check.

    namespace is the calling library's globals(), where the checks live, and
    table maps each task function's name to (check name, where to find it).
    """
    import contextlib
    import io
    for fn in functions:
        check, where = table[fn.__name__]
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                namespace[check](fn)
            ok = "PASS" in out.getvalue()
        except Exception:
            ok = False
        stop_unless(ok, "%s is not finished (%s). Finish it so that its check "
                        "prints PASS, then run this cell again." % (where, fn.__name__))
