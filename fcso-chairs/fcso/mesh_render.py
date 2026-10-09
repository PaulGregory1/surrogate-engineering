"""Marching cubes on decoded logit grids and offscreen PyVista renders with one fixed camera.

All renders share CAMERA (3/4 view, y up, white background) and never reset the camera per shape, so
sizes are comparable across panels, methods and milestones.
"""
import numpy as np
import pyvista as pv
import trimesh
from skimage.measure import marching_cubes

pv.OFF_SCREEN = True

BOUND = 1.0
# 3/4 front view: chairs are y-up, backrest at +z, so they face -z (checked on the surface clouds).
CAMERA = [(2.6, 1.6, -3.4), (0.0, -0.05, 0.0), (0.0, 1.0, 0.0)]
PARALLEL_SCALE = 1.3  # orthographic half-height in world units; [-1, 1]^3 fits with a margin
MESH_COLOR = "#c8a27a"
STRAY_COLOR = "#d62728"  # connected components other than the largest (floating fragments)
POINT_COLOR = "#4a6fa5"


def logits_to_mesh(vol, bound=BOUND, level=0.0):
    """vol: (R, R, R) logits indexed [x, y, z] on linspace(-bound, bound, R) (prior.make_grid order).
    Returns a trimesh.Trimesh with every connected component kept, or None if the level set is empty."""
    vol = np.asarray(vol, dtype=np.float32)
    if not (vol.min() < level < vol.max()):
        return None
    R = vol.shape[0]
    h = 2.0 * bound / (R - 1)
    verts, faces, _, _ = marching_cubes(vol, level=level, spacing=(h, h, h))
    verts -= bound
    m = trimesh.Trimesh(verts, faces, process=False)
    if m.volume < 0:  # orient faces outward (logits are positive inside) so trimesh volume is positive
        m.invert()
    return m


def _to_pv(obj):
    if isinstance(obj, trimesh.Trimesh):
        faces = np.hstack([np.full((len(obj.faces), 1), 3), obj.faces]).astype(np.int64)
        return pv.PolyData(np.asarray(obj.vertices, dtype=np.float32), faces.ravel()), "mesh"
    pts = np.asarray(obj, dtype=np.float32)
    return pv.PolyData(pts), "points"


def _add(pl, obj, max_points=30000, seed=0, color_strays=False, empty_text="(empty)"):
    if obj is None:
        if empty_text:
            pl.add_text(empty_text, position="upper_right", font_size=9, color="red")
        return
    if not isinstance(obj, trimesh.Trimesh) and len(obj) > max_points:
        obj = np.asarray(obj)[np.random.default_rng(seed).choice(len(obj), max_points, replace=False)]
    if color_strays and isinstance(obj, trimesh.Trimesh):
        comps = obj.split(only_watertight=False)
        if len(comps) > 1:
            comps = sorted(comps, key=lambda c: abs(float(c.volume)), reverse=True)
            for i, c in enumerate(comps):
                pl.add_mesh(_to_pv(c)[0], color=MESH_COLOR if i == 0 else STRAY_COLOR, smooth_shading=True,
                            specular=0.2)
            return
    data, kind = _to_pv(obj)
    if kind == "mesh":
        pl.add_mesh(data, color=MESH_COLOR, smooth_shading=True, specular=0.2)
    else:
        pl.add_points(data, color=POINT_COLOR, point_size=3.0, render_points_as_spheres=True)


def _set_camera(pl):
    pl.camera_position = CAMERA
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = PARALLEL_SCALE


def render_grid(objs, path, labels=None, ncols=None, panel=(400, 400), color_strays=False, empty_text="(empty)"):
    """Render objs (trimesh meshes, (N, 3) point arrays or None) in a grid of panels with the fixed camera,
    one optional text label per panel, and save a PNG to path. Row-major order. color_strays: draw every
    connected component but the largest in STRAY_COLOR."""
    n = len(objs)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols
    pl = pv.Plotter(off_screen=True, shape=(nrows, ncols), window_size=(panel[0] * ncols, panel[1] * nrows),
                    border=False)
    pl.set_background("white")
    for i, obj in enumerate(objs):
        pl.subplot(i // ncols, i % ncols)
        _add(pl, obj, color_strays=color_strays, empty_text=empty_text)
        if labels is not None and labels[i]:
            pl.add_text(str(labels[i]), position="upper_left", font_size=9, color="black")
        _set_camera(pl)
    pl.screenshot(path)
    pl.close()
    return path


def render(obj, path, label=None, size=(600, 600)):
    """Single-panel render of one mesh or point cloud."""
    return render_grid([obj], path, labels=[label] if label else None, ncols=1, panel=size)
