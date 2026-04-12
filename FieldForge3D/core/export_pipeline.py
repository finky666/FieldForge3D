from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyvista as pv
from PIL import Image

from core.plugins import PluginManager, PluginInfo

DEFAULT_EXPORT_STATE = {
    "DARK_BG": True,
    "SMOOTH_SHADING": True,
    "COLOR_MODE": "radius",
    "COLORMAP_DARK": "turbo",
    "COLORMAP_LIGHT": "viridis",
    "SHOW_SCALAR_BAR": False,
}

DEFAULT_BATCH_PARAMS = {
    "N": 180,
    "BOUNDS": 1.5,
    "ISO": 0.55,
}

DEFAULT_PRINTABLE_IDS = {
    "torus",
    "superquadric",
    "heart_implicit",
    "metaballs",
    "trefoil_knot",
    "dancing_eggs",
    "organic_blob",
    "ripple_shell",
}


@dataclass
class ExportRecord:
    plugin_id: str
    plugin_name: str
    image_path: str | None = None
    stl_path: str | None = None
    skipped: str | None = None


def _safe_slug(text: str) -> str:
    chars = []
    for ch in (text or ""):
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch.lower())
        elif ch in (" ", "/"):
            chars.append("_")
    slug = "".join(chars).strip("_")
    return slug or "item"


def build_mesh_from_field(field, params: dict) -> pv.PolyData:
    n = int(params["N"])
    bounds = float(params["BOUNDS"])
    iso = float(params["ISO"])

    spacing = (2 * bounds / (n - 1), 2 * bounds / (n - 1), 2 * bounds / (n - 1))
    origin = (-bounds, -bounds, -bounds)

    grid = pv.ImageData(dimensions=(n, n, n), spacing=spacing, origin=origin)
    arr = np.asarray(field, dtype=np.float32)
    grid["value"] = arr.ravel(order="F")

    surf = grid.contour(
        isosurfaces=[iso],
        scalars="value",
        compute_normals=False,
        compute_gradients=False,
    )
    surf = surf.clean()

    if surf.n_points > 0:
        if "value" not in surf.array_names:
            surf["value"] = np.zeros(surf.n_points, dtype=np.float32)
        surf["height"] = surf.points[:, 2].astype(np.float32)
        surf["radius"] = np.linalg.norm(surf.points, axis=1).astype(np.float32)

    return surf


def _mesh_scalars(mesh: pv.PolyData, color_mode: str) -> str:
    return {"value": "value", "height": "height", "radius": "radius"}.get(color_mode, "value")


def render_mesh_image(
    mesh: pv.PolyData,
    output_path: str | Path,
    *,
    state: dict | None = None,
    window_size: tuple[int, int] = (1280, 960),
) -> None:
    state = {**DEFAULT_EXPORT_STATE, **(state or {})}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("black" if state["DARK_BG"] else "white")

    scalars = _mesh_scalars(mesh, str(state.get("COLOR_MODE", "radius")))
    cmap = state["COLORMAP_DARK"] if state["DARK_BG"] else state["COLORMAP_LIGHT"]

    plotter.add_mesh(
        mesh,
        scalars=scalars,
        cmap=cmap,
        show_scalar_bar=bool(state.get("SHOW_SCALAR_BAR", False)),
        smooth_shading=bool(state.get("SMOOTH_SHADING", True)),
        specular=0.55,
        specular_power=55,
        diffuse=0.8,
        ambient=0.2,
    )

    key = pv.Light(position=(3, 2, 4), focal_point=(0, 0, 0), intensity=1.0, light_type="scene light")
    fill = pv.Light(position=(-3, -2, 2), focal_point=(0, 0, 0), intensity=0.55, light_type="scene light")
    rim = pv.Light(position=(0, 5, -2), focal_point=(0, 0, 0), intensity=0.35, light_type="scene light")
    plotter.add_light(key)
    plotter.add_light(fill)
    plotter.add_light(rim)
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.screenshot(str(output_path))
    plotter.close()


def save_gif_from_images(
    image_paths: Iterable[str | Path],
    output_gif: str | Path,
    *,
    duration_ms: int = 700,
    loop: int = 0,
) -> None:
    paths = [Path(p) for p in image_paths if Path(p).exists()]
    if not paths:
        return
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
    output_gif = Path(output_gif)
    output_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )


def scale_mesh_to_max_size(mesh: pv.PolyData, max_size_mm: float = 100.0) -> pv.PolyData:
    mesh = mesh.copy(deep=True)
    bounds = mesh.bounds
    size_x = abs(bounds[1] - bounds[0])
    size_y = abs(bounds[3] - bounds[2])
    size_z = abs(bounds[5] - bounds[4])
    max_size = max(size_x, size_y, size_z, 1e-9)
    scale = float(max_size_mm) / max_size
    mesh.scale([scale, scale, scale], inplace=True)
    mesh.translate(-np.array(mesh.center), inplace=True)
    return mesh


def export_stl(mesh: pv.PolyData, output_path: str | Path, max_size_mm: float = 100.0) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = scale_mesh_to_max_size(mesh, max_size_mm=max_size_mm)
    out = out.extract_surface().triangulate().clean()
    out.save(str(output_path))
    return output_path


def plugin_export_params(info: PluginInfo, *, preview_n: int = 180) -> dict:
    params = dict(DEFAULT_BATCH_PARAMS)
    defaults = dict(info.module.get_defaults() or {})
    params.update(defaults)
    params["N"] = min(int(params.get("N", preview_n)), int(preview_n))
    params.setdefault("BOUNDS", DEFAULT_BATCH_PARAMS["BOUNDS"])
    params.setdefault("ISO", DEFAULT_BATCH_PARAMS["ISO"])
    return params


def is_printable_plugin(info: PluginInfo) -> bool:
    meta = getattr(info.module, "PLUGIN_META", {}) or {}
    if "printable" in meta:
        return bool(meta.get("printable"))
    return info.id in DEFAULT_PRINTABLE_IDS


def is_hidden_plugin(info: PluginInfo) -> bool:
    meta = getattr(info.module, "PLUGIN_META", {}) or {}
    return bool(meta.get("hidden", False))


def export_all_pack(
    plugins_dir: str | Path,
    output_dir: str | Path,
    *,
    include_hidden: bool = False,
    preview_n: int = 180,
    max_size_mm: float = 100.0,
    state: dict | None = None,
) -> dict:
    pm = PluginManager(Path(plugins_dir))
    pm.scan()

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    print_dir = output_dir / "print"
    manifest_path = output_dir / "manifest.json"
    gif_path = output_dir / "gallery.gif"

    images_dir.mkdir(parents=True, exist_ok=True)
    print_dir.mkdir(parents=True, exist_ok=True)

    state = {**DEFAULT_EXPORT_STATE, **(state or {})}
    records: list[ExportRecord] = []
    image_paths: list[Path] = []

    for info in pm.list():
        if is_hidden_plugin(info) and not include_hidden:
            continue

        slug = _safe_slug(info.id)
        params = plugin_export_params(info, preview_n=preview_n)

        try:
            field = info.module.compute(params)
            mesh = build_mesh_from_field(field, params)
        except Exception as exc:
            records.append(ExportRecord(plugin_id=info.id, plugin_name=info.name, skipped=f"compute failed: {exc!r}"))
            continue

        if mesh.n_points == 0 or mesh.n_cells == 0:
            records.append(ExportRecord(plugin_id=info.id, plugin_name=info.name, skipped="empty surface"))
            continue

        image_path = images_dir / f"{slug}.png"
        try:
            render_mesh_image(mesh, image_path, state=state)
            image_paths.append(image_path)
        except Exception as exc:
            records.append(ExportRecord(plugin_id=info.id, plugin_name=info.name, skipped=f"render failed: {exc!r}"))
            continue

        stl_path = None
        if is_printable_plugin(info):
            try:
                stl_path = export_stl(mesh, print_dir / f"{slug}.stl", max_size_mm=max_size_mm)
            except Exception:
                stl_path = None

        records.append(
            ExportRecord(
                plugin_id=info.id,
                plugin_name=info.name,
                image_path=str(image_path.relative_to(output_dir)),
                stl_path=str(stl_path.relative_to(output_dir)) if stl_path else None,
            )
        )

    if image_paths:
        save_gif_from_images(image_paths, gif_path, duration_ms=900)

    manifest = {
        "version": "1.0.0",
        "images_dir": str(images_dir.relative_to(output_dir)),
        "gif": str(gif_path.relative_to(output_dir)) if gif_path.exists() else None,
        "records": [record.__dict__ for record in records],
        "load_errors": pm.load_errors,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
