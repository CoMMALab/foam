"""Viser helpers shared by the visualization scripts."""
from pathlib import Path

import matplotlib as mpl
import numpy as np
import viser
import viser.transforms as vtf
import yourdfpy
from viser.extras import ViserUrdf

from .model import Sphere


def load_robot(urdf_path: Path, collision: bool = False) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        str(urdf_path),
        load_collision_meshes = collision,
        build_collision_scene_graph = collision,
        )


def make_server(port: int = 8080) -> viser.ViserServer:
    server = viser.ViserServer(port = port)
    server.scene.add_grid("/ground", width = 2, height = 2, cell_size = 0.1)
    return server


def add_joint_sliders(server: viser.ViserServer, urdf: yourdfpy.URDF) -> list[viser.GuiInputHandle]:
    sliders = []
    with server.gui.add_folder("Joints"):
        for name in urdf.actuated_joint_names:
            limit = urdf.joint_map[name].limit
            lower = limit.lower if limit is not None and limit.lower is not None else -np.pi
            upper = limit.upper if limit is not None and limit.upper is not None else np.pi
            sliders.append(
                server.gui.add_slider(
                    name, min = lower, max = upper, step = 1e-3, initial_value = (lower + upper) / 2
                    )
                )

    initial = [s.value for s in sliders]
    reset = server.gui.add_button("Reset joints")

    @reset.on_click
    def _(_):
        for s, value in zip(sliders, initial):
            s.value = value

    return sliders


def link_color(i: int) -> tuple[float, float, float]:
    return mpl.colormaps['tab10'](i % 10)[:3]


class LinkSpheres:
    """Spheres parented under one frame per link; `update` moves the frames with FK."""

    def __init__(self, server: viser.ViserServer, urdf: yourdfpy.URDF, root: str = "/spheres"):
        self.server = server
        self.urdf = urdf
        self.root = root
        self.frames: dict[str, viser.FrameHandle] = {}
        self.handles: dict[tuple[str, int], viser.IcosphereHandle] = {}
        self.colors = {link: link_color(i) for i, link in enumerate(urdf.link_map)}

    def set(self, link_spheres: dict[str, list[Sphere]], opacity: float = 0.7, visible: bool = True):
        self.clear()
        for link, spheres in link_spheres.items():
            self.frames[link] = self.server.scene.add_frame(f"{self.root}/{link}", show_axes = False)
            for i, sphere in enumerate(spheres):
                self.add(link, i, sphere, opacity = opacity, visible = visible)

        self.update(self.urdf.cfg)

    def add(self, link: str, i: int, sphere: Sphere, color = None, opacity: float = 0.7, visible: bool = True):
        self.handles[(link, i)] = self.server.scene.add_icosphere(
            f"{self.root}/{link}/{i}",
            radius = float(sphere.radius),
            position = tuple(sphere.origin),
            color = color or self.colors[link],
            opacity = opacity,
            visible = visible,
            )
        return self.handles[(link, i)]

    def clear(self):
        for frame in self.frames.values():
            frame.remove()

        self.frames.clear()
        self.handles.clear()

    def update(self, cfg: np.ndarray):
        self.urdf.update_cfg(cfg)
        for link, frame in self.frames.items():
            T = self.urdf.get_transform(link)
            frame.wxyz = vtf.SO3.from_matrix(T[:3, :3]).wxyz
            frame.position = T[:3, 3]


__all__ = ['ViserUrdf', 'load_robot', 'make_server', 'add_joint_sliders', 'link_color', 'LinkSpheres']
