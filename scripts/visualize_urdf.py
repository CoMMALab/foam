import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import viser
import yourdfpy
from viser import transforms as tf
from viser.extras import ViserUrdf
from foam.utility import load_srdf, _urdf_clean_filename


def _load_srdf_spheres(srdf_path: Path) -> dict[str, list[tuple[np.ndarray, float]]]:
    spheres_by_link: dict[str, list[tuple[np.ndarray, float]]] = {}
    if not srdf_path.exists():
        return spheres_by_link

    srdf = load_srdf(srdf_path)
    approximations = srdf["robot"].get("link_sphere_approximation")
    if approximations is None:
        return spheres_by_link

    if not isinstance(approximations, list):
        approximations = [approximations]

    for approximation in approximations:
        link_name = approximation.get("@link")
        if not link_name:
            continue

        spheres: list[tuple[np.ndarray, float]] = []
        sphere_entries = approximation.get("sphere", [])
        if not isinstance(sphere_entries, list):
            sphere_entries = [sphere_entries]

        for sphere in sphere_entries:
            center = sphere.get("@center")
            radius = sphere.get("@radius")
            if center is None or radius is None:
                continue

            xyz = np.fromiter((float(v) for v in center.split()), dtype=float)
            if xyz.shape[0] != 3:
                continue
            spheres.append((xyz, float(radius)))

        if spheres:
            spheres_by_link[link_name] = spheres

    return spheres_by_link


def _origin_to_matrix(origin: np.ndarray | None) -> np.ndarray:
    if origin is None:
        return np.eye(4)
    return origin.copy()


def _wxyz_from_matrix(transform: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in tf.SO3.from_matrix(transform[:3, :3]).wxyz)


def _xyz_from_matrix(transform: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in transform[:3, 3])


@dataclass
class CollisionMarker:
    handle: object
    link_name: str
    local_tf: np.ndarray
    source: str


def _make_collision_mesh(urdf_dir: Path, geom: yourdfpy.Geometry) -> trimesh.Trimesh | None:
    if geom.mesh is not None:
        mesh_filename = _urdf_clean_filename(geom.mesh.filename)
        mesh_path = urdf_dir / mesh_filename
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]  # type: ignore[attr-defined]
            )
        if geom.mesh.scale is not None:
            mesh = mesh.copy()
            mesh.apply_scale(np.asarray(geom.mesh.scale, dtype=float))
        return mesh

    if geom.box is not None:
        size = np.asarray(geom.box.size, dtype=float)
        return trimesh.creation.box(extents=size)

    if geom.sphere is not None:
        return trimesh.creation.icosphere(radius=float(geom.sphere.radius), subdivisions=2)

    if geom.cylinder is not None:
        cyl = trimesh.creation.cylinder(
            radius=float(geom.cylinder.radius),
            height=float(geom.cylinder.length),
            sections=32,
        )
        rot = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
        cyl.apply_transform(rot)
        return cyl

    return None


def _build_collision_markers(
    server: viser.ViserServer,
    urdf: yourdfpy.URDF,
    urdf_path: Path,
    srdf_spheres: dict[str, list[tuple[np.ndarray, float]]],
) -> list[CollisionMarker]:
    markers: list[CollisionMarker] = []
    urdf_dir = urdf_path.parent

    for link_name, link in urdf.link_map.items():
        if link_name in srdf_spheres:
            for i, (center, radius) in enumerate(srdf_spheres[link_name]):
                name = f"/robot_collision/{link_name}/srdf_sphere_{i}"
                handle = server.scene.add_icosphere(
                    name,
                    radius=radius,
                    color=(255, 140, 0),
                    opacity=0.6,
                    cast_shadow=False,
                )
                local_tf = np.eye(4)
                local_tf[:3, 3] = center
                markers.append(
                    CollisionMarker(
                        handle=handle,
                        link_name=link_name,
                        local_tf=local_tf,
                        source="srdf",
                    )
                )

        for i, collision in enumerate(link.collisions or []):
            mesh = _make_collision_mesh(urdf_dir, collision.geometry)
            if mesh is None:
                continue

            name = f"/robot_collision/{link_name}/urdf_collision_{i}"
            handle = server.scene.add_mesh_simple(
                name,
                mesh.vertices,
                mesh.faces,
                color=(40, 170, 255),
                opacity=0.7,
                cast_shadow=False,
                receive_shadow=False,
            )
            markers.append(
                CollisionMarker(
                    handle=handle,
                    link_name=link_name,
                    local_tf=_origin_to_matrix(collision.origin),
                    source="urdf",
                )
            )

    return markers


def _update_collision_markers(
    urdf: yourdfpy.URDF,
    markers: list[CollisionMarker],
) -> None:
    for marker in markers:
        link_tf = urdf.get_transform(marker.link_name)
        world_tf = link_tf @ marker.local_tf
        marker.handle.wxyz = _wxyz_from_matrix(world_tf)
        marker.handle.position = _xyz_from_matrix(world_tf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize URDF and collisions with viser.")
    parser.add_argument("urdf_file", type=str, help="Path to URDF.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--srdf", type=str, default=None, help="Optional SRDF path. Defaults to URDF sidecar.")
    args = parser.parse_args()

    urdf_path = Path(args.urdf_file).resolve()
    srdf_path = Path(args.srdf).resolve() if args.srdf else urdf_path.with_suffix(".srdf")

    srdf_spheres = _load_srdf_spheres(srdf_path)
    if srdf_spheres:
        print(f"Loaded SRDF spheres for {len(srdf_spheres)} links from {srdf_path}")
    else:
        print(f"No SRDF link_sphere_approximation data found at {srdf_path}")

    urdf = yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        filename_handler=lambda fname: str(urdf_path.parent / _urdf_clean_filename(fname)),
    )

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/ground", width=4.0, height=4.0, cell_size=0.25, plane="xy")
    server.scene.add_frame("/robot", show_axes=False)

    robot = ViserUrdf(
        server,
        urdf_or_path=urdf,
        root_node_name="/robot",
        load_meshes=True,
        load_collision_meshes=False,
    )

    collision_markers = _build_collision_markers(server, urdf, urdf_path, srdf_spheres)

    joint_names = robot.get_actuated_joint_names()
    joint_limits = robot.get_actuated_joint_limits()
    current_cfg = np.zeros(len(joint_names), dtype=float)

    with server.gui.add_folder("Display"):
        show_robot_toggle = server.gui.add_checkbox(
            "Show Robot Visuals",
            initial_value=True,
        )
        show_srdf_toggle = server.gui.add_checkbox(
            "Show SRDF Spheres",
            initial_value=True,
        )
        show_urdf_toggle = server.gui.add_checkbox(
            "Show URDF Collision",
            initial_value=True,
        )

    def _update_collision_visibility() -> None:
        for marker in collision_markers:
            marker.handle.visible = (
                (marker.source == "srdf" and show_srdf_toggle.value)
                or (marker.source == "urdf" and show_urdf_toggle.value)
            )

    @show_srdf_toggle.on_update
    def _(_event) -> None:
        _update_collision_visibility()

    @show_urdf_toggle.on_update
    def _(_event) -> None:
        _update_collision_visibility()

    @show_robot_toggle.on_update
    def _(_event) -> None:
        robot.show_visual = show_robot_toggle.value

    sliders = []
    with server.gui.add_folder("Joints"):
        for i, joint_name in enumerate(joint_names):
            lower, upper = joint_limits[joint_name]
            lo = -np.pi if lower is None else float(lower)
            hi = np.pi if upper is None else float(upper)
            if hi < lo:
                lo, hi = hi, lo
            initial = 0.5 * (lo + hi)
            current_cfg[i] = initial
            slider = server.gui.add_slider(
                joint_name,
                min=lo,
                max=hi,
                step=0.001,
                initial_value=initial,
            )
            sliders.append(slider)

    def _apply_cfg() -> None:
        robot.update_cfg(current_cfg)
        _update_collision_markers(urdf, collision_markers)

    _apply_cfg()
    robot.show_visual = show_robot_toggle.value
    _update_collision_visibility()

    for i, slider in enumerate(sliders):
        @slider.on_update
        def _(_event, idx=i, s=slider) -> None:
            current_cfg[idx] = s.value
            _apply_cfg()

    print(f"Viser URDF viewer running at http://localhost:{args.port}")

    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
