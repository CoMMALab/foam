"""Spherize a URDF interactively: tune foam's parameters, preview the spheres on the robot, export.

    python scripts/spherize_urdf_interactive.py assets/panda/panda.urdf
"""
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from fire import Fire

from foam import *
from foam.viz import *

METHODS = ("medial", "spawn", "grid", "hubbard", "octree")


def main(urdf: str, output: str = "spherized.urdf", threads: int = 16, port: int = 8080):
    urdf_dict = load_urdf(Path(urdf))
    robot = load_robot(Path(urdf))
    links = sorted({m.name.split(":")[0] for m in get_urdf_meshes(urdf_dict) + get_urdf_primitives(urdf_dict)})

    server = make_server(port)
    viser_urdf = ViserUrdf(server, robot, root_node_name = "/robot")
    visuals = LinkSpheres(server, robot)
    result: dict[str, list[Spherization]] = {}

    tabs = server.gui.add_tab_group()
    with tabs.add_tab("Spherize"):
        method = server.gui.add_dropdown("Method", METHODS, initial_value = "medial")
        depth = server.gui.add_slider("Depth", min = 1, max = 4, step = 1, initial_value = 1)
        branch = server.gui.add_slider("Max branch", min = 1, max = 64, step = 1, initial_value = 8)
        volume_ratio = server.gui.add_slider("Volume ratio", min = 0.1, max = 2.0, step = 0.05, initial_value = 0.7)
        with server.gui.add_folder("Preprocessing", expand_by_default = False):
            manifold_leaves = server.gui.add_slider("Manifold leaves", min = 100, max = 10000, step = 100, initial_value = 1000)
            simplification = server.gui.add_slider("Simplify ratio", min = 0.05, max = 1.0, step = 0.05, initial_value = 0.2)
            shrinkage = server.gui.add_slider("Shrinkage", min = 0.5, max = 1.0, step = 0.01, initial_value = 1.0)
        with server.gui.add_folder("Per-link budget scale", expand_by_default = False):
            link_scale = {link: server.gui.add_slider(link, min = 0.0, max = 4.0, step = 0.05, initial_value = 1.0) for link in links}
        run = server.gui.add_button("Spherize")
        status = server.gui.add_markdown("Press **Spherize** to start.")

    with tabs.add_tab("View"):
        level = server.gui.add_slider("Level", min = 0, max = 1, step = 1, initial_value = 1)
        opacity = server.gui.add_slider("Opacity", min = 0.1, max = 1.0, step = 0.1, initial_value = 0.7)
        show_spheres = server.gui.add_checkbox("Show spheres", initial_value = True)
        show_visual = server.gui.add_checkbox("Show robot", initial_value = True)
        sliders = add_joint_sliders(server, robot)

    with tabs.add_tab("Export"):
        filename = server.gui.add_text("Output URDF", initial_value = output)
        export = server.gui.add_button("Export URDF", disabled = True)

    def link_spheres_at(lvl: int) -> dict[str, list[Sphere]]:
        grouped = defaultdict(list)
        for name, levels in result.items():
            grouped[name.split(":")[0]].extend(levels[min(lvl, len(levels) - 1)].spheres)
        return grouped

    def draw(_ = None):
        if result:
            visuals.set(link_spheres_at(level.value), opacity.value, show_spheres.value)

    @run.on_click
    def _(_):
        run.disabled = True
        status.content = "Spherizing..."
        start = time.time()
        try:
            result.clear()
            result.update(
                spherize_urdf(
                    urdf_dict,
                    method = method.value,
                    depth = depth.value,
                    branch = branch.value,
                    volume_heuristic_ratio = volume_ratio.value,
                    link_scale = {link: s.value for link, s in link_scale.items()},
                    threads = threads,
                    shrinkage = shrinkage.value,
                    manifold_leaves = manifold_leaves.value,
                    simplification_ratio = simplification.value,
                    )
                )
        except Exception as e:
            status.content = f"**Failed:** {e}"
            return
        finally:
            run.disabled = False

        level.max = depth.value
        level.value = depth.value
        draw()
        counts = {link: len(s) for link, s in link_spheres_at(depth.value).items()}
        status.content = f"{sum(counts.values())} spheres in {time.time() - start:.1f}s\n\n" + "\n".join(
            f"- {link}: {n}" for link, n in counts.items()
            )
        export.disabled = False

    for handle in (level, opacity, show_spheres):
        handle.on_update(draw)

    @show_visual.on_update
    def _(_):
        viser_urdf.show_visual = show_visual.value

    @export.on_click
    def _(_):
        out = load_urdf(Path(urdf))  # fresh copy, set_urdf_spheres rewrites collisions in place
        set_urdf_spheres(out, {name: levels[min(level.value, len(levels) - 1)] for name, levels in result.items()})
        save_urdf(out, Path(filename.value))
        status.content += f"\n\nExported {filename.value}"
        print(f"Exported {filename.value}")

    print(f"Spherizer ready at http://localhost:{port}")
    while True:
        cfg = np.array([s.value for s in sliders])
        viser_urdf.update_cfg(cfg)
        visuals.update(cfg)
        time.sleep(0.05)


if __name__ == "__main__":
    Fire(main)
