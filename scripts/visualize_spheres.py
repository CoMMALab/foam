"""View a mesh and its spheres (JSON from generate_spheres.py) in the browser.

    python scripts/visualize_spheres.py mesh.obj mesh-spheres.json
"""
import time
from json import load as jsload
from pathlib import Path

from fire import Fire

from foam import *
from foam.viz import link_color, make_server


def main(mesh: str, spheres: str | None = None, depth: int = 1, port: int = 8080):
    server = make_server(port)
    loaded = load_mesh_file(Path(mesh))
    mesh_handle = server.scene.add_mesh_simple("/mesh", loaded.vertices, loaded.faces, color = (0.7, 0.7, 0.7))

    levels = []
    if spheres:
        with open(spheres, 'r') as f:
            levels = jsload(f, cls = SphereDecoder)

        if depth >= len(levels):
            raise RuntimeError(f"Depth {depth} not available ({len(levels)} levels)!")

    with server.gui.add_folder("Visualization"):
        show_mesh = server.gui.add_checkbox("Show mesh", initial_value = True)
        level = server.gui.add_slider("Level", min = 0, max = max(len(levels) - 1, 0), step = 1, initial_value = depth)
        opacity = server.gui.add_slider("Opacity", min = 0.1, max = 1.0, step = 0.1, initial_value = 0.6)
        info = server.gui.add_markdown("")

    @show_mesh.on_update
    def _(_):
        mesh_handle.visible = show_mesh.value

    handles = []

    def draw(_ = None):
        for h in handles:
            h.remove()

        handles.clear()
        if not levels:
            return

        spherization = levels[level.value]
        for i, sphere in enumerate(spherization.spheres):
            handles.append(server.scene.add_icosphere(
                f"/spheres/{i}",
                radius = sphere.radius,
                position = tuple(sphere.origin),
                color = link_color(i),
                opacity = opacity.value,
                ))

        info.content = f"{len(spherization)} spheres, mean error {spherization.mean_error:.4f}"

    level.on_update(draw)
    opacity.on_update(draw)
    draw()

    print(f"Visualization ready at http://localhost:{port}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    Fire(main)
