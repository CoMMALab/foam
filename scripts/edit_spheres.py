"""Hand-edit the spheres of a spherized URDF in the browser.

Click a sphere to select it, drag the gizmo to move it, use the slider to resize, then Save.

    python scripts/edit_spheres.py spherized.urdf [--output edited.urdf]
"""
import time
from pathlib import Path

import numpy as np
from fire import Fire

from foam import *
from foam.viz import *

HIGHLIGHT = (0.2, 1.0, 0.8)


def main(urdf: str, output: str | None = None, port: int = 8080):
    output = Path(output or urdf)
    urdf_dict = load_urdf(Path(urdf))
    link_spheres = get_link_spheres(urdf_dict)
    if not link_spheres:
        raise SystemExit(f"No sphere collision geometry in {urdf}; is it spherized?")

    robot = load_robot(Path(urdf))
    server = make_server(port)
    viser_urdf = ViserUrdf(server, robot, root_node_name = "/robot")
    visuals = LinkSpheres(server, robot)
    selected: list = []  # [(link, i), transform control]

    with server.gui.add_folder("Editor"):
        status = server.gui.add_markdown("")
        save = server.gui.add_button("Save")
        opacity = server.gui.add_slider("Opacity", min = 0.1, max = 1.0, step = 0.1, initial_value = 0.7)

    with server.gui.add_folder("Selected sphere"):
        selection = server.gui.add_markdown("Click a sphere to select it.")
        radius = server.gui.add_slider("Radius", min = 0.001, max = 0.3, step = 0.001, initial_value = 0.05, disabled = True)
        deselect = server.gui.add_button("Deselect", disabled = True)
        delete = server.gui.add_button("Delete sphere", color = "red", disabled = True)

    sliders = add_joint_sliders(server, robot)

    def update_status(dirty: bool):
        n = sum(len(s) for s in link_spheres.values())
        status.content = f"{n} spheres in {len(link_spheres)} links" + (" (unsaved)" if dirty else "")

    def draw(key = None, color = None):
        """Redraw one sphere, or rebuild all of them."""
        if key is None:
            visuals.set(link_spheres, opacity.value)
            for k, h in visuals.handles.items():
                h.on_click(lambda _, k = k: select(k))
            return

        link, i = key
        visuals.handles[key].remove()
        h = visuals.add(link, i, link_spheres[link][i], color = color, opacity = opacity.value)
        h.on_click(lambda _, k = key: select(k))

    def select(key):
        if selected:
            return

        link, i = key
        sphere = link_spheres[link][i]
        draw(key, HIGHLIGHT)
        control = server.scene.add_transform_controls(
            f"/spheres/{link}/control",
            scale = 0.2,
            depth_test = False,
            disable_rotations = True,
            position = tuple(sphere.origin),
            )

        @control.on_update
        def _(_):
            sphere.origin = np.array(control.position)
            visuals.handles[key].position = control.position
            update_status(True)

        selected[:] = [key, control]
        selection.content = f"**{link}** #{i}"
        radius.value = sphere.radius
        for g in (radius, deselect, delete):
            g.disabled = False

    def clear_selection():
        if not selected:
            return

        key, control = selected
        control.remove()
        selected.clear()
        selection.content = "Click a sphere to select it."
        for g in (radius, deselect, delete):
            g.disabled = True

        return key

    @radius.on_update
    def _(_):
        if selected and not radius.disabled:
            link, i = selected[0]
            link_spheres[link][i].radius = radius.value
            draw(selected[0], HIGHLIGHT)
            update_status(True)

    @deselect.on_click
    def _(_):
        key = clear_selection()
        draw(key)

    @delete.on_click
    def _(_):
        link, i = clear_selection()
        del link_spheres[link][i]  # indices shift, so rebuild everything
        draw()
        update_status(True)

    @save.on_click
    def _(_):
        set_link_spheres(urdf_dict, link_spheres)
        save_urdf(urdf_dict, output)
        update_status(False)
        print(f"Saved {output}")

    @opacity.on_update
    def _(_):
        clear_selection()
        draw()

    draw()
    update_status(False)
    print(f"Editor ready at http://localhost:{port}")
    while True:
        cfg = np.array([s.value for s in sliders])
        viser_urdf.update_cfg(cfg)
        visuals.update(cfg)
        time.sleep(0.05)


if __name__ == "__main__":
    Fire(main)
