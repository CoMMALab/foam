"""View a URDF (e.g. a spherized one) in the browser with joint sliders.

    python scripts/visualize_urdf.py spherized.urdf
"""
import time
from pathlib import Path

import numpy as np
from fire import Fire

from foam.viz import *


def main(urdf: str, port: int = 8080):
    robot = load_robot(Path(urdf), collision = True)
    server = make_server(port)
    viser_urdf = ViserUrdf(
        server,
        robot,
        root_node_name = "/robot",
        load_collision_meshes = True,
        collision_mesh_color_override = (0.2, 0.6, 1.0, 0.6),
        )

    with server.gui.add_folder("Visualization"):
        show_visual = server.gui.add_checkbox("Show visual", initial_value = True)
        show_collision = server.gui.add_checkbox("Show collision", initial_value = True)

    @show_visual.on_update
    def _(_):
        viser_urdf.show_visual = show_visual.value

    @show_collision.on_update
    def _(_):
        viser_urdf.show_collision = show_collision.value

    sliders = add_joint_sliders(server, robot)
    print(f"Visualization ready at http://localhost:{port}")
    while True:
        viser_urdf.update_cfg(np.array([s.value for s in sliders]))
        time.sleep(0.02)


if __name__ == "__main__":
    Fire(main)
