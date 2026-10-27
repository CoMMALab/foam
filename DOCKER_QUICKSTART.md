# foam in Docker

The image bundles foam, its compiled spherization binaries, and the browser-based (viser) tools, so
everything runs the same on Linux, macOS, and Windows. No display forwarding is needed.

## Build

```bash
docker build -t foam .
```

## Run

Mount the folder holding your robot files at `/data`. Paths you pass are relative to it, and outputs land there.
Name any script in `scripts/` (with or without `.py`) as the command:

```bash
# Linux/macOS (PowerShell: use ${PWD} instead of $(pwd))
alias foam='docker run --rm -it -p 8080:8080 -v "$(pwd):/data" foam'

foam generate_sphere_urdf --filename panda/panda.urdf --output panda/panda_spherized.urdf
foam generate_spheres mesh.obj
foam generate_srdf panda/panda_spherized.urdf

# Browser tools: open http://localhost:8080
foam spherize_urdf_interactive panda/panda.urdf
foam edit_spheres panda/panda_spherized.urdf
foam visualize_urdf panda/panda_spherized.urdf
foam visualize_spheres mesh.obj mesh-spheres.json
```

Anything that isn't a script name runs as-is, e.g. `foam bash` for a shell or `foam python3 my_script.py`.

On Linux, add `-u "$(id -u):$(id -g)"` so output files are owned by you rather than root.
To run on another port, map it (`-p 9000:8080`). For two tools at once, give each its own port.

URDF mesh paths are relative, so mount a folder that contains both the URDF and its meshes.

## Developing foam

Mount your checkout over `/foam` to run your local source instead of the baked-in copy.
If the checkout has no compiled binaries, they're built on the first run:

```bash
docker run --rm -it -p 8080:8080 -v "$(pwd):/foam" -v "/path/to/robots:/data" foam bash
```
