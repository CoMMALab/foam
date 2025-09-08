# Foam: Spherical Approximations of URDFs
<p align="center">
  <img src="images/menagerie.png" width="40%"/>
</p>

## Foam
The foam tool is an automated mesh simplification framework for generating spherical approximations of robot geometries directly from Universal Robot Description Format (URDF) specifications. Foam outputs a procesed URDF file with all the collision geometries replaced with spherical primitives. Spherized meshes can reduce the computational load of modeling robots, have better support across simulators, and fix mesh defects while maintaining an accurate approximation of the source model.

## Citation:
If you'd like to cite this work, please use the following and see the [associated paper](https://arxiv.org/abs/2503.13704):
```
@misc{2025foamtoolsphericalapproximation,
      title={Foam: A Tool for Spherical Approximation of Robot Geometry}, 
      author={Sai Coumar and Gilbert Chang and Nihar Kodkani and Zachary Kingston},
      year={2025},
      eprint={2503.13704},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2503.13704}, 
}
```

## Installation
Foam can either be installed natively on linux/wsl or through a dockerized container. We recommend using the dockerized build for cleaner dependency management and resource utilization monitoring.

## Building with Docker 
Navigate to the [Docker quickstart](https://github.com/CoMMALab/foam/blob/master/DOCKER_QUICKSTART.md) for detailed instructions.

## Installation with pixi

Foam supports installation with [pixi](https://pixi.sh/latest/).

To install foam with pixi, run the following command:

```bash
pixi run build
```

## Obtaining & Building Dependencies 
Foam is supported on Linux/WSL with python3.11 and cmake 3.29. Output visualization can be run on Windows/MacOS natively using spherized files.
```sh
git clone --recursive https://github.com/CoMMALab/foam.git
cd foam
cmake -Bbuild -GNinja .
cmake --build build/
pip install -r requirements.txt
```


## Using Scripts

In the `scripts` directory:

 - `python generate_spheres.py <mesh>`: Generates and outputs a JSON file in scripts directory given a mesh input file.
  
   > Specify `<mesh>` with the path to the mesh file that will be spherized.

   > Optionally specify spherization algorithm using `--method <spherization algorithm>`. Uses medial algorithm for spherization by default. Available algorithms include `medial`, `spawn`, `grid`, `hubbard`, and `octree`.
  
   > Optionally specify arguments such as `--depth <depth>` and `--branch <branching factor>` to control sphere generation process. Full list of spherization arguments for different methods can be found at [mlund/spheretree on GitHub](https://github.com/mlund/spheretree?tab=readme-ov-file#programs).
  
   > Optionally specify `--manifold-leaves <leaves>` to control mesh correction on invalid meshes.
  
   > Valid mesh formats include `.DAE`, `.STL`, and `.OBJ`.
- `python generate_sphere_urdf.py <urdf>`: Generates and outputs a JSON file in scripts directory given a URDF input file.
  
  > Specify `<urdf>` with the path to the urdf file that will be spherized.

  > Optionally specify spherization algorithm using `--method <spherization algorithm>`. Uses medial algorithm for spherization by default. Available algorithms include `medial`, `spawn`, `grid`, `hubbard`, and `octree`.
  
  > Optionally specify arguments such as `--depth <depth>` and `--branch <branching factor>` to control sphere generation process. Full list of spherization arguments for different methods can be found at [mlund/spheretree on GitHub](https://github.com/mlund/spheretree?tab=readme-ov-file#programs).
  
  > Optionally specify `--manifold-leaves <leaves>` to control mesh correction on invalid meshes.
  
  > Takes urdfs as input rather than mesh formats.
- `python visualize_spheres.py <mesh> <spheres>`: Visualizes spheres and mesh.
  
  > Specify `<mesh>` with the path to the original mesh file.
  
  > Optionally specify `<spheres>` to visualize the spherized approximation on top of the original mesh.
  
  > Optionally specify `--depth <depth>` for the sphere level to visualize.

## Third-party Dependencies

Third-party dependencies are stored in the `./external` directory.
Compiled script binaries are copied into the `foam/external` directory.



#### [Manifold](https://github.com/hjwdzh/Manifold)/[ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus)
  Manifold is used to ensure mesh data format and convert a triangle mesh `.obj` file into a manifold `.obj` file. Manifolding creates a continuous surface without gaps to avoid dynamics and collision issues. ManifoldPlus performs the same functionality with an advanced algorithm. Manifold code is included as a compiled submodule. ManifoldPlus code is included as a submodule.

#### [Quadric Mesh Simplification](https://github.com/sp4cerat/Fast-Quadric-Mesh-Simplification)
  Quadric Mesh Simplification reduces the complexity of the generated manifold file as a preprocessing step before spherization. This code is included as a submodule.

#### [SphereTree](https://github.com/mlund/spheretree)
  The code in the `./spheretree` directory has been copied from the linked directory and modified to build on modern systems with `CMake` rather than `autotools`. SphereTree has a variety of algorithms to construct sphere-tree representations of polygonal models, but foam currently uses the MedialTree algorithm for SphereTree generation. The output is formatted and dumped into an output file.

<img src="images/pipeline_w_curobo.png" alt="Pipeline" />

