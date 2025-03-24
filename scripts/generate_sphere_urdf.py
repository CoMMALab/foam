from pathlib import Path

from fire import Fire

from foam import *

from trimesh.primitives import Sphere
from trimesh.nsphere import minimum_nsphere

def main(
        filename: str = "assets/panda/panda.urdf",
        output: str = "spherized.urdf",
        database: str = "sphere_database.json",
        depth: int = 1,
        branch: int = 8,
        method: str = "medial",
        testerLevels: int = 2,
        numCover: int = 5000,
        minCover: int = 5,
        initSpheres: int = 1000,
        minSpheres: int = 200,
        erFact: int = 2,
        expand: bool = True,
        merge: bool = True,
        burst: bool = False,
        optimise: bool = True,
        maxOptLevel: int = 1,
        balExcess: float = 0.05,
        verify: bool = True,
        num_samples: int = 500,
        min_samples: int = 1,
        use_volume_heuristic: bool = False,
        volume_heuristic_ratio: float = 0.7,
        manifold_leaves: int = 1000,
        simplification_ratio: float = 0.2,
        threads: int = 16,
        shrinkage: float = 1.,
        **kwargs: float
    ):

    sh = SpherizationHelper(Path(database), threads)

    urdf = load_urdf(Path(filename))
    meshes = get_urdf_meshes(urdf, shrinkage)

    for mesh in meshes:

        # branch_value = max(
        #     int(mesh.mesh.volume * 10000 * volume_heuristic_ratio), branch
        #     ) if use_volume_heuristic else branch

        center, radius = minimum_nsphere(mesh.mesh.vertices)
        vr = Sphere(radius, center).volume / mesh.mesh.volume
        branch_value = min(int(vr * volume_heuristic_ratio), branch)

        key = mesh.name.split(":")[0]
        if key in kwargs:
            branch_value = int(kwargs[key] * branch_value)

        print(f"Link::Mesh: {mesh.name}\n  Target Spheres: {branch_value}")

        sh.spherize_mesh(
            mesh.name,
            mesh.mesh,
            mesh.scale,
            mesh.xyz,
            method,
            mesh.rpy,
            depth=depth,
            branch=branch_value,
            testerLevels=testerLevels,
            numCover=numCover,
            minCover=minCover,
            initSpheres=initSpheres,
            minSpheres=minSpheres,
            erFact=erFact,
            expand=expand,
            merge=merge,
            burst=burst,
            optimise=optimise,
            maxOptLevel=maxOptLevel,
            balExcess=balExcess,
            verify=verify,
            num_samples=num_samples,
            min_samples=min_samples,        
            manifold_leaves=manifold_leaves,
            simplification_ratio=simplification_ratio,

            )

    primitives = get_urdf_primitives(urdf, shrinkage)
    for primitive in primitives:
        center, radius = minimum_nsphere(primitive.mesh.vertices)
        vr = Sphere(radius, center).volume / primitive.mesh.volume
        branch_value = min(int(vr * volume_heuristic_ratio), branch)

        key = primitive.name.split(":")[0]
        if key in kwargs:
            branch_value = int(kwargs[key] * branch_value)

        print(primitive.name, vr, branch_value)

        sh.spherize_mesh(
            primitive.name,
            primitive.mesh,
            primitive.scale,
            primitive.xyz,
            method,
            primitive.rpy,
            depth=depth,
            branch=branch_value,
            testerLevels=testerLevels,
            numCover=numCover,
            minCover=minCover,
            initSpheres=initSpheres,
            minSpheres=minSpheres,
            erFact=erFact,
            expand=expand,
            merge=merge,
            burst=burst,
            optimise=optimise,
            maxOptLevel=maxOptLevel,
            balExcess=balExcess,
            verify=verify,
            num_samples=num_samples,
            min_samples=min_samples,        
            manifold_leaves=manifold_leaves,
            simplification_ratio=simplification_ratio,

            )

    mesh_spheres = {mesh.name: sh.get_spherization(mesh.name, depth, branch) for mesh in meshes}
    primitive_spheres = {
        primitive.name: sh.get_spherization(primitive.name, depth, branch, cache = False)
        for primitive in primitives
        }

    set_urdf_spheres(urdf, mesh_spheres | primitive_spheres)
    save_urdf(urdf, Path(output))


if __name__ == "__main__":
    Fire(main)
