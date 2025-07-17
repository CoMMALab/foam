from json import dumps
from pathlib import Path

from fire import Fire

from foam import *
import time


def main(
        mesh: str,
        output: str | None = None,
        depth: int = 1,
        branch: int = 8,
        method: str = "medial",
        scale: float = 1.,
        manifold_leaves: int = 1000,
        simplify_ratio: float = 0.2,
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
        eval: bool = False,
        num_samples: int = 500,
        min_samples: int = 1
    ):

    start_time = time.time()

    mesh_filepath = Path(mesh)
    if not mesh_filepath.exists():
        raise RuntimeError(f"Path {mesh} does not exist!")
    
    # Prepare the spherization parameters
    spherization_kwargs = {
        'depth': depth,
        'branch': branch,
        'method': method,
        'testerLevels': testerLevels,
        'numCover': numCover,
        'minCover': minCover,
        'initSpheres': initSpheres,
        'minSpheres': minSpheres,
        'erFact': erFact,
        'expand': expand,
        'merge': merge,
        'burst': burst,
        'optimise': optimise,
        'maxOptLevel': maxOptLevel,
        'balExcess': balExcess,
        'verify': verify,
        'num_samples': num_samples,
        'min_samples': min_samples
    }
    
    # Processing kwargs for mesh processing
    process_kwargs = {
        'manifold_leaves': manifold_leaves,
        'ratio': simplify_ratio,
    }

    # Call spherize_mesh with the updated kwargs
    spheres = spherize_mesh(
        mesh,
        mesh_filepath,
        scale=np.array([scale] * 3),
        spherization_kwargs=spherization_kwargs,
        process_kwargs=process_kwargs
    )

    # Set the default output filename if not provided
    if not output:
        output = mesh_filepath.stem + "-spheres.json"

    # Write the result to a JSON file
    with open(output, 'w') as f:
        f.write(dumps(spheres, indent=4, cls=SphereEncoder))

    end_time = time.time()

    print(f"Generated spheres in {end_time - start_time:.6f} seconds")



if __name__ == "__main__":
    Fire(main)
