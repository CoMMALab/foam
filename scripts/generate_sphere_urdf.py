from pathlib import Path

from fire import Fire

from foam import *


def main(
        filename: str = "assets/panda/panda.urdf",
        output: str = "spherized.urdf",
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
        volume_heuristic_ratio: float = 0.7,
        manifold_leaves: int = 1000,
        simplification_ratio: float = 0.2,
        threads: int = 16,
        shrinkage: float = 1.,
        **kwargs: float
    ):
    """Extra `--<link_name> <float>` flags scale that link's sphere budget."""
    urdf = load_urdf(Path(filename))
    spheres = spherize_urdf(
        urdf,
        depth = depth,
        branch = branch,
        volume_heuristic_ratio = volume_heuristic_ratio,
        link_scale = kwargs,
        threads = threads,
        shrinkage = shrinkage,
        manifold_leaves = manifold_leaves,
        simplification_ratio = simplification_ratio,
        method = method,
        testerLevels = testerLevels,
        numCover = numCover,
        minCover = minCover,
        initSpheres = initSpheres,
        minSpheres = minSpheres,
        erFact = erFact,
        expand = expand,
        merge = merge,
        burst = burst,
        optimise = optimise,
        maxOptLevel = maxOptLevel,
        balExcess = balExcess,
        verify = verify,
        num_samples = num_samples,
        min_samples = min_samples,
        )

    set_urdf_spheres(urdf, {name: levels[depth] for name, levels in spheres.items()})
    save_urdf(urdf, Path(output))


if __name__ == "__main__":
    Fire(main)
