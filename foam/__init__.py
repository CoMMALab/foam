from typing import Any
from json import load as jsload
from json import dumps as jsdumps
from concurrent.futures import ThreadPoolExecutor, Future
from concurrent.futures import wait as future_wait

from .utility import *
from .external import *
from .model import *

from trimesh.transformations import compose_matrix, euler_matrix, translation_matrix, quaternion_matrix


def smooth_manifold(mesh: Trimesh, manifold_leaves: int = 1000, ratio = 0.2) -> Trimesh:
    mesh = manifold(mesh, manifold_leaves)
    mesh = simplify_manifold(mesh, ratio)
    smooth_mesh(mesh)

    return mesh


def spherize_mesh(
        mesh: Trimesh | Path,
        scale: NDArray | None = None,
        position: NDArray | None = None,
        orientation: NDArray | None = None,
        spherization_kwargs: dict[str, Any] = {},
        process_kwargs: dict[str, Any] = {}
    ) -> list[Spherization]:

    if isinstance(mesh, Path):
        loaded_mesh = load_mesh_file(mesh)
    else:
        loaded_mesh = mesh

    loaded_mesh = loaded_mesh.copy()

    if position is not None or orientation is not None:
        tf = compose_matrix(angles=orientation, translate=position)
        loaded_mesh.apply_transform(tf)

    if scale is not None:
        loaded_mesh.apply_scale(scale)

    # Normalize center
    low_bounds, high_bounds = loaded_mesh.bounds
    offset = (high_bounds + low_bounds) / 2
    loaded_mesh.apply_transform(translation_matrix(-offset))

    method = spherization_kwargs['method']
    if not check_valid_for_spherization(method, loaded_mesh):
        loaded_mesh = smooth_manifold(loaded_mesh, **process_kwargs)

    if not check_valid_for_spherization(method, loaded_mesh):
        raise RuntimeError("Failed to make loaded_mesh valid!")

    try:
        spheres = compute_spheres(loaded_mesh, **spherization_kwargs)

    except:
        try:
            loaded_mesh = smooth_manifold(loaded_mesh, **process_kwargs)
            spheres = compute_spheres(loaded_mesh, **spherization_kwargs)

        except:
            raise RuntimeError("Failed to process loaded_mesh.")

    for sphere in spheres:
        sphere.offset(offset)

    return spheres


class ParallelSpherizer:

    def __init__(self, threads: int = 4):
        self.executor = ThreadPoolExecutor(max_workers = threads)
        self.waiting = {}

    def spherize_mesh(
            self,
            name: str,
            mesh: Trimesh | Path,
            scale: NDArray | None = None,
            position: NDArray | None = None,
            orientation: NDArray | None = None,
            spherization_kwargs: dict[str, Any] = {},
            process_kwargs: dict[str, Any] = {}
        ) -> Future[list[Spherization]]:
        future = self.executor.submit(
            spherize_mesh, mesh, scale, position, orientation, spherization_kwargs, process_kwargs
            )

        self.waiting[name] = future
        return future

    def wait(self):
        future_wait(self.waiting.values())

    def get(self, name: str) -> list[Spherization]:
        return self.waiting[name].result()


class SpherizationDatabase:

    def __init__(self, path: Path):
        self.path = path

        if path.exists():
            with open(path, 'r') as json_file:
                self.db = jsload(json_file, cls = SphereDecoder)
                self.db = {
                    mk: {
                        int(bk): {
                            int(dk): dv
                            for dk, dv in bv.items()
                            }
                        for bk, bv in mv.items()
                        }
                    for mk,
                    mv in self.db.items()
                    }

        else:
            self.db = {}

    def __del__(self):
        with open(self.path, 'w') as f:
            f.write(jsdumps(self.db, indent = 4, cls = SphereEncoder))

    def add(self, mesh: str, branch: int, depth: int, spherization: Spherization):
        if mesh not in self.db:
            self.db[mesh] = {}

        if branch not in self.db[mesh]:
            self.db[mesh][branch] = {}

        if depth not in self.db[mesh][branch]:
            self.db[mesh][branch][depth] = spherization

        else:
            if spherization < self.db[mesh][branch][depth]:
                self.db[mesh][branch][depth] = spherization

    def get(self, mesh: str, branch: int, depth: int) -> Spherization:
        return self.db[mesh][branch][depth]

    def exists(self, mesh: str, branch: int, depth: int) -> bool:
        if mesh in self.db:
            if branch in self.db[mesh]:
                if depth in self.db[mesh][branch]:
                    return True

        return False


class SpherizationHelper:

    def __init__(self, database: Path, threads: int = 8):
        self.ps = ParallelSpherizer(threads)
        self.db = SpherizationDatabase(database)

    def spherize_mesh(
            self,
            name: str,
            mesh: Trimesh | Path,
            scale: NDArray | None = None,
            position: NDArray | None = None,
            method: str = "medial",
            orientation: NDArray | None = None,
            depth: int = 1,
            branch: int = 8,
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
            min_samples: int = 1,
            manifold_leaves: int = 1000,
            simplification_ratio: float = 0.2
        ):
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
        if not self.db.exists(name, branch, depth):
            self.ps.spherize_mesh(
                name,
                mesh,
                scale,
                position,
                orientation,
                spherization_kwargs,
                {
                    'manifold_leaves': manifold_leaves,
                    'ratio': simplification_ratio,
                    },
                )

    def get_spherization(self, name: str, depth: int = 1, branch: int = 8, cache: bool = True) -> Spherization:
        if not self.db.exists(name, branch, depth):
            spherization = self.ps.get(name)
            if cache:
                for level, sphere_level in enumerate(spherization):
                    self.db.add(name, branch, level, sphere_level)

            return spherization[depth]

        else:
            return self.db.get(name, branch, depth)
