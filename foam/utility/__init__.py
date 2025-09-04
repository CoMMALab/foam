from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from numpy.typing import NDArray

from trimesh.base import Trimesh
from trimesh.scene.scene import Scene
from trimesh.util import concatenate
from trimesh.exchange.load import load_mesh
from trimesh.repair import *
from trimesh.smoothing import filter_humphrey, filter_taubin
from trimesh.primitives import Box, Cylinder
from trimesh.primitives import Sphere as TMSphere

import numpy as np

import resolve_robotics_uri_py
import xmltodict


def fix_mesh(mesh: Trimesh):
    fix_normals(mesh)
    fix_inversion(mesh)
    fix_winding(mesh)
    mesh.vertex_normals = -mesh.vertex_normals


def smooth_mesh(mesh: Trimesh):
    filter_humphrey(mesh, iterations = 100)


@contextmanager
def tempmesh():
    f = NamedTemporaryFile('w', suffix = f'.obj')
    try:
        yield f, Path(f.name)
    finally:
        f.close()


def as_mesh(scene_or_mesh: Trimesh | Scene) -> Trimesh | None:
    if isinstance(scene_or_mesh, Scene):
        if len(scene_or_mesh.geometry) == 0:
            return None
        else:
            return concatenate(
                [Trimesh(vertices = g.vertices, faces = g.faces) for g in scene_or_mesh.geometry.values()]
                )                                                                                          # type: ignore
    else:
        return scene_or_mesh


def load_mesh_file(mesh_filepath: Path) -> Trimesh:
    try:
        mesh_filepath = resolve_robotics_uri_py.resolve_robotics_uri(str(mesh_filepath))
        mesh = as_mesh(load_mesh(mesh_filepath, process = False)) # type: ignore
        if mesh is None:
            raise RuntimeError("Failed to load mesh!")

        return mesh

    except Exception as e:
        raise e


@dataclass
class URDFMesh:
    name: str
    mesh: Trimesh
    xyz: NDArray
    rpy: NDArray
    scale: NDArray


@dataclass
class URDFPrimitive:
    name: str
    mesh: Trimesh
    xyz: NDArray
    rpy: NDArray
    scale: NDArray


URDFDict = dict[str, Any]


def _urdf_array_to_np(array: str) -> NDArray:
    return np.fromiter(map(float, array.split()), dtype = float)


def _urdf_clean_filename(filename: str) -> str:
    if filename.startswith('package://'):
        filename = filename[len('package://'):]

    return filename


def load_urdf(urdf_path: Path) -> URDFDict:
    with open(urdf_path, 'r') as f:
        xml = xmltodict.parse(f.read())
        xml['robot']['@path'] = urdf_path
        return xml


def get_urdf_primitives(urdf: URDFDict, shrinkage: float = 1.) -> list[URDFPrimitive]:
    primitives = []
    for link in urdf['robot']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        scale = np.array([shrinkage] * 3)
        for i, collision in enumerate(collisions):
            if 'origin' in collision:
                xyz = _urdf_array_to_np(collision['origin']['@xyz'])
                rpy = _urdf_array_to_np(collision['origin']['@rpy'])
            else:
                xyz = np.array([0., 0., 0.])
                rpy = np.array([0., 0., 0.])

            geometry = collision['geometry']
            if 'box' in geometry:
                box = geometry['box']
                size = _urdf_array_to_np(box['@size'])
                primitives.append(URDFPrimitive(f"{name}::primitive{i}", Box(size), xyz, rpy, scale))

            elif 'sphere' in geometry:
                sphere = geometry['sphere']
                radius = float(sphere['@radius'])
                primitives.append(URDFPrimitive(f"{name}::primitive{i}", TMSphere(radius), xyz, rpy, scale))

            elif 'cylinder' in geometry:
                cylinder = geometry['cylinder']
                length = float(cylinder['@length'])
                radius = float(cylinder['@radius'])
                primitives.append(
                    URDFPrimitive(f"{name}::primitive{i}", Cylinder(radius, length), xyz, rpy, scale)
                    )

    return primitives


def get_urdf_meshes(urdf: URDFDict, shrinkage: float = 1.) -> list[URDFMesh]:
    urdf_dir = Path(urdf['robot']['@path']).parent

    meshes = []
    for link in urdf['robot']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        for collision in collisions:
            if 'origin' in collision:
                xyz = _urdf_array_to_np(collision['origin']['@xyz'])
                rpy = _urdf_array_to_np(collision['origin']['@rpy'])
            else:
                xyz = np.array([0., 0., 0.])
                rpy = np.array([0., 0., 0.])

            geometry = collision['geometry']
            if 'mesh' in geometry:
                mesh = geometry['mesh']
                filename = _urdf_clean_filename(mesh['@filename'])
                scale = _urdf_array_to_np(mesh['@scale']) if 'scale' in mesh else np.array([1., 1., 1.])
                scale *= shrinkage                                                                        # HACK: need to scale down to get some tight self collision working
                meshes.append(
                    URDFMesh(f"{name}::{filename}", load_mesh_file(urdf_dir / filename), xyz, rpy, scale)
                    )

    return meshes


def get_urdf_spheres(urdf: URDFDict) -> Iterator[tuple[float, float, float, float]]:
    primitives = []
    for link in urdf['robot']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        if not isinstance(link['collision'], list):
            link['collision'] = [link['collision']]

        for collision in link['collision']:
            if 'origin' in collision:
                xyz = _urdf_array_to_np(collision['origin']['@xyz'])
                rpy = _urdf_array_to_np(collision['origin']['@rpy'])
            else:
                xyz = np.array([0., 0., 0.])
                rpy = np.array([0., 0., 0.])

            geometry = collision['geometry']
            if 'sphere' in geometry:
                sphere = geometry['sphere']
                radius = float(sphere['@radius'])
                yield *xyz.tolist(), radius


def set_urdf_spheres(urdf: URDFDict, spheres):
    total_spheres = 0
    for link in urdf['robot']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        spherizations = []
        for i, collision in enumerate(collisions):

            geometry = collision['geometry']

            if 'box' in geometry or 'cylinder' in geometry or 'sphere' in geometry:
                key = f"{name}::primitive{i}"
                if key in spheres:
                    spherizations.append(spheres[key])

            elif 'mesh' in geometry:
                mesh = geometry['mesh']
                filename = _urdf_clean_filename(mesh['@filename'])
                key = f"{name}::{filename}"

                if key in spheres:
                    spherizations.append(spheres[key])

        collision = []
        for spherization in spherizations:
            for sphere in spherization.spheres:
                total_spheres += 1
                collision.append(
                    {
                        'geometry': {
                            'sphere': {
                                '@radius': sphere.radius
                                }
                            },
                        'origin': {
                            '@xyz': ' '.join(map(str, sphere.origin)), '@rpy': '0 0 0'
                            }
                        }
                    )

        link['collision'] = collision
    print(f"spheres: {total_spheres}")


def save_urdf(urdf: URDFDict, filename: Path):
    with open(filename, 'w') as f:
        f.write(xmltodict.unparse(urdf, pretty = True))


## SDFs
def load_sdf(sdf_path: Path) -> URDFDict:
    with open(sdf_path, 'r') as f:
        xml = xmltodict.parse(f.read())
        return xml


def get_sdf_meshes(sdf: URDFDict, sdf_dir, shrinkage: float = 1.) -> list[URDFMesh]:
    meshes = []
    for link in sdf['sdf']['model']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        for collision in collisions:
            if 'pose' in collision:
                xyzrpy = _urdf_array_to_np(collision['pose'])
                xyz = xyzrpy[:3]
                rpy = xyzrpy[3:]
            else:
                xyz = np.array([0., 0., 0.])
                rpy = np.array([0., 0., 0.])

            geometry = collision['geometry']
            if 'mesh' in geometry:
                mesh = geometry['mesh']
                filename = _urdf_clean_filename(mesh['uri'])
                scale = _urdf_array_to_np(mesh['scale']) if 'scale' in mesh else np.array([1., 1., 1.])
                scale *= shrinkage                                                                       # HACK: need to scale down to get some tight self collision working
                meshes.append(
                    URDFMesh(f"{name}::{filename}", load_mesh_file(sdf_dir / filename), xyz, rpy, scale)
                    )

    return meshes


def get_sdf_primitives(sdf: URDFDict, shrinkage: float = 1.) -> list[URDFPrimitive]:
    primitives = []
    for link in sdf['sdf']['model']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        scale = np.array([shrinkage] * 3)
        for i, collision in enumerate(collisions):
            if 'pose' in collision:
                xyzrpy = _urdf_array_to_np(collision['pose'])
                xyz = xyzrpy[:3]
                rpy = xyzrpy[3:]
            else:
                xyz = np.array([0., 0., 0.])
                rpy = np.array([0., 0., 0.])

            geometry = collision['geometry']
            if 'box' in geometry:
                box = geometry['box']
                size = _urdf_array_to_np(box['size'])
                primitives.append(URDFPrimitive(f"{name}::primitive{i}", Box(size), xyz, rpy, scale))

            elif 'sphere' in geometry:
                sphere = geometry['sphere']
                radius = float(sphere['radius'])
                primitives.append(URDFPrimitive(f"{name}::primitive{i}", TMSphere(radius), xyz, rpy, scale))

            elif 'cylinder' in geometry:
                cylinder = geometry['cylinder']
                length = float(cylinder['length'])
                radius = float(cylinder['radius'])
                primitives.append(
                    URDFPrimitive(f"{name}::primitive{i}", Cylinder(radius, length), xyz, rpy, scale)
                    )

    return primitives


def set_sdf_spheres(sdf: URDFDict, spheres):
    total_spheres = 0
    for link in sdf['sdf']['model']['link']:
        name = link['@name']
        if 'collision' not in link:
            continue

        collisions = link['collision']

        if not isinstance(collisions, list):
            collisions = [collisions]

        spherizations = []
        for i, collision in enumerate(collisions):

            geometry = collision['geometry']

            if 'box' in geometry or 'cylinder' in geometry or 'sphere' in geometry:
                key = f"{name}::primitive{i}"
                if key in spheres:
                    spherizations.append(spheres[key])

            elif 'mesh' in geometry:
                mesh = geometry['mesh']
                filename = _urdf_clean_filename(mesh['uri'])
                key = f"{name}::{filename}"

                if key in spheres:
                    spherizations.append(spheres[key])

        collision = []
        for spherization in spherizations:
            for sphere in spherization.spheres:
                total_spheres += 1
                collision.append(
                    {
                        '@name': f'sphere_{total_spheres}',
                        'geometry': {
                            'sphere': {
                                'radius': sphere.radius
                                }
                            },
                        'pose': {' '.join(map(str, sphere.origin)) + ' 0 0 0'}
                        }
                    )

        link['collision'] = collision
    print(f"spheres: {total_spheres}")


def save_sdf(sdf: URDFDict, filename: Path):
    with open(filename, 'w') as f:
        f.write(xmltodict.unparse(sdf, pretty = True))
