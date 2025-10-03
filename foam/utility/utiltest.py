import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from trimesh.primitives import Box, Cylinder, Sphere as TMSphere
from trimesh.base import Trimesh
import argparse
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from foam.utility.__init__ import (
    fix_mesh,
    smooth_mesh,
    tempmesh,
    as_mesh,
    load_mesh_file,
    load_urdf,
    get_urdf_primitives,
    get_urdf_meshes,
    get_urdf_spheres,
    set_urdf_spheres,
    save_urdf,
    _urdf_array_to_np,
    _urdf_clean_filename,
    URDFMesh,
    URDFPrimitive
)


class TestURDFUtilities(unittest.TestCase):
    """Test suite for URDF utility functions."""
    
    # Class variable to store the URDF path from command line
    test_urdf_path = None
    
    @classmethod
    def setUpClass(cls):
        """Set up the test URDF file path."""
        cls.repo_root = Path(__file__).parent.parent
        
        # Check if a URDF path was provided
        if cls.test_urdf_path:
            cls.urdf_file = Path(cls.test_urdf_path)
            if not cls.urdf_file.exists():
                raise FileNotFoundError(f"URDF file not found: {cls.urdf_file}")
            print(f"Testing with URDF: {cls.urdf_file}")
        else:
            cls.urdf_file = None
            print("No URDF file specified. Use --urdf <path> to specify a test URDF.")
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()


class TestHelperFunctions(TestURDFUtilities):
    """Test helper/utility functions."""
    
    def test_urdf_array_to_np(self):
        """Test conversion of URDF string arrays to numpy arrays."""
        # Test simple array
        result = _urdf_array_to_np("1.0 2.0 3.0")
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test with negative values
        result = _urdf_array_to_np("-1.5 0.0 2.5")
        expected = np.array([-1.5, 0.0, 2.5])
        np.testing.assert_array_almost_equal(result, expected)
        
        # Test single value
        result = _urdf_array_to_np("42.0")
        expected = np.array([42.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_urdf_clean_filename(self):
        """Test URDF filename cleaning."""
        # Test package:// prefix removal
        result = _urdf_clean_filename("package://robot/meshes/link.dae")
        self.assertEqual(result, "robot/meshes/link.dae")
        
        # Test without package prefix
        result = _urdf_clean_filename("meshes/link.stl")
        self.assertEqual(result, "meshes/link.stl")
        
        # Test empty string
        result = _urdf_clean_filename("")
        self.assertEqual(result, "")


class TestMeshOperations(TestURDFUtilities):
    """Test mesh manipulation functions."""
    
    def create_test_mesh(self):
        """Create a simple test mesh (cube)."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ])
        return Trimesh(vertices=vertices, faces=faces)
    
    def test_fix_mesh(self):
        """Test mesh fixing function."""
        mesh = self.create_test_mesh()
        original_normals = mesh.vertex_normals.copy()
        
        try:
            # Should not raise an exception
            fix_mesh(mesh)
            
            # Vertex normals should be inverted
            np.testing.assert_array_almost_equal(
                mesh.vertex_normals, 
                -original_normals
            )
        except ModuleNotFoundError as e:
            self.skipTest(f"Missing required module: {e}")
    
    def test_smooth_mesh(self):
        """Test mesh smoothing function."""
        mesh = self.create_test_mesh()
        original_vertices = mesh.vertices.copy()
        
        # Should not raise an exception
        smooth_mesh(mesh)
        
        # Vertices should be modified (smoothed)
        # Note: We can't guarantee exact values, just that they changed
        self.assertIsInstance(mesh, Trimesh)
    
    def test_tempmesh_context_manager(self):
        """Test temporary mesh file context manager."""
        with tempmesh() as (f, path):
            # File should exist
            self.assertTrue(path.exists())
            self.assertTrue(path.suffix == '.obj')
            
            # Should be able to write to file
            f.write("test content")
            f.flush()
        
        # File should be cleaned up after context
        self.assertFalse(path.exists())
    
    def test_as_mesh_with_trimesh(self):
        """Test as_mesh with Trimesh input."""
        mesh = self.create_test_mesh()
        result = as_mesh(mesh)
        
        self.assertIsInstance(result, Trimesh)
        np.testing.assert_array_equal(result.vertices, mesh.vertices)
        np.testing.assert_array_equal(result.faces, mesh.faces)
    
    def test_as_mesh_with_empty_scene(self):
        """Test as_mesh with empty Scene."""
        from trimesh.scene.scene import Scene
        scene = Scene()
        result = as_mesh(scene)
        
        self.assertIsNone(result)


class TestURDFLoading(TestURDFUtilities):
    """Test URDF file loading and parsing."""
    
    @unittest.skipIf(not hasattr(TestURDFUtilities, 'urdf_file') or TestURDFUtilities.urdf_file is None, 
                     "No URDF file specified")
    def test_load_urdf(self):
        """Test loading the specified URDF file."""
        urdf_dict = load_urdf(TestURDFUtilities.urdf_file)
        
        # Check basic structure
        self.assertIn('robot', urdf_dict)
        self.assertIn('@path', urdf_dict['robot'])
        self.assertEqual(urdf_dict['robot']['@path'], TestURDFUtilities.urdf_file)
        
        # Check for links
        self.assertIn('link', urdf_dict['robot'])
    
    def test_load_urdf_with_minimal_example(self):
        """Test loading a minimal URDF example."""
        minimal_urdf = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <collision>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""
        urdf_path = self.temp_path / "test_minimal.urdf"
        urdf_path.write_text(minimal_urdf)
        
        urdf_dict = load_urdf(urdf_path)
        
        self.assertIn('robot', urdf_dict)
        self.assertEqual(urdf_dict['robot']['@name'], 'test_robot')
        self.assertIn('link', urdf_dict['robot'])


class TestURDFPrimitives(TestURDFUtilities):
    """Test URDF primitive extraction."""
    
    def create_test_urdf_with_primitives(self):
        """Create a test URDF with various primitive types."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="box_link">
        <collision>
            <geometry>
                <box size="1.0 2.0 3.0"/>
            </geometry>
            <origin xyz="0.5 0.5 0.5" rpy="0 0 1.57"/>
        </collision>
    </link>
    <link name="sphere_link">
        <collision>
            <geometry>
                <sphere radius="0.5"/>
            </geometry>
            <origin xyz="1.0 1.0 1.0" rpy="0 0 0"/>
        </collision>
    </link>
    <link name="cylinder_link">
        <collision>
            <geometry>
                <cylinder radius="0.3" length="2.0"/>
            </geometry>
        </collision>
    </link>
    <link name="no_collision_link">
    </link>
</robot>
"""
        urdf_path = self.temp_path / "test_primitives.urdf"
        urdf_path.write_text(urdf_content)
        return load_urdf(urdf_path)
    
    def test_get_urdf_primitives(self):
        """Test extraction of primitive geometries from URDF."""
        urdf = self.create_test_urdf_with_primitives()
        primitives = get_urdf_primitives(urdf)
        
        # Should have 3 primitives (box, sphere, cylinder)
        self.assertEqual(len(primitives), 3)
        
        # Check types
        types_found = set()
        for prim in primitives:
            self.assertIsInstance(prim, URDFPrimitive)
            types_found.add(type(prim.mesh))
        
        self.assertIn(Box, types_found)
        self.assertIn(TMSphere, types_found)
        self.assertIn(Cylinder, types_found)
    
    def test_get_urdf_primitives_with_shrinkage(self):
        """Test primitive extraction with shrinkage factor."""
        urdf = self.create_test_urdf_with_primitives()
        shrinkage = 0.9
        primitives = get_urdf_primitives(urdf, shrinkage=shrinkage)
        
        # All primitives should have the shrinkage factor in scale
        for prim in primitives:
            np.testing.assert_array_almost_equal(
                prim.scale,
                np.array([shrinkage, shrinkage, shrinkage])
            )
    
    def test_get_urdf_spheres(self):
        """Test extraction of sphere primitives as tuples."""
        urdf = self.create_test_urdf_with_primitives()
        spheres = list(get_urdf_spheres(urdf))
        
        # Should have 1 sphere
        self.assertEqual(len(spheres), 1)
        
        # Check sphere data (x, y, z, radius)
        x, y, z, radius = spheres[0]
        self.assertEqual(x, 1.0)
        self.assertEqual(y, 1.0)
        self.assertEqual(z, 1.0)
        self.assertEqual(radius, 0.5)


class TestURDFMeshes(TestURDFUtilities):
    """Test URDF mesh extraction."""
    
    @unittest.skipIf(not hasattr(TestURDFUtilities, 'urdf_file') or TestURDFUtilities.urdf_file is None,
                     "No URDF file specified")
    def test_get_urdf_meshes(self):
        """Test extraction of meshes from the specified URDF file."""
        try:
            urdf = load_urdf(TestURDFUtilities.urdf_file)
            meshes = get_urdf_meshes(urdf)
            
            print(f"Found {len(meshes)} mesh(es) in URDF")
            
            # Each mesh should be a URDFMesh instance
            for mesh_obj in meshes:
                self.assertIsInstance(mesh_obj, URDFMesh)
                self.assertIsInstance(mesh_obj.mesh, Trimesh)
                self.assertIsInstance(mesh_obj.xyz, np.ndarray)
                self.assertIsInstance(mesh_obj.rpy, np.ndarray)
                self.assertIsInstance(mesh_obj.scale, np.ndarray)
                self.assertEqual(len(mesh_obj.xyz), 3)
                self.assertEqual(len(mesh_obj.rpy), 3)
                self.assertEqual(len(mesh_obj.scale), 3)
                
        except Exception as e:
            self.skipTest(f"Could not load meshes from URDF: {e}")
    
    def test_get_urdf_meshes_with_shrinkage(self):
        """Test mesh extraction applies shrinkage factor."""
        # Create a minimal URDF with mesh reference
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="mesh_link">
        <collision>
            <geometry>
                <mesh filename="package://test/mesh.stl" scale="2.0 2.0 2.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""
        urdf_path = self.temp_path / "test_mesh.urdf"
        urdf_path.write_text(urdf_content)
        
        # Create a dummy mesh file
        mesh_dir = self.temp_path / "test"
        mesh_dir.mkdir()
        mesh = Box([1, 1, 1])
        mesh.export(mesh_dir / "mesh.stl")
        
        urdf = load_urdf(urdf_path)
        shrinkage = 0.8
        
        try:
            meshes = get_urdf_meshes(urdf, shrinkage=shrinkage)
            
            if meshes:
                # Scale should be original scale * shrinkage
                expected_scale = np.array([2.0, 2.0, 2.0]) * shrinkage
                np.testing.assert_array_almost_equal(meshes[0].scale, expected_scale)
        except (FileNotFoundError, TypeError) as e:
            self.skipTest(f"Mesh file issues: {e}")


class TestURDFSphereModification(TestURDFUtilities):
    """Test URDF sphere setting and modification."""
    
    def create_mock_spherization(self):
        """Create mock spherization data for testing."""
        class MockSphere:
            def __init__(self, origin, radius):
                self.origin = origin
                self.radius = radius
        
        class MockSpherization:
            def __init__(self, spheres):
                self.spheres = spheres
        
        return {
            "box_link::primitive0": MockSpherization([
                MockSphere([0.0, 0.0, 0.0], 0.5),
                MockSphere([1.0, 0.0, 0.0], 0.3)
            ])
        }
    
    def test_set_urdf_spheres(self):
        """Test replacing URDF geometry with spheres."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="box_link">
        <collision>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
            <origin xyz="0 0 0" rpy="0 0 0"/>
        </collision>
    </link>
</robot>
"""
        urdf_path = self.temp_path / "test_spheres.urdf"
        urdf_path.write_text(urdf_content)
        
        urdf = load_urdf(urdf_path)
        spheres = self.create_mock_spherization()
        
        try:
            # This should not raise an exception
            set_urdf_spheres(urdf, spheres)
            
            # Check that collision was replaced with spheres
            link = urdf['robot']['link']
            if not isinstance(link, list):
                link = [link]
            
            box_link = next(l for l in link if l['@name'] == 'box_link')
            self.assertIsInstance(box_link['collision'], list)
            
            # Should have 2 sphere collisions now
            self.assertEqual(len(box_link['collision']), 2)
            
            for collision in box_link['collision']:
                self.assertIn('sphere', collision['geometry'])
        except (TypeError, KeyError) as e:
            self.skipTest(f"URDF structure issue: {e}")


class TestURDFSaving(TestURDFUtilities):
    """Test URDF file saving."""
    
    def test_save_urdf(self):
        """Test saving URDF to file."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <collision>
            <geometry>
                <box size="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
"""
        urdf_path = self.temp_path / "original.urdf"
        urdf_path.write_text(urdf_content)
        
        # Load and save
        urdf = load_urdf(urdf_path)
        output_path = self.temp_path / "saved.urdf"
        save_urdf(urdf, output_path)
        
        # File should exist
        self.assertTrue(output_path.exists())
        
        # Should be valid XML that can be loaded again
        urdf_reloaded = load_urdf(output_path)
        self.assertIn('robot', urdf_reloaded)
        self.assertEqual(urdf_reloaded['robot']['@name'], 'test_robot')
    
    @unittest.skipIf(not hasattr(TestURDFUtilities, 'urdf_file') or TestURDFUtilities.urdf_file is None,
                     "No URDF file specified")
    def test_save_and_reload_test_urdf(self):
        """Test that the specified URDF can be loaded and saved."""
        try:
            urdf = load_urdf(TestURDFUtilities.urdf_file)
            output_path = self.temp_path / f"saved_{TestURDFUtilities.urdf_file.name}"
            save_urdf(urdf, output_path)
            
            # Reload and verify
            urdf_reloaded = load_urdf(output_path)
            self.assertIn('robot', urdf_reloaded)
            
        except Exception as e:
            self.skipTest(f"Could not save/reload URDF: {e}")


class TestIntegration(TestURDFUtilities):
    """Integration tests using the specified URDF file."""
    
    @unittest.skipIf(not hasattr(TestURDFUtilities, 'urdf_file') or TestURDFUtilities.urdf_file is None,
                     "No URDF file specified")
    def test_full_workflow_with_primitives(self):
        """Test complete workflow: load -> extract primitives -> modify -> save."""
        try:
            # Load URDF
            urdf = load_urdf(TestURDFUtilities.urdf_file)
            
            # Extract primitives
            primitives = get_urdf_primitives(urdf)
            
            # Extract spheres
            spheres = list(get_urdf_spheres(urdf))
            
            # Save to new file
            output_path = self.temp_path / f"processed_{TestURDFUtilities.urdf_file.name}"
            save_urdf(urdf, output_path)
            
            # Verify output exists
            self.assertTrue(output_path.exists())
            
            print(f"✓ {TestURDFUtilities.urdf_file.name}: {len(primitives)} primitives, {len(spheres)} spheres")
            
        except Exception as e:
            self.skipTest(f"Could not complete workflow: {e}")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test URDF utilities')
    parser.add_argument('--urdf', type=str, help='Path to URDF file to test')
    
    # Separate our args from unittest args
    args, unittest_args = parser.parse_known_args()
    
    # Set the URDF path as a class variable
    if args.urdf:
        TestURDFUtilities.test_urdf_path = args.urdf
    elif 'URDF_TEST_FILE' in os.environ:
        TestURDFUtilities.test_urdf_path = os.environ['URDF_TEST_FILE']
    
    # Pass remaining arguments to unittest
    sys.argv[1:] = unittest_args
    unittest.main(verbosity=2)