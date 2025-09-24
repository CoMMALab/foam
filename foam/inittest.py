import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil
import json
import xml.etree.ElementTree as ET
from concurrent.futures import Future
import numpy as np
import glob
import os
from typing import Any, List, Dict, Optional
import urllib.parse
from dataclasses import dataclass

# Import foam modules without the problematic __init__ import
import foam.utility
import foam.external
import foam.model
import foam

# Import the classes directly from foam (not foam.__init__)
from foam import (
    smooth_manifold,
    spherize_mesh, 
    ParallelSpherizer,
    SpherizationDatabase,
    SpherizationHelper
)


@dataclass
class URDFTestCase:
    """Test case information for a URDF file."""
    urdf_path: Path
    name: str
    mesh_files: List[Path]
    has_visual_meshes: bool
    has_collision_meshes: bool


class URDFDiscovery:
    """Discover and analyze URDF files in the repository."""
    
    def __init__(self, repo_root: Path = None):
        """Initialize URDF discovery.
        
        Args:
            repo_root: Root directory of the repository. If None, attempts to auto-detect.
        """
        if repo_root is None:
            # Try to find repository root by looking for common indicators
            current_dir = Path.cwd()
            for potential_root in [current_dir] + list(current_dir.parents):
                if self._is_repo_root(potential_root):
                    repo_root = potential_root
                    break
            else:
                repo_root = current_dir
        
        self.repo_root = Path(repo_root)
    
    def _is_repo_root(self, path: Path) -> bool:
        """Check if a path looks like a repository root."""
        indicators = ['.git', 'setup.py', 'pyproject.toml', 'CMakeLists.txt', 'README.md']
        return any((path / indicator).exists() for indicator in indicators)
    
    def analyze_urdf(self, urdf_path: Path) -> URDFTestCase:
        """Analyze a URDF file to extract test-relevant information."""
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            robot_name = root.get('name', urdf_path.stem)
            
            # Find mesh files referenced in URDF
            mesh_files = []
            has_visual = False
            has_collision = False
            
            for mesh_elem in root.findall('.//mesh'):
                filename = mesh_elem.get('filename')
                if filename:
                    mesh_path = self._resolve_mesh_path(filename, urdf_path)
                    if mesh_path and mesh_path.exists():
                        mesh_files.append(mesh_path)
                    
                    # Check if in visual or collision context
                    parent = mesh_elem.getparent()
                    while parent is not None:
                        if parent.tag == 'visual':
                            has_visual = True
                            break
                        elif parent.tag == 'collision':
                            has_collision = True
                            break
                        parent = parent.getparent()
            
            return URDFTestCase(
                urdf_path=urdf_path,
                name=robot_name,
                mesh_files=list(set(mesh_files)),
                has_visual_meshes=has_visual,
                has_collision_meshes=has_collision
            )
        
        except Exception as e:
            # Return minimal test case for invalid URDFs
            return URDFTestCase(
                urdf_path=urdf_path,
                name=urdf_path.stem,
                mesh_files=[],
                has_visual_meshes=False,
                has_collision_meshes=False
            )
    
    def _resolve_mesh_path(self, filename: str, urdf_path: Path) -> Optional[Path]:
        """Resolve mesh file path from URDF filename attribute."""
        try:
            if filename.startswith('package://'):
                # Simple package resolution
                package_path = filename[10:]
                mesh_path = urdf_path.parent / package_path
            elif filename.startswith('file://'):
                parsed = urllib.parse.urlparse(filename)
                mesh_path = Path(parsed.path)
            else:
                mesh_path = urdf_path.parent / filename
            
            return mesh_path.resolve()
        except:
            return None


class TestSmoothManifold(unittest.TestCase):
    
    @patch('foam.manifold')
    @patch('foam.simplify_manifold')
    @patch('foam.smooth_mesh')
    def test_smooth_manifold_default_params(self, mock_smooth, mock_simplify, mock_manifold):
        """Test smooth_manifold with default parameters."""
        mock_mesh = Mock()
        mock_manifold.return_value = mock_mesh
        mock_simplify.return_value = mock_mesh
        
        result = smooth_manifold(mock_mesh)
        
        mock_manifold.assert_called_once_with(mock_mesh, 1000)
        mock_simplify.assert_called_once_with(mock_mesh, 0.2)
        mock_smooth.assert_called_once_with(mock_mesh)
        self.assertEqual(result, mock_mesh)
    
    @patch('foam.manifold')
    @patch('foam.simplify_manifold')
    @patch('foam.smooth_mesh')
    def test_smooth_manifold_custom_params(self, mock_smooth, mock_simplify, mock_manifold):
        """Test smooth_manifold with custom parameters."""
        mock_mesh = Mock()
        mock_manifold.return_value = mock_mesh
        mock_simplify.return_value = mock_mesh
        
        result = smooth_manifold(mock_mesh, manifold_leaves=2000, ratio=0.5)
        
        mock_manifold.assert_called_once_with(mock_mesh, 2000)
        mock_simplify.assert_called_once_with(mock_mesh, 0.5)
        mock_smooth.assert_called_once_with(mock_mesh)
        self.assertEqual(result, mock_mesh)


class TestSpherizeMesh(unittest.TestCase):
    
    def setUp(self):
        self.mock_mesh = Mock()
        self.mock_mesh.copy.return_value = self.mock_mesh
        self.mock_mesh.bounds = np.array([[0, 0, 0], [10, 10, 10]])
        
    @patch('foam.load_mesh_file')
    @patch('foam.check_valid_for_spherization')
    @patch('foam.compute_spheres')
    def test_spherize_mesh_with_path(self, mock_compute, mock_check, mock_load):
        """Test spherize_mesh with file path input."""
        mock_load.return_value = self.mock_mesh
        mock_check.return_value = True
        mock_spheres = [Mock()]
        mock_compute.return_value = mock_spheres
        
        mesh_path = Path("test_mesh.obj")
        spherization_kwargs = {'method': 'medial'}
        
        result = spherize_mesh("test", mesh_path, spherization_kwargs=spherization_kwargs)
        
        mock_load.assert_called_once_with(mesh_path)
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)
    
    @patch('foam.check_valid_for_spherization')
    @patch('foam.compute_spheres')
    def test_spherize_mesh_with_trimesh_object(self, mock_compute, mock_check):
        """Test spherize_mesh with trimesh object input."""
        mock_check.return_value = True
        mock_spheres = [Mock()]
        mock_compute.return_value = mock_spheres
        
        spherization_kwargs = {'method': 'medial'}
        
        result = spherize_mesh("test", self.mock_mesh, spherization_kwargs=spherization_kwargs)
        
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)


class TestParallelSpherizer(unittest.TestCase):
    
    def setUp(self):
        self.spherizer = ParallelSpherizer(threads=2)
    
    def tearDown(self):
        self.spherizer.executor.shutdown(wait=True)
    
    @patch('foam.spherize_mesh')
    def test_spherize_mesh_parallel(self, mock_spherize):
        """Test parallel spherization."""
        mock_result = [Mock()]
        mock_spherize.return_value = mock_result
        mock_mesh = Mock()
        
        future = self.spherizer.spherize_mesh("test", mock_mesh)
        result = future.result()
        
        self.assertIsInstance(future, Future)
        self.assertEqual(result, mock_result)
        self.assertIn("test", self.spherizer.waiting)


class TestSpherizationDatabase(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.json"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_creation_new_file(self):
        """Test database creation with new file."""
        db = SpherizationDatabase(self.db_path)
        
        self.assertEqual(db.db, {})
        self.assertEqual(db.path, self.db_path)
    
    def test_add_spherization(self):
        """Test adding spherization to database."""
        db = SpherizationDatabase(self.db_path)
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        
        db.add("mesh1", 1, 0, spherization)
        
        self.assertIn("mesh1", db.db)
        self.assertIn(1, db.db["mesh1"])
        self.assertIn(0, db.db["mesh1"][1])
        self.assertEqual(db.db["mesh1"][1][0], spherization)
    
    def test_exists(self):
        """Test checking if spherization exists in database."""
        db = SpherizationDatabase(self.db_path)
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        db.add("mesh1", 1, 0, spherization)
        
        self.assertTrue(db.exists("mesh1", 1, 0))
        self.assertFalse(db.exists("mesh2", 1, 0))
        self.assertFalse(db.exists("mesh1", 2, 0))
        self.assertFalse(db.exists("mesh1", 1, 1))


class TestSpherizationHelper(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.json"
        self.helper = SpherizationHelper(self.db_path, threads=2)
    
    def tearDown(self):
        self.helper.ps.executor.shutdown(wait=True)
        shutil.rmtree(self.temp_dir)
    
    def test_spherize_mesh_not_in_db(self):
        """Test spherizing mesh not in database."""
        mock_mesh = Mock()
        
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        self.assertIn("test_mesh", self.helper.ps.waiting)
    
    def test_spherize_mesh_already_in_db(self):
        """Test spherizing mesh already in database."""
        mock_mesh = Mock()
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        self.helper.db.add("test_mesh", 8, 1, spherization)
        
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        # Should not be added to waiting queue
        self.assertNotIn("test_mesh", self.helper.ps.waiting)


class TestURDFIntegration(unittest.TestCase):
    """Integration tests using specific URDF files."""
    
    # Class variables to be set when testing with specific URDF
    test_cases = []
    urdf_files = []
    
    def test_urdf_parsing(self):
        """Test that the specified URDF file can be parsed."""
        if not self.test_cases:
            self.skipTest("No URDF file specified for testing")
        
        for test_case in self.test_cases:
            with self.subTest(urdf=test_case.name):
                try:
                    tree = ET.parse(test_case.urdf_path)
                    root = tree.getroot()
                    self.assertEqual(root.tag, 'robot', f"URDF {test_case.urdf_path} should have 'robot' as root tag")
                except ET.ParseError as e:
                    self.fail(f"Failed to parse URDF {test_case.urdf_path}: {e}")

    @patch('foam.load_mesh_file')
    @patch('foam.check_valid_for_spherization')  
    @patch('foam.compute_spheres')
    def test_spherize_urdf_meshes(self, mock_compute, mock_check, mock_load):
        """Test spherization with meshes from the specified URDF."""
        if not self.test_cases:
            self.skipTest("No URDF file specified for testing")
        
        # Setup mocks
        mock_mesh = Mock()
        mock_mesh.copy.return_value = mock_mesh
        mock_mesh.bounds = np.array([[0, 0, 0], [1, 1, 1]])
        mock_load.return_value = mock_mesh
        mock_check.return_value = True
        mock_spheres = [Mock()]
        mock_compute.return_value = mock_spheres
        
        tested_meshes = 0
        for test_case in self.test_cases:
            for mesh_file in test_case.mesh_files[:3]:  # Limit to first 3 meshes per URDF
                if mesh_file.exists():
                    with self.subTest(urdf=test_case.name, mesh=mesh_file.name):
                        result = spherize_mesh(
                            f"{test_case.name}_{mesh_file.stem}",
                            mesh_file,
                            spherization_kwargs={'method': 'medial'}
                        )
                        self.assertEqual(result, mock_spheres)
                        tested_meshes += 1
        
        if tested_meshes == 0:
            self.skipTest("No existing mesh files found in the specified URDF")
    
    def test_spherization_helper_with_urdf_meshes(self):
        """Test SpherizationHelper with URDF-derived meshes."""
        if not self.test_cases:
            self.skipTest("No URDF file specified for testing")
        
        temp_dir = tempfile.mkdtemp()
        try:
            db_path = Path(temp_dir) / "urdf_test_db.json"
            helper = SpherizationHelper(db_path, threads=2)
            
            tested_meshes = 0
            for test_case in self.test_cases:
                for mesh_file in test_case.mesh_files[:2]:  # Limit testing
                    if mesh_file.exists():
                        mesh_name = f"{test_case.name}_{mesh_file.stem}"
                        
                        # Test queueing
                        mock_mesh = Mock()
                        helper.spherize_mesh(mesh_name, mock_mesh)
                        tested_meshes += 1
            
            helper.ps.executor.shutdown(wait=True)
            
            if tested_meshes == 0:
                self.skipTest("No existing mesh files found in the specified URDF")
                
        finally:
            shutil.rmtree(temp_dir)


class TestURDFMeshFormats(unittest.TestCase):
    """Test handling of different mesh formats found in URDFs."""
    
    # Class variable to be set when testing with specific URDF
    test_cases = []
    
    def test_mesh_format_support(self):
        """Test that various mesh formats in the URDF are recognized."""
        if not self.test_cases:
            self.skipTest("No URDF file specified for testing")
        
        supported_formats = {'.stl', '.obj', '.dae', '.ply', '.mesh'}
        found_formats = set()
        
        for test_case in self.test_cases:
            for mesh_file in test_case.mesh_files:
                found_formats.add(mesh_file.suffix.lower())
        
        if found_formats:
            # Check that we found some supported formats
            supported_found = found_formats.intersection(supported_formats)
            self.assertTrue(len(supported_found) > 0, 
                          f"Should find some supported mesh formats. Found: {found_formats}")
        else:
            self.skipTest("No mesh files found in the specified URDF")


def run_urdf_tests(specific_urdf: str):
    """Run tests with a specific URDF file.
    
    Args:
        specific_urdf: Path to URDF file to test with (required)
    """
    discovery = URDFDiscovery()
    urdf_path = Path(specific_urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {specific_urdf}")
        print(f"Please check the path and try again.")
        return False
    
    print(f"Loading URDF: {urdf_path}")
    try:
        test_case = discovery.analyze_urdf(urdf_path)
        print(f"Robot name: {test_case.name}")
        print(f"Mesh files found: {len(test_case.mesh_files)}")
        print(f"Has visual meshes: {test_case.has_visual_meshes}")
        print(f"Has collision meshes: {test_case.has_collision_meshes}")
        
        if test_case.mesh_files:
            print("Mesh files:")
            for mesh_file in test_case.mesh_files:
                exists_str = "✓" if Path(mesh_file).exists() else "✗"
                print(f"  {exists_str} {mesh_file}")
        
        print("\nRunning tests...\n")
        
        # Set the class-level test data for URDF-specific tests
        TestURDFIntegration.test_cases = [test_case]
        TestURDFIntegration.urdf_files = [urdf_path]
        TestURDFMeshFormats.test_cases = [test_case]
        
    except Exception as e:
        print(f"Error analyzing URDF: {e}")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSmoothManifold))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizeMesh))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelSpherizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizationDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizationHelper))
    suite.addTests(loader.loadTestsFromTestCase(TestURDFIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestURDFMeshFormats))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test foam functionality with a specific URDF file',
        epilog='Examples:\n'
               '  python3 inittest.py robot.urdf         # Test with specific URDF\n'
               '  python3 inittest.py path/to/robot.urdf # Test with URDF at path',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('urdf', help='Path to URDF file to test with (required)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Testing with URDF: {args.urdf}")
    success = run_urdf_tests(args.urdf)
    
    if success:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed!")
        sys.exit(1)