import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Optional
import urllib.parse
from dataclasses import dataclass

# Import foam modules
import foam.external
import foam.model
from foam.external import (
    read_spherization_file,
    compute_spheres_helper,
    check_valid_for_spherization,
    compute_spheres,
    simplify,
    simplify_manifold,
    manifold,
    manifold_plus
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
        if repo_root is None:
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
            
            mesh_files = []
            has_visual = False
            has_collision = False
            
            for mesh_elem in root.findall('.//mesh'):
                filename = mesh_elem.get('filename')
                if filename:
                    mesh_path = self._resolve_mesh_path(filename, urdf_path)
                    if mesh_path and mesh_path.exists():
                        mesh_files.append(mesh_path)
                    
                    # Check context
                    parent = mesh_elem
                    for _ in range(5):  # Check up to 5 levels up
                        parent = list(root.iter())[list(root.iter()).index(parent) - 1] if list(root.iter()).index(parent) > 0 else None
                        if parent is None:
                            break
                        if parent.tag == 'visual':
                            has_visual = True
                            break
                        elif parent.tag == 'collision':
                            has_collision = True
                            break
            
            return URDFTestCase(
                urdf_path=urdf_path,
                name=robot_name,
                mesh_files=list(set(mesh_files)),
                has_visual_meshes=has_visual,
                has_collision_meshes=has_collision
            )
        
        except Exception as e:
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


class TestReadSpherizationFile(unittest.TestCase):
    """Test read_spherization_file function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test.sph"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_read_spherization_file_basic(self):
        """Test reading a basic spherization file."""
        content = """Num Spheres: 2
Best Error: 0.1
Worst Error: 0.5
Mean Error: 0.3
0.0 0.0 0.0 1.0 0.2
1.0 1.0 1.0 0.5 0.1
"""
        self.test_file.write_text(content)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = read_spherization_file(self.test_file, offset)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].spheres), 2)
        self.assertAlmostEqual(result[0].mean, 0.3)
        self.assertAlmostEqual(result[0].best, 0.1)
        self.assertAlmostEqual(result[0].worst, 0.5)
    
    def test_read_spherization_file_multiple_levels(self):
        """Test reading spherization file with multiple levels."""
        content = """Num Spheres: 2
Best Error: 0.1
Worst Error: 0.5
Mean Error: 0.3
0.0 0.0 0.0 1.0 0.2
1.0 1.0 1.0 0.5 0.1
Num Spheres: 1
Best Error: 0.05
Worst Error: 0.2
Mean Error: 0.1
2.0 2.0 2.0 0.8 0.05
"""
        self.test_file.write_text(content)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = read_spherization_file(self.test_file, offset)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].spheres), 2)
        self.assertEqual(len(result[1].spheres), 1)
    
    def test_read_spherization_file_filters_zero_radius(self):
        """Test that spheres with zero radius are filtered out."""
        content = """Num Spheres: 3
Best Error: 0.1
Worst Error: 0.5
Mean Error: 0.3
0.0 0.0 0.0 1.0 0.2
1.0 1.0 1.0 0.0 0.0
2.0 2.0 2.0 0.5 0.1
"""
        self.test_file.write_text(content)
        offset = np.array([0.0, 0.0, 0.0])
        
        result = read_spherization_file(self.test_file, offset)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].spheres), 2)  # Zero radius sphere filtered


class TestComputeSpheresHelper(unittest.TestCase):
    """Test compute_spheres_helper function."""
    
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    @patch('foam.external.read_spherization_file')
    @patch('foam.external.remove_file')
    def test_compute_spheres_helper_success(self, mock_remove, mock_read, mock_tempmesh, mock_run):
        """Test successful sphere computation."""
        # Setup mocks
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_mesh.bounds = np.array([[0, 0, 0], [10, 10, 10]])
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/test.obj")
        mock_input_path_mock = Mock()
        mock_input_path_mock.parent = Path("/tmp")
        mock_input_path_mock.stem = "test"
        
        mock_tempmesh.return_value.__enter__ = Mock(return_value=(mock_input_file, mock_input_path_mock))
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        mock_spherization = Mock()
        mock_read.return_value = [mock_spherization]
        
        command = ["makeTreeMedial", "-nopause"]
        result = compute_spheres_helper(mock_mesh, command, "medial")
        
        self.assertEqual(result, [mock_spherization])
        mock_run.assert_called_once()
    
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_compute_spheres_helper_failure(self, mock_tempmesh, mock_run):
        """Test sphere computation failure handling."""
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_mesh.bounds = np.array([[0, 0, 0], [10, 10, 10]])
        
        mock_input_file = Mock()
        mock_input_path = Mock()
        mock_input_path.parent = Path("/tmp")
        mock_input_path.stem = "test"
        
        mock_tempmesh.return_value.__enter__ = Mock(return_value=(mock_input_file, mock_input_path))
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = b"Error message"
        mock_run.return_value = mock_result
        
        command = ["makeTreeMedial", "-nopause"]
        
        with self.assertRaises(RuntimeError):
            compute_spheres_helper(mock_mesh, command, "medial")


class TestCheckValidForSpherization(unittest.TestCase):
    """Test check_valid_for_spherization function."""
    
    @patch('foam.external.compute_spheres_helper')
    def test_check_valid_medial_method(self, mock_helper):
        """Test validation with medial method."""
        mock_mesh = Mock()
        mock_helper.return_value = [Mock()]
        
        result = check_valid_for_spherization("medial", mock_mesh)
        
        self.assertTrue(result)
        mock_helper.assert_called_once()
    
    @patch('foam.external.compute_spheres_helper')
    def test_check_valid_grid_method(self, mock_helper):
        """Test validation with grid method."""
        mock_mesh = Mock()
        mock_helper.return_value = [Mock()]
        
        result = check_valid_for_spherization("grid", mock_mesh)
        
        self.assertTrue(result)
    
    @patch('foam.external.compute_spheres_helper')
    def test_check_valid_failure(self, mock_helper):
        """Test validation failure."""
        mock_mesh = Mock()
        mock_helper.side_effect = Exception("Invalid mesh")
        
        result = check_valid_for_spherization("medial", mock_mesh)
        
        self.assertFalse(result)


class TestComputeSpheres(unittest.TestCase):
    """Test compute_spheres function with various parameters."""
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_medial_default(self, mock_helper):
        """Test compute_spheres with medial method and default parameters."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(mock_mesh, method="medial")
        
        self.assertEqual(result, mock_spherization)
        mock_helper.assert_called_once()
        
        # Check command contains expected flags
        call_args = mock_helper.call_args[0]
        command = call_args[1]
        
        self.assertIn('-nopause', command)
        self.assertIn('-branch', command)
        self.assertIn('-depth', command)
        self.assertIn('-testerLevels', command)
        self.assertIn('-numCover', command)
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_grid_method(self, mock_helper):
        """Test compute_spheres with grid method."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(mock_mesh, method="grid", depth=2)
        
        self.assertEqual(result, mock_spherization)
        call_args = mock_helper.call_args[0]
        command = call_args[1]
        
        self.assertIn('2', command)  # Depth value
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_hubbard_method(self, mock_helper):
        """Test compute_spheres with hubbard method."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(mock_mesh, method="hubbard", num_samples=1000)
        
        self.assertEqual(result, mock_spherization)
        call_args = mock_helper.call_args[0]
        command = call_args[1]
        
        self.assertIn('-numSamples', command)
        self.assertIn('1000', command)
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_octree_method(self, mock_helper):
        """Test compute_spheres with octree method."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(mock_mesh, method="octree")
        
        self.assertEqual(result, mock_spherization)
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_spawn_method(self, mock_helper):
        """Test compute_spheres with spawn method."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(mock_mesh, method="spawn")
        
        self.assertEqual(result, mock_spherization)
    
    @patch('foam.external.compute_spheres_helper')
    def test_compute_spheres_with_optimization_flags(self, mock_helper):
        """Test compute_spheres with various optimization flags."""
        mock_mesh = Mock()
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        result = compute_spheres(
            mock_mesh,
            method="medial",
            expand=True,
            merge=True,
            burst=True,
            optimise=True,
            balExcess=0.1
        )
        
        call_args = mock_helper.call_args[0]
        command = call_args[1]
        
        self.assertIn('-expand', command)
        self.assertIn('-merge', command)
        self.assertIn('-burst', command)
        self.assertIn('-balExcess', command)


class TestSimplify(unittest.TestCase):
    """Test simplify function."""
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_simplify_default_params(self, mock_tempmesh, mock_run, mock_load):
        """Test simplify with default parameters."""
        mock_mesh = Mock()
        mock_simplified = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_simplified
        
        result = simplify(mock_mesh)
        
        self.assertEqual(result, mock_simplified)
        mock_run.assert_called_once()
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_simplify_custom_params(self, mock_tempmesh, mock_run, mock_load):
        """Test simplify with custom parameters."""
        mock_mesh = Mock()
        mock_simplified = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_simplified
        
        result = simplify(mock_mesh, ratio=0.3, aggressiveness=5.0)
        
        self.assertEqual(result, mock_simplified)


class TestSimplifyManifold(unittest.TestCase):
    """Test simplify_manifold function."""
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_simplify_manifold_basic(self, mock_tempmesh, mock_run, mock_load):
        """Test simplify_manifold basic functionality."""
        mock_mesh = Mock()
        mock_simplified = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_simplified
        
        result = simplify_manifold(mock_mesh, ratio=0.7)
        
        self.assertEqual(result, mock_simplified)
        mock_run.assert_called_once()


class TestManifold(unittest.TestCase):
    """Test manifold function."""
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_manifold_default_leaves(self, mock_tempmesh, mock_run, mock_load):
        """Test manifold with default leaves parameter."""
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_manifold_mesh = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_manifold_mesh
        
        result = manifold(mock_mesh)
        
        self.assertEqual(result, mock_manifold_mesh)
        mock_run.assert_called_once()
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_manifold_custom_leaves(self, mock_tempmesh, mock_run, mock_load):
        """Test manifold with custom leaves parameter."""
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_manifold_mesh = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_manifold_mesh
        
        result = manifold(mock_mesh, leaves=2000)
        
        self.assertEqual(result, mock_manifold_mesh)


class TestManifoldPlus(unittest.TestCase):
    """Test manifold_plus function."""
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_manifold_plus_default_depth(self, mock_tempmesh, mock_run, mock_load):
        """Test manifold_plus with default depth."""
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_manifold_mesh = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_manifold_mesh
        
        result = manifold_plus(mock_mesh)
        
        self.assertEqual(result, mock_manifold_mesh)
        mock_run.assert_called_once()
    
    @patch('foam.external.load_mesh_file')
    @patch('foam.external.run')
    @patch('foam.external.tempmesh')
    def test_manifold_plus_custom_depth(self, mock_tempmesh, mock_run, mock_load):
        """Test manifold_plus with custom depth."""
        mock_mesh = Mock()
        mock_mesh.vertex_normals = np.array([[0, 1, 0]])
        mock_manifold_mesh = Mock()
        
        mock_input_file = Mock()
        mock_input_path = Path("/tmp/input.obj")
        mock_output_path = Path("/tmp/output.obj")
        
        mock_tempmesh.return_value.__enter__ = Mock(side_effect=[
            (mock_input_file, mock_input_path),
            (None, mock_output_path)
        ])
        mock_tempmesh.return_value.__exit__ = Mock(return_value=False)
        
        mock_load.return_value = mock_manifold_mesh
        
        result = manifold_plus(mock_mesh, depth=10)
        
        self.assertEqual(result, mock_manifold_mesh)


class TestURDFIntegration(unittest.TestCase):
    """Integration tests using a specific URDF file."""
    
    test_case = None
    
    def test_urdf_parsing(self):
        """Test that the specified URDF file can be parsed."""
        if not self.test_case:
            self.skipTest("No URDF file specified for testing")
        
        try:
            tree = ET.parse(self.test_case.urdf_path)
            root = tree.getroot()
            self.assertEqual(root.tag, 'robot', 
                           f"URDF {self.test_case.urdf_path} should have 'robot' as root tag")
        except ET.ParseError as e:
            self.fail(f"Failed to parse URDF {self.test_case.urdf_path}: {e}")
    
    def test_urdf_has_meshes(self):
        """Test that the URDF contains mesh references."""
        if not self.test_case:
            self.skipTest("No URDF file specified for testing")
        
        if not self.test_case.mesh_files:
            self.skipTest(f"URDF {self.test_case.name} contains no mesh files")
        
        self.assertGreater(len(self.test_case.mesh_files), 0,
                          "URDF should contain at least one mesh reference")
    
    @patch('foam.external.compute_spheres_helper')
    def test_spherization_methods_with_urdf_meshes(self, mock_helper):
        """Test all spherization methods with URDF meshes."""
        if not self.test_case or not self.test_case.mesh_files:
            self.skipTest("No URDF meshes available for testing")
        
        mock_spherization = [Mock()]
        mock_helper.return_value = mock_spherization
        
        methods = ["medial", "grid", "hubbard", "spawn", "octree"]
        
        # Test first available mesh with each method
        mock_mesh = Mock()
        mock_mesh.bounds = np.array([[0, 0, 0], [1, 1, 1]])
        
        for method in methods:
            with self.subTest(method=method):
                result = compute_spheres(mock_mesh, method=method, depth=1)
                self.assertEqual(result, mock_spherization)


def run_urdf_tests(urdf_path: str):
    """Run tests with a specific URDF file.
    
    Args:
        urdf_path: Path to URDF file to test with (required)
    """
    discovery = URDFDiscovery()
    urdf_file = Path(urdf_path)
    
    if not urdf_file.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        print(f"Please check the path and try again.")
        return False
    
    print(f"\n{'='*70}")
    print(f"Loading URDF: {urdf_file}")
    print(f"{'='*70}\n")
    
    try:
        test_case = discovery.analyze_urdf(urdf_file)
        print(f"Robot name: {test_case.name}")
        print(f"Mesh files found: {len(test_case.mesh_files)}")
        print(f"Has visual meshes: {test_case.has_visual_meshes}")
        print(f"Has collision meshes: {test_case.has_collision_meshes}")
        
        if test_case.mesh_files:
            print(f"\nMesh files ({len(test_case.mesh_files)}):")
            for i, mesh_file in enumerate(test_case.mesh_files, 1):
                exists_str = "✓" if mesh_file.exists() else "✗"
                print(f"  {i}. {exists_str} {mesh_file.name} ({mesh_file.suffix})")
        else:
            print("\nNo mesh files found in URDF")
        
        print(f"\n{'='*70}")
        print("Running Tests")
        print(f"{'='*70}\n")
        
        # Set the class-level test data
        TestURDFIntegration.test_case = test_case
        
    except Exception as e:
        print(f"Error analyzing URDF: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestReadSpherizationFile))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeSpheresHelper))
    suite.addTests(loader.loadTestsFromTestCase(TestCheckValidForSpherization))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeSpheres))
    suite.addTests(loader.loadTestsFromTestCase(TestSimplify))
    suite.addTests(loader.loadTestsFromTestCase(TestSimplifyManifold))
    suite.addTests(loader.loadTestsFromTestCase(TestManifold))
    suite.addTests(loader.loadTestsFromTestCase(TestManifoldPlus))
    suite.addTests(loader.loadTestsFromTestCase(TestURDFIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*70}")
    print("Test Summary")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"{'='*70}\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test foam.external module with a specific URDF file',
        epilog='Examples:\n'
               '  python3 test_external.py robot.urdf\n'
               '  python3 test_external.py path/to/robot.urdf\n'
               '  python3 test_external.py -v robot.urdf',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('urdf', help='Path to URDF file to test with (required)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output (shows detailed test information)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"FOAM External Module Test Suite")
    print(f"{'='*70}")
    print(f"Testing with URDF: {args.urdf}")
    print(f"{'='*70}\n")
    
    success = run_urdf_tests(args.urdf)
    
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)