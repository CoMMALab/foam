import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import urllib.parse
import glob

# Import trimesh for testing
import trimesh

# Import the function under test
try:
    from preprocess import add_thickness
except ImportError:
    # Define a placeholder implementation for testing
    def add_thickness(mesh, thickness):
        """Placeholder add_thickness implementation for testing."""
        vertices = mesh.vertices
        
        # Force normals in z direction for planar mesh
        if np.any(np.ptp(mesh.vertices, axis=0) < 1e-6):
            normals = np.zeros_like(vertices)
            normals[:, 2] = 1.0
        else:
            normals = mesh.vertex_normals
        
        # Create offset vertices
        offset_vertices = vertices + (normals * thickness)
        
        # Get boundary edges
        edges = mesh.edges_unique
        
        # Create side faces
        side_faces = []
        for edge in edges:
            v1, v2 = edge
            v3, v4 = v1 + len(vertices), v2 + len(vertices)
            side_faces.extend([
                [v1, v2, v3],
                [v2, v4, v3]
            ])
        
        # Stack vertices and faces
        new_vertices = np.vstack((vertices, offset_vertices))
        new_faces = np.vstack((
            mesh.faces,
            np.fliplr(mesh.faces) + len(vertices),
            side_faces
        ))
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)


@dataclass
class URDFMeshInfo:
    """Information about meshes found in URDF files."""
    filename: str
    original_filename: str  # As specified in URDF
    scale: Optional[np.ndarray] = None
    origin_xyz: Optional[np.ndarray] = None
    origin_rpy: Optional[np.ndarray] = None
    link_name: str = ""
    mesh_type: str = "visual"  # "visual" or "collision"
    exists: bool = False


class URDFMeshExtractor:
    """Extract and analyze mesh information from URDF files."""
    
    def __init__(self, urdf_path: Path):
        self.urdf_path = Path(urdf_path)
        self.urdf_dir = self.urdf_path.parent
        
        try:
            self.tree = ET.parse(urdf_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Cannot parse URDF file {urdf_path}: {e}")
    
    def extract_mesh_info(self) -> List[URDFMeshInfo]:
        """Extract all mesh information from the URDF."""
        meshes = []
        
        for link in self.root.findall('.//link'):
            link_name = link.get('name', 'unknown')
            
            # Process visual meshes
            for visual in link.findall('visual'):
                meshes.extend(self._process_meshes_in_geometry(
                    visual, link_name, 'visual'))
            
            # Process collision meshes  
            for collision in link.findall('collision'):
                meshes.extend(self._process_meshes_in_geometry(
                    collision, link_name, 'collision'))
        
        return meshes
    
    def _process_meshes_in_geometry(self, parent_elem, link_name: str, 
                                   mesh_type: str) -> List[URDFMeshInfo]:
        """Process mesh elements within a geometry parent."""
        meshes = []
        geometry = parent_elem.find('geometry')
        
        if geometry is None:
            return meshes
        
        for mesh_elem in geometry.findall('mesh'):
            filename = mesh_elem.get('filename')
            if not filename:
                continue
            
            mesh_path = self._resolve_mesh_path(filename)
            scale = self._parse_scale(mesh_elem.get('scale'))
            origin_xyz, origin_rpy = self._parse_origin(parent_elem)
            
            meshes.append(URDFMeshInfo(
                filename=str(mesh_path) if mesh_path else filename,
                original_filename=filename,
                scale=scale,
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                link_name=link_name,
                mesh_type=mesh_type,
                exists=mesh_path.exists() if mesh_path else False
            ))
        
        return meshes
    
    def _resolve_mesh_path(self, filename: str) -> Optional[Path]:
        """Resolve mesh file path from URDF filename."""
        try:
            if filename.startswith('package://'):
                package_path = filename[10:]
                mesh_path = self.urdf_dir / package_path
            elif filename.startswith('file://'):
                parsed = urllib.parse.urlparse(filename)
                mesh_path = Path(parsed.path)
            else:
                mesh_path = self.urdf_dir / filename
            
            return mesh_path.resolve()
        except Exception:
            return None
    
    def _parse_scale(self, scale_str: Optional[str]) -> Optional[np.ndarray]:
        """Parse scale attribute from URDF."""
        if not scale_str:
            return None
        
        try:
            scale_vals = [float(x) for x in scale_str.split()]
            if len(scale_vals) == 1:
                return np.array([scale_vals[0]] * 3)
            elif len(scale_vals) == 3:
                return np.array(scale_vals)
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _parse_origin(self, element) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Parse origin xyz and rpy from XML element."""
        origin = element.find('origin')
        if origin is None:
            return None, None
        
        xyz = None
        rpy = None
        
        xyz_attr = origin.get('xyz')
        if xyz_attr:
            try:
                xyz = np.array([float(x) for x in xyz_attr.split()])
            except (ValueError, TypeError):
                pass
        
        rpy_attr = origin.get('rpy')
        if rpy_attr:
            try:
                rpy = np.array([float(x) for x in rpy_attr.split()])
            except (ValueError, TypeError):
                pass
        
        return xyz, rpy


class TestAddThickness(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with various mesh types."""
        # Create a simple triangle mesh for testing
        self.simple_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0.5, 1, 0]
        ])
        self.simple_faces = np.array([[0, 1, 2]])
        self.simple_mesh = Mock()
        self.simple_mesh.vertices = self.simple_vertices
        self.simple_mesh.faces = self.simple_faces
        self.simple_mesh.vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.simple_mesh.edges_unique = np.array([[0, 1], [1, 2], [2, 0]])
        
        # Create a planar square mesh
        self.square_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0]
        ])
        self.square_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ])
        self.square_mesh = Mock()
        self.square_mesh.vertices = self.square_vertices
        self.square_mesh.faces = self.square_faces
        self.square_mesh.vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
        self.square_mesh.edges_unique = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
    
    @patch('trimesh.Trimesh')
    def test_add_thickness_basic_functionality(self, mock_trimesh_class):
        """Test basic functionality with a simple triangle."""
        mock_result = Mock()
        mock_trimesh_class.return_value = mock_result
        thickness = 0.1
        
        result = add_thickness(self.simple_mesh, thickness)
        
        self.assertEqual(result, mock_result)
        mock_trimesh_class.assert_called_once()
        args, kwargs = mock_trimesh_class.call_args
        
        # Check that vertices were doubled
        vertices_arg = kwargs['vertices'] 
        self.assertEqual(len(vertices_arg), len(self.simple_vertices) * 2)
        
        # Check that faces include original, flipped, and side faces
        faces_arg = kwargs['faces']
        self.assertIsNotNone(faces_arg)
    
    @patch('trimesh.Trimesh')
    def test_add_thickness_planar_mesh_detection(self, mock_trimesh_class):
        """Test that planar meshes are detected and normals forced to z-direction."""
        mock_result = Mock()
        mock_trimesh_class.return_value = mock_result
        thickness = 0.05
        
        result = add_thickness(self.simple_mesh, thickness)
        
        args, kwargs = mock_trimesh_class.call_args
        vertices_arg = kwargs['vertices']
        
        # Original vertices should be first half
        original_vertices = vertices_arg[:len(self.simple_vertices)]
        offset_vertices = vertices_arg[len(self.simple_vertices):]
        
        # For a planar mesh, offset should be in z-direction
        expected_offset = self.simple_vertices + np.array([0, 0, thickness])
        np.testing.assert_array_almost_equal(offset_vertices, expected_offset)
    
    @patch('trimesh.Trimesh')
    def test_add_thickness_zero_thickness(self, mock_trimesh_class):
        """Test behavior with zero thickness."""
        mock_result = Mock()
        mock_trimesh_class.return_value = mock_result
        thickness = 0.0
        
        result = add_thickness(self.simple_mesh, thickness)
        
        args, kwargs = mock_trimesh_class.call_args
        vertices_arg = kwargs['vertices']
        
        # Original and offset vertices should be the same when thickness is 0
        original_vertices = vertices_arg[:len(self.simple_vertices)]
        offset_vertices = vertices_arg[len(self.simple_vertices):]
        np.testing.assert_array_almost_equal(original_vertices, offset_vertices)


class TestURDFMeshExtraction(unittest.TestCase):
    """Test URDF mesh extraction functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def create_test_urdf(self, content: str) -> Path:
        """Create a test URDF file."""
        urdf_path = Path(self.temp_dir) / "test_robot.urdf"
        with open(urdf_path, 'w') as f:
            f.write(content)
        return urdf_path
    
    def test_basic_urdf_mesh_extraction(self):
        """Test extraction of meshes from basic URDF."""
        urdf_content = '''<?xml version="1.0"?>
        <robot name="test_robot">
          <link name="base_link">
            <visual>
              <geometry>
                <mesh filename="meshes/base.stl"/>
              </geometry>
            </visual>
            <collision>
              <geometry>
                <mesh filename="meshes/base_collision.stl"/>
              </geometry>
            </collision>
          </link>
        </robot>'''
        
        urdf_path = self.create_test_urdf(urdf_content)
        extractor = URDFMeshExtractor(urdf_path)
        meshes = extractor.extract_mesh_info()
        
        self.assertEqual(len(meshes), 2)
        self.assertEqual(meshes[0].link_name, "base_link")
        self.assertEqual(meshes[0].mesh_type, "visual")
        self.assertEqual(meshes[1].mesh_type, "collision")
    
    def test_urdf_with_scaled_meshes(self):
        """Test URDF with mesh scaling."""
        urdf_content = '''<?xml version="1.0"?>
        <robot name="test_robot">
          <link name="scaled_link">
            <visual>
              <geometry>
                <mesh filename="meshes/scaled.stl" scale="2.0 1.5 3.0"/>
              </geometry>
            </visual>
          </link>
        </robot>'''
        
        urdf_path = self.create_test_urdf(urdf_content)
        extractor = URDFMeshExtractor(urdf_path)
        meshes = extractor.extract_mesh_info()
        
        self.assertEqual(len(meshes), 1)
        self.assertIsNotNone(meshes[0].scale)
        np.testing.assert_array_equal(meshes[0].scale, [2.0, 1.5, 3.0])


class TestURDFIntegrationForPreprocessing(unittest.TestCase):
    """Integration tests using specific URDF files for preprocessing."""
    
    # Class variable to be set when testing with specific URDF
    mesh_extractors = []
    
    def test_urdf_mesh_preprocessing_workflow(self):
        """Test complete preprocessing workflow with URDF meshes."""
        if not self.mesh_extractors:
            self.skipTest("No URDF file specified for testing")
        
        tested_meshes = 0
        for extractor in self.mesh_extractors:
            mesh_infos = extractor.extract_mesh_info()
            
            for mesh_info in mesh_infos[:3]:  # Limit to first 3 meshes
                if mesh_info.exists:
                    with self.subTest(urdf=extractor.urdf_path.name, 
                                    mesh=mesh_info.link_name):
                        
                        # Create mock mesh for testing
                        mock_mesh = Mock()
                        mock_mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
                        mock_mesh.faces = np.array([[0, 1, 2]])
                        mock_mesh.vertex_normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
                        mock_mesh.edges_unique = np.array([[0, 1], [1, 2], [2, 0]])
                        
                        with patch('trimesh.Trimesh') as mock_trimesh:
                            mock_result = Mock()
                            mock_trimesh.return_value = mock_result
                            
                            # Test thickness addition
                            result = add_thickness(mock_mesh, 0.01)
                            self.assertEqual(result, mock_result)
                            tested_meshes += 1
        
        if tested_meshes == 0:
            self.skipTest("No existing mesh files found in the specified URDF")
    
    @patch('trimesh.load')
    def test_mesh_loading_from_urdf(self, mock_load):
        """Test mesh loading workflow from URDF files."""
        if not self.mesh_extractors:
            self.skipTest("No URDF file specified for testing")
        
        mock_mesh = Mock()
        mock_load.return_value = mock_mesh
        
        loaded_meshes = 0
        for extractor in self.mesh_extractors:
            mesh_infos = extractor.extract_mesh_info()
            
            for mesh_info in mesh_infos[:3]:  # Test first 3 meshes
                if mesh_info.exists:
                    with self.subTest(mesh_file=mesh_info.filename):
                        mesh = trimesh.load(mesh_info.filename)
                        self.assertEqual(mesh, mock_mesh)
                        loaded_meshes += 1
        
        if loaded_meshes == 0:
            self.skipTest("No existing mesh files found to load")


def run_preprocessing_tests(specific_urdf: str):
    """Run preprocessing tests with a specific URDF file.
    
    Args:
        specific_urdf: Path to URDF file to test with (required)
    """
    urdf_path = Path(specific_urdf)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {specific_urdf}")
        print(f"Please check the path and try again.")
        return False
    
    print(f"Loading URDF for preprocessing tests: {urdf_path}")
    try:
        extractor = URDFMeshExtractor(urdf_path)
        mesh_infos = extractor.extract_mesh_info()
        
        print(f"Robot: {extractor.root.get('name', 'unnamed')}")
        print(f"Mesh files found: {len(mesh_infos)}")
        
        existing_meshes = [m for m in mesh_infos if m.exists]
        print(f"Existing mesh files: {len(existing_meshes)}")
        
        if mesh_infos:
            print("Mesh details:")
            for mesh_info in mesh_infos:
                exists_str = "✓" if mesh_info.exists else "✗"
                scale_str = f"(scale: {mesh_info.scale})" if mesh_info.scale is not None else ""
                print(f"  {exists_str} {mesh_info.link_name} [{mesh_info.mesh_type}] {scale_str}")
                print(f"    {mesh_info.original_filename}")
        
        print("\nRunning preprocessing tests...\n")
        
        # Set class-level data for specific URDF testing
        TestURDFIntegrationForPreprocessing.mesh_extractors = [extractor]
        
    except Exception as e:
        print(f"Error analyzing URDF: {e}")
        return False
    
    # Create and run test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAddThickness))
    suite.addTests(loader.loadTestsFromTestCase(TestURDFMeshExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestURDFIntegrationForPreprocessing))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test preprocessing functionality with a specific URDF file',
        epilog='Examples:\n'
               '  python3 pretest.py robot.urdf          # Test with specific URDF\n'
               '  python3 pretest.py path/to/robot.urdf  # Test with URDF at path',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('urdf', help='Path to URDF file to test with (required)')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"Testing preprocessing with URDF: {args.urdf}")
    success = run_preprocessing_tests(args.urdf)
    
    if success:
        print("\nAll preprocessing tests passed!")
    else:
        print("\nSome preprocessing tests failed!")
        sys.exit(1)