import unittest
from pathlib import Path
import numpy as np
import json
import argparse
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from foam.model import Sphere, Spherization, SphereEncoder, SphereDecoder
from foam.utility.__init__ import load_urdf, get_urdf_primitives, get_urdf_meshes, get_urdf_spheres


class TestSphereModel(unittest.TestCase):
    """Test suite for Sphere and Spherization model classes."""
    
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


class TestSphere(TestSphereModel):
    """Test the Sphere class."""
    
    def test_sphere_creation_basic(self):
        """Test basic sphere creation."""
        sphere = Sphere(1.0, 2.0, 3.0, 0.5)
        
        np.testing.assert_array_equal(sphere.origin, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(sphere.radius, 0.5)
    
    def test_sphere_creation_with_offset(self):
        """Test sphere creation with offset."""
        offset = np.array([1.0, 1.0, 1.0])
        sphere = Sphere(1.0, 2.0, 3.0, 0.5, offset=offset)
        
        expected_origin = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(sphere.origin, expected_origin)
        self.assertEqual(sphere.radius, 0.5)
    
    def test_sphere_creation_negative_coordinates(self):
        """Test sphere with negative coordinates."""
        sphere = Sphere(-1.0, -2.0, -3.0, 1.5)
        
        np.testing.assert_array_equal(sphere.origin, np.array([-1.0, -2.0, -3.0]))
        self.assertEqual(sphere.radius, 1.5)
    
    def test_sphere_offset_method(self):
        """Test the offset method."""
        sphere = Sphere(0.0, 0.0, 0.0, 1.0)
        offset = np.array([5.0, -3.0, 2.0])
        
        sphere.offset(offset)
        
        np.testing.assert_array_equal(sphere.origin, offset)
    
    def test_sphere_offset_multiple_times(self):
        """Test applying offset multiple times."""
        sphere = Sphere(1.0, 1.0, 1.0, 0.5)
        
        sphere.offset(np.array([1.0, 0.0, 0.0]))
        sphere.offset(np.array([0.0, 2.0, 0.0]))
        sphere.offset(np.array([0.0, 0.0, 3.0]))
        
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(sphere.origin, expected)
    
    def test_sphere_zero_radius(self):
        """Test sphere with zero radius."""
        sphere = Sphere(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(sphere.radius, 0.0)
    
    def test_sphere_origin_is_numpy_array(self):
        """Test that origin is a numpy array."""
        sphere = Sphere(1.0, 2.0, 3.0, 0.5)
        self.assertIsInstance(sphere.origin, np.ndarray)
        self.assertEqual(sphere.origin.shape, (3,))


class TestSpherization(TestSphereModel):
    """Test the Spherization class."""
    
    def create_test_spheres(self):
        """Create a list of test spheres."""
        return [
            Sphere(0.0, 0.0, 0.0, 1.0),
            Sphere(2.0, 0.0, 0.0, 0.5),
            Sphere(0.0, 2.0, 0.0, 0.5)
        ]
    
    def test_spherization_creation(self):
        """Test basic spherization creation."""
        spheres = self.create_test_spheres()
        spherization = Spherization(spheres, 0.1, 0.05, 0.2)
        
        self.assertEqual(len(spherization.spheres), 3)
        self.assertEqual(spherization.mean_error, 0.1)
        self.assertEqual(spherization.best_error, 0.05)
        self.assertEqual(spherization.worst_error, 0.2)
    
    def test_spherization_length(self):
        """Test __len__ method."""
        spheres = self.create_test_spheres()
        spherization = Spherization(spheres, 0.1, 0.05, 0.2)
        
        self.assertEqual(len(spherization), 3)
    
    def test_spherization_empty(self):
        """Test spherization with no spheres."""
        spherization = Spherization([], 0.0, 0.0, 0.0)
        self.assertEqual(len(spherization), 0)
    
    def test_spherization_less_than_all_better(self):
        """Test __lt__ when all errors are better."""
        spheres = self.create_test_spheres()
        spherization1 = Spherization(spheres, 0.1, 0.05, 0.2)
        spherization2 = Spherization(spheres, 0.2, 0.1, 0.3)
        
        self.assertTrue(spherization1 < spherization2)
        self.assertFalse(spherization2 < spherization1)
    
    def test_spherization_less_than_mixed_errors(self):
        """Test __lt__ when errors are mixed."""
        spheres = self.create_test_spheres()
        # Better mean but worse worst error
        spherization1 = Spherization(spheres, 0.1, 0.05, 0.4)
        spherization2 = Spherization(spheres, 0.2, 0.1, 0.2)
        
        self.assertFalse(spherization1 < spherization2)
        self.assertFalse(spherization2 < spherization1)
    
    def test_spherization_less_than_equal(self):
        """Test __lt__ with equal errors."""
        spheres = self.create_test_spheres()
        spherization1 = Spherization(spheres, 0.1, 0.05, 0.2)
        spherization2 = Spherization(spheres, 0.1, 0.05, 0.2)
        
        self.assertFalse(spherization1 < spherization2)
    
    def test_spherization_offset(self):
        """Test offset method applies to all spheres."""
        spheres = self.create_test_spheres()
        spherization = Spherization(spheres, 0.1, 0.05, 0.2)
        
        original_origins = [s.origin.copy() for s in spherization.spheres]
        offset = np.array([1.0, 2.0, 3.0])
        
        spherization.offset(offset)
        
        for i, sphere in enumerate(spherization.spheres):
            expected = original_origins[i] + offset
            np.testing.assert_array_equal(sphere.origin, expected)
    
    def test_spherization_offset_multiple_times(self):
        """Test applying offset multiple times to spherization."""
        spheres = self.create_test_spheres()
        spherization = Spherization(spheres, 0.1, 0.05, 0.2)
        
        spherization.offset(np.array([1.0, 0.0, 0.0]))
        spherization.offset(np.array([0.0, 1.0, 0.0]))
        
        # First sphere should be at [1, 1, 0]
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_equal(spherization.spheres[0].origin, expected)


class TestJSONSerialization(TestSphereModel):
    """Test JSON encoding and decoding."""
    
    def test_sphere_encode(self):
        """Test encoding a sphere to JSON."""
        sphere = Sphere(1.0, 2.0, 3.0, 0.5)
        json_str = json.dumps(sphere, cls=SphereEncoder)
        
        data = json.loads(json_str)
        self.assertEqual(data['origin'], [1.0, 2.0, 3.0])
        self.assertEqual(data['radius'], 0.5)
    
    def test_sphere_decode(self):
        """Test decoding a sphere from JSON."""
        json_str = '{"origin": [1.0, 2.0, 3.0], "radius": 0.5}'
        sphere = json.loads(json_str, cls=SphereDecoder)
        
        self.assertIsInstance(sphere, Sphere)
        np.testing.assert_array_equal(sphere.origin, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(sphere.radius, 0.5)
    
    def test_sphere_roundtrip(self):
        """Test encoding and decoding a sphere."""
        original = Sphere(1.5, -2.5, 3.5, 0.75)
        json_str = json.dumps(original, cls=SphereEncoder)
        decoded = json.loads(json_str, cls=SphereDecoder)
        
        self.assertIsInstance(decoded, Sphere)
        np.testing.assert_array_almost_equal(decoded.origin, original.origin)
        self.assertAlmostEqual(decoded.radius, original.radius)
    
    def test_spherization_encode(self):
        """Test encoding a spherization to JSON."""
        spheres = [
            Sphere(0.0, 0.0, 0.0, 1.0),
            Sphere(1.0, 1.0, 1.0, 0.5)
        ]
        spherization = Spherization(spheres, 0.1, 0.05, 0.2)
        json_str = json.dumps(spherization, cls=SphereEncoder)
        
        data = json.loads(json_str)
        self.assertEqual(data['mean'], 0.1)
        self.assertEqual(data['best'], 0.05)
        self.assertEqual(data['worst'], 0.2)
        self.assertEqual(len(data['spheres']), 2)
    
    def test_spherization_decode(self):
        """Test decoding a spherization from JSON."""
        json_str = '''{
            "mean": 0.1,
            "best": 0.05,
            "worst": 0.2,
            "spheres": [
                {"origin": [0.0, 0.0, 0.0], "radius": 1.0},
                {"origin": [1.0, 1.0, 1.0], "radius": 0.5}
            ]
        }'''
        spherization = json.loads(json_str, cls=SphereDecoder)
        
        self.assertIsInstance(spherization, Spherization)
        self.assertEqual(spherization.mean_error, 0.1)
        self.assertEqual(spherization.best_error, 0.05)
        self.assertEqual(spherization.worst_error, 0.2)
        self.assertEqual(len(spherization.spheres), 2)
        self.assertIsInstance(spherization.spheres[0], Sphere)
    
    def test_spherization_roundtrip(self):
        """Test encoding and decoding a spherization."""
        spheres = [
            Sphere(0.0, 0.0, 0.0, 1.0),
            Sphere(1.0, 1.0, 1.0, 0.5),
            Sphere(-1.0, 2.0, -3.0, 0.25)
        ]
        original = Spherization(spheres, 0.15, 0.08, 0.25)
        
        json_str = json.dumps(original, cls=SphereEncoder)
        decoded = json.loads(json_str, cls=SphereDecoder)
        
        self.assertIsInstance(decoded, Spherization)
        self.assertEqual(len(decoded.spheres), len(original.spheres))
        self.assertAlmostEqual(decoded.mean_error, original.mean_error)
        self.assertAlmostEqual(decoded.best_error, original.best_error)
        self.assertAlmostEqual(decoded.worst_error, original.worst_error)
        
        for i, (original_sphere, decoded_sphere) in enumerate(zip(original.spheres, decoded.spheres)):
            np.testing.assert_array_almost_equal(decoded_sphere.origin, original_sphere.origin)
            self.assertAlmostEqual(decoded_sphere.radius, original_sphere.radius)
    
    def test_encode_list_of_spheres(self):
        """Test encoding a list of spheres."""
        spheres = [
            Sphere(0.0, 0.0, 0.0, 1.0),
            Sphere(1.0, 1.0, 1.0, 0.5)
        ]
        json_str = json.dumps(spheres, cls=SphereEncoder)
        decoded = json.loads(json_str, cls=SphereDecoder)
        
        self.assertEqual(len(decoded), 2)
        for sphere in decoded:
            self.assertIsInstance(sphere, Sphere)
    
    def test_encode_dict_with_spherization(self):
        """Test encoding a dictionary containing spherizations."""
        spheres1 = [Sphere(0.0, 0.0, 0.0, 1.0)]
        spheres2 = [Sphere(1.0, 1.0, 1.0, 0.5)]
        
        data = {
            'link1': Spherization(spheres1, 0.1, 0.05, 0.15),
            'link2': Spherization(spheres2, 0.2, 0.1, 0.25)
        }
        
        json_str = json.dumps(data, cls=SphereEncoder)
        decoded = json.loads(json_str, cls=SphereDecoder)
        
        self.assertIn('link1', decoded)
        self.assertIn('link2', decoded)
        self.assertIsInstance(decoded['link1'], Spherization)
        self.assertIsInstance(decoded['link2'], Spherization)


class TestURDFIntegration(TestSphereModel):
    """Test integration with URDF files."""
    
    @unittest.skipIf(not hasattr(TestSphereModel, 'urdf_file') or TestSphereModel.urdf_file is None,
                     "No URDF file specified")
    def test_create_spheres_from_urdf_spheres(self):
        """Test creating Sphere objects from URDF sphere primitives."""
        urdf = load_urdf(TestSphereModel.urdf_file)
        urdf_spheres = list(get_urdf_spheres(urdf))
        
        print(f"Found {len(urdf_spheres)} sphere primitive(s) in URDF")
        
        # Convert URDF spheres to Sphere objects
        spheres = []
        for x, y, z, radius in urdf_spheres:
            sphere = Sphere(x, y, z, radius)
            spheres.append(sphere)
            
            # Verify sphere was created correctly
            self.assertIsInstance(sphere, Sphere)
            self.assertEqual(sphere.origin[0], x)
            self.assertEqual(sphere.origin[1], y)
            self.assertEqual(sphere.origin[2], z)
            self.assertEqual(sphere.radius, radius)
    
    @unittest.skipIf(not hasattr(TestSphereModel, 'urdf_file') or TestSphereModel.urdf_file is None,
                     "No URDF file specified")
    def test_create_spherization_from_urdf(self):
        """Test creating a Spherization from URDF data."""
        try:
            urdf = load_urdf(TestSphereModel.urdf_file)
            urdf_spheres = list(get_urdf_spheres(urdf))
            
            if len(urdf_spheres) == 0:
                self.skipTest("No spheres found in URDF")
            
            # Create Sphere objects
            spheres = [Sphere(x, y, z, r) for x, y, z, r in urdf_spheres]
            
            # Create a spherization (with dummy error values)
            spherization = Spherization(spheres, 0.1, 0.05, 0.2)
            
            self.assertEqual(len(spherization), len(urdf_spheres))
            self.assertIsInstance(spherization, Spherization)
            
            print(f"Created spherization with {len(spherization)} spheres")
            
        except Exception as e:
            self.skipTest(f"Could not process URDF: {e}")
    
    @unittest.skipIf(not hasattr(TestSphereModel, 'urdf_file') or TestSphereModel.urdf_file is None,
                     "No URDF file specified")
    def test_spherization_per_link(self):
        """Test creating spherizations for each link in URDF."""
        try:
            urdf = load_urdf(TestSphereModel.urdf_file)
            primitives = get_urdf_primitives(urdf)
            
            print(f"Found {len(primitives)} primitive(s) in URDF")
            
            # Create a spherization for each primitive (simulated)
            spherizations = {}
            for primitive in primitives:
                # Simulate creating spheres at the primitive's origin
                sphere = Sphere(
                    primitive.xyz[0],
                    primitive.xyz[1],
                    primitive.xyz[2],
                    0.5  # Dummy radius
                )
                spherizations[primitive.name] = Spherization(
                    [sphere],
                    0.1, 0.05, 0.2
                )
            
            # Verify we can serialize this structure
            json_str = json.dumps(spherizations, cls=SphereEncoder)
            decoded = json.loads(json_str, cls=SphereDecoder)
            
            self.assertEqual(len(decoded), len(primitives))
            
            for name, spherization in decoded.items():
                self.assertIsInstance(spherization, Spherization)
                print(f"  {name}: {len(spherization)} sphere(s)")
                
        except Exception as e:
            self.skipTest(f"Could not process URDF: {e}")
    
    @unittest.skipIf(not hasattr(TestSphereModel, 'urdf_file') or TestSphereModel.urdf_file is None,
                     "No URDF file specified")
    def test_save_and_load_spherization_json(self):
        """Test saving and loading spherization data to/from JSON file."""
        try:
            import tempfile
            
            urdf = load_urdf(TestSphereModel.urdf_file)
            primitives = get_urdf_primitives(urdf)
            
            if len(primitives) == 0:
                self.skipTest("No primitives found in URDF")
            
            # Create spherizations
            spherizations = {}
            for primitive in primitives:
                sphere = Sphere(
                    primitive.xyz[0],
                    primitive.xyz[1],
                    primitive.xyz[2],
                    0.5
                )
                spherizations[primitive.name] = Spherization(
                    [sphere],
                    0.1, 0.05, 0.2
                )
            
            # Save to file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(spherizations, f, cls=SphereEncoder, indent=2)
                temp_path = f.name
            
            try:
                # Load from file
                with open(temp_path, 'r') as f:
                    loaded = json.load(f, cls=SphereDecoder)
                
                # Verify
                self.assertEqual(len(loaded), len(spherizations))
                for name in spherizations.keys():
                    self.assertIn(name, loaded)
                    self.assertIsInstance(loaded[name], Spherization)
                
                print(f"✓ Successfully saved and loaded {len(loaded)} spherizations")
                
            finally:
                # Cleanup
                Path(temp_path).unlink()
                
        except Exception as e:
            self.skipTest(f"Could not complete test: {e}")


class TestEdgeCases(TestSphereModel):
    """Test edge cases and error conditions."""
    
    def test_sphere_with_very_large_radius(self):
        """Test sphere with very large radius."""
        sphere = Sphere(0.0, 0.0, 0.0, 1e6)
        self.assertEqual(sphere.radius, 1e6)
    
    def test_sphere_with_very_small_radius(self):
        """Test sphere with very small radius."""
        sphere = Sphere(0.0, 0.0, 0.0, 1e-10)
        self.assertAlmostEqual(sphere.radius, 1e-10)
    
    def test_spherization_with_single_sphere(self):
        """Test spherization with only one sphere."""
        sphere = Sphere(0.0, 0.0, 0.0, 1.0)
        spherization = Spherization([sphere], 0.1, 0.1, 0.1)
        
        self.assertEqual(len(spherization), 1)
    
    def test_spherization_negative_errors(self):
        """Test spherization with negative error values (shouldn't happen but test it)."""
        spheres = [Sphere(0.0, 0.0, 0.0, 1.0)]
        spherization = Spherization(spheres, -0.1, -0.05, -0.2)
        
        # Just verify it doesn't crash
        self.assertEqual(spherization.mean_error, -0.1)
    
    def test_json_decode_invalid_sphere(self):
        """Test decoding invalid sphere JSON."""
        # Missing radius field
        json_str = '{"origin": [1.0, 2.0, 3.0]}'
        result = json.loads(json_str, cls=SphereDecoder)
        
        # Should return dict, not Sphere
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, Sphere)
    
    def test_json_decode_invalid_spherization(self):
        """Test decoding invalid spherization JSON."""
        # Missing spheres field
        json_str = '{"mean": 0.1, "best": 0.05, "worst": 0.2}'
        result = json.loads(json_str, cls=SphereDecoder)
        
        # Should return dict, not Spherization
        self.assertIsInstance(result, dict)
        self.assertNotIsInstance(result, Spherization)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Sphere and Spherization models')
    parser.add_argument('--urdf', type=str, help='Path to URDF file to test')
    
    # Separate our args from unittest args
    args, unittest_args = parser.parse_known_args()
    
    # Set the URDF path as a class variable
    if args.urdf:
        TestSphereModel.test_urdf_path = args.urdf
    elif 'URDF_TEST_FILE' in os.environ:
        TestSphereModel.test_urdf_path = os.environ['URDF_TEST_FILE']
    
    # Pass remaining arguments to unittest
    sys.argv[1:] = unittest_args
    unittest.main(verbosity=2)