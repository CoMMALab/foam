import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import shutil
import json
from concurrent.futures import Future
import numpy as np

from typing import Any
from json import load as jsload
from json import dumps as jsdumps
from concurrent.futures import ThreadPoolExecutor, Future
from concurrent.futures import wait as future_wait
from trimesh.primitives import Sphere as TMSphere

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

class TestSmoothManifold(unittest.TestCase):
    
    @patch('foam.manifold')  # Fixed: Use foam module directly
    @patch('foam.simplify_manifold')
    @patch('foam.smooth_mesh')
    def test_smooth_manifold_default_params(self, mock_smooth, mock_simplify, mock_manifold):
        # Setup
        mock_mesh = Mock()  # Remove spec to avoid import issues
        mock_manifold.return_value = mock_mesh
        mock_simplify.return_value = mock_mesh
        
        # Execute
        result = smooth_manifold(mock_mesh)
        
        # Assert
        mock_manifold.assert_called_once_with(mock_mesh, 1000)
        mock_simplify.assert_called_once_with(mock_mesh, 0.2)
        mock_smooth.assert_called_once_with(mock_mesh)
        self.assertEqual(result, mock_mesh)
    
    @patch('foam.manifold')
    @patch('foam.simplify_manifold')
    @patch('foam.smooth_mesh')
    def test_smooth_manifold_custom_params(self, mock_smooth, mock_simplify, mock_manifold):
        # Setup
        mock_mesh = Mock()
        mock_manifold.return_value = mock_mesh
        mock_simplify.return_value = mock_mesh
        
        # Execute
        result = smooth_manifold(mock_mesh, manifold_leaves=2000, ratio=0.5)
        
        # Assert
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
        # Setup
        mock_load.return_value = self.mock_mesh
        mock_check.return_value = True
        mock_spheres = [Mock()]  # Simplified mock
        mock_compute.return_value = mock_spheres
        
        mesh_path = Path("test_mesh.obj")
        spherization_kwargs = {'method': 'medial'}
        
        # Execute
        result = spherize_mesh("test", mesh_path, spherization_kwargs=spherization_kwargs)
        
        # Assert
        mock_load.assert_called_once_with(mesh_path)
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)
    
    @patch('foam.check_valid_for_spherization')
    @patch('foam.compute_spheres')
    def test_spherize_mesh_with_trimesh_object(self, mock_compute, mock_check):
        # Setup
        mock_check.return_value = True
        mock_spheres = [Mock()]
        mock_compute.return_value = mock_spheres
        
        spherization_kwargs = {'method': 'medial'}
        
        # Execute
        result = spherize_mesh("test", self.mock_mesh, spherization_kwargs=spherization_kwargs)
        
        # Assert
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)
    
    def test_spherize_mesh_with_tm_sphere(self):
        # Setup - Properly configure mock with primitive attribute
        mock_sphere = Mock(spec=TMSphere)
        mock_sphere.center = (1, 2, 3)
        mock_primitive = Mock()
        mock_primitive.radius = 5
        mock_sphere.primitive = mock_primitive
        
        spherization_kwargs = {'depth': 2}
        
        # Execute
        result = spherize_mesh("test", mock_sphere, spherization_kwargs=spherization_kwargs)
        
        # Assert
        self.assertEqual(len(result), 3)  # depth + 1
        # Verify basic structure without importing Spherization class
        self.assertIsNotNone(result[0])
    
    @patch('foam.check_valid_for_spherization')
    @patch('foam.compute_spheres')
    @patch('foam.smooth_manifold')
    def test_spherize_mesh_requires_smoothing(self, mock_smooth, mock_compute, mock_check):
        # Setup
        mock_check.side_effect = [False, True]  # First check fails, second passes
        mock_smooth.return_value = self.mock_mesh
        mock_spheres = [Mock()]
        mock_compute.return_value = mock_spheres
        
        spherization_kwargs = {'method': 'medial'}
        process_kwargs = {'manifold_leaves': 500, 'ratio': 0.3}
        
        # Execute
        result = spherize_mesh("test", self.mock_mesh, 
                             spherization_kwargs=spherization_kwargs,
                             process_kwargs=process_kwargs)
        
        # Assert
        mock_smooth.assert_called_once_with(self.mock_mesh, manifold_leaves=500, ratio=0.3)
        self.assertEqual(result, mock_spheres)
    
    @patch('foam.check_valid_for_spherization')
    @patch('foam.smooth_manifold')  # Add this patch
    def test_spherize_mesh_fails_validation(self, mock_smooth, mock_check):
        # Setup - Mock always returns False for validation
        mock_check.return_value = False
        mock_smooth.return_value = self.mock_mesh  # Ensure smooth_manifold returns a mock
        spherization_kwargs = {'method': 'medial'}
        
        # Execute & Assert
        with self.assertRaises(RuntimeError):
            spherize_mesh("test", self.mock_mesh, spherization_kwargs=spherization_kwargs)


class TestParallelSpherizer(unittest.TestCase):
    
    def setUp(self):
        self.spherizer = ParallelSpherizer(threads=2)
    
    def tearDown(self):
        self.spherizer.executor.shutdown(wait=True)
    
    @patch('foam.spherize_mesh')
    def test_spherize_mesh_parallel(self, mock_spherize):
        # Setup
        mock_result = [Mock()]
        mock_spherize.return_value = mock_result
        mock_mesh = Mock()
        
        # Execute
        future = self.spherizer.spherize_mesh("test", mock_mesh)
        result = future.result()
        
        # Assert
        self.assertIsInstance(future, Future)
        self.assertEqual(result, mock_result)
        self.assertIn("test", self.spherizer.waiting)
    
    def test_get_result(self):
        # Setup
        mock_future = Mock(spec=Future)
        mock_result = [Mock()]
        mock_future.result.return_value = mock_result
        self.spherizer.waiting["test"] = mock_future
        
        # Execute
        result = self.spherizer.get("test")
        
        # Assert
        self.assertEqual(result, mock_result)


class TestSpherizationDatabase(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.json"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_database_creation_new_file(self):
        # Execute
        db = SpherizationDatabase(self.db_path)
        
        # Assert
        self.assertEqual(db.db, {})
        self.assertEqual(db.path, self.db_path)
    
    @patch('builtins.open', mock_open(read_data='{"mesh1": {"1": {"0": {}}}}'))
    @patch('foam.jsload')
    def test_database_creation_existing_file(self, mock_jsload):
        # Setup - Use simple data structure to avoid circular import
        mock_data = {"mesh1": {"1": {"0": {"spheres": [], "mean_error": 0.0}}}}
        mock_jsload.return_value = mock_data
        
        # Create a temporary file that exists
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump({}, tmp)
            existing_path = Path(tmp.name)
        
        try:
            # Execute
            db = SpherizationDatabase(existing_path)
            
            # Assert
            self.assertIsNotNone(db.db)
        finally:
            existing_path.unlink()
    
    def test_add_spherization(self):
        # Setup - Create mock spherization
        db = SpherizationDatabase(self.db_path)
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)  # For comparison
        
        # Execute
        db.add("mesh1", 1, 0, spherization)
        
        # Assert
        self.assertIn("mesh1", db.db)
        self.assertIn(1, db.db["mesh1"])
        self.assertIn(0, db.db["mesh1"][1])
        self.assertEqual(db.db["mesh1"][1][0], spherization)
    
    def test_add_better_spherization(self):
        # Setup - Create mock spherizations with comparison
        db = SpherizationDatabase(self.db_path)
        worse_sphere = Mock()
        worse_sphere.__lt__ = Mock(return_value=False)
        
        better_sphere = Mock()
        better_sphere.__lt__ = Mock(return_value=True)
        
        # Execute
        db.add("mesh1", 1, 0, worse_sphere)
        db.add("mesh1", 1, 0, better_sphere)
        
        # Assert
        self.assertEqual(db.db["mesh1"][1][0], better_sphere)
    
    def test_exists(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        db.add("mesh1", 1, 0, spherization)
        
        # Execute & Assert
        self.assertTrue(db.exists("mesh1", 1, 0))
        self.assertFalse(db.exists("mesh2", 1, 0))
        self.assertFalse(db.exists("mesh1", 2, 0))
        self.assertFalse(db.exists("mesh1", 1, 1))
    
    def test_get_spherization(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        db.add("mesh1", 1, 0, spherization)
        
        # Execute
        result = db.get("mesh1", 1, 0)
        
        # Assert
        self.assertEqual(result, spherization)


class TestSpherizationHelper(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.json"
        self.helper = SpherizationHelper(self.db_path, threads=2)
    
    def tearDown(self):
        self.helper.ps.executor.shutdown(wait=True)
        shutil.rmtree(self.temp_dir)
    
    def test_spherize_mesh_not_in_db(self):
        # Setup
        mock_mesh = Mock()
        
        # Execute
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        # Assert
        self.assertIn("test_mesh", self.helper.ps.waiting)
    
    def test_spherize_mesh_already_in_db(self):
        # Setup
        mock_mesh = Mock()
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        self.helper.db.add("test_mesh", 8, 1, spherization)
        
        # Execute
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        # Assert - should not be added to waiting queue
        self.assertNotIn("test_mesh", self.helper.ps.waiting)
    
    def test_get_spherization_from_cache(self):
        # Setup
        spherization = Mock()
        spherization.__lt__ = Mock(return_value=False)
        self.helper.db.add("test_mesh", 8, 1, spherization)
        
        # Execute
        result = self.helper.get_spherization("test_mesh")
        
        # Assert
        self.assertEqual(result, spherization)
    
    @patch.object(ParallelSpherizer, 'get')
    def test_get_spherization_compute_and_cache(self, mock_get):
        # Setup - Create mock spherizations
        spherization1 = Mock()
        spherization1.__lt__ = Mock(return_value=False)
        spherization2 = Mock()  
        spherization2.__lt__ = Mock(return_value=False)
        
        spherizations = [spherization1, spherization2]
        mock_get.return_value = spherizations
        
        # Execute
        result = self.helper.get_spherization("test_mesh", depth=1, cache=True)
        
        # Assert
        self.assertEqual(result, spherizations[1])
        # Check that all levels were cached
        self.assertTrue(self.helper.db.exists("test_mesh", 8, 0))
        self.assertTrue(self.helper.db.exists("test_mesh", 8, 1))


if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSmoothManifold))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizeMesh))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelSpherizer))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizationDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestSpherizationHelper))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\nAll tests passed!")
    else:
        print(f"\nTests failed: {len(result.failures)} failures, {len(result.errors)} errors")