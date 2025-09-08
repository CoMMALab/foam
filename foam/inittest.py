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

from foam.utility import *
from foam.external import *
from foam.model import *
import foam.__init__ as your_module

from trimesh.transformations import compose_matrix, euler_matrix, translation_matrix, quaternion_matrix

# Import all the classes and functions directly so we can use them without prefix
from foam.__init__ import (
    smooth_manifold,
    spherize_mesh, 
    ParallelSpherizer,
    SpherizationDatabase,
    SpherizationHelper
)

#TODO THESE ARE STILL VERY BUGGY!!!!!!!!!!!!!!

class TestSmoothManifold(unittest.TestCase):
    
    @patch('your_module.manifold') # IMPORTANT THESE CREATE "FAKE" VERSIONS OF THE FUNCTIONS
    @patch('your_module.simplify_manifold')
    @patch('your_module.smooth_mesh')
    def test_smooth_manifold_default_params(self, mock_smooth, mock_simplify, mock_manifold):
        # Setup
        mock_mesh = Mock(spec=Trimesh)
        mock_manifold.return_value = mock_mesh # this is what the fake function will return
        mock_simplify.return_value = mock_mesh
        
        # Execute
        result = smooth_manifold(mock_mesh)
        
        # Assert
        mock_manifold.assert_called_once_with(mock_mesh, 1000) #asserts calls 
        mock_simplify.assert_called_once_with(mock_mesh, 0.2)
        mock_smooth.assert_called_once_with(mock_mesh)
        self.assertEqual(result, mock_mesh) #asserts that the result is the same as the mock mesh
    
    @patch('your_module.manifold')
    @patch('your_module.simplify_manifold')
    @patch('your_module.smooth_mesh')
    def test_smooth_manifold_custom_params(self, mock_smooth, mock_simplify, mock_manifold):
        # Setup
        mock_mesh = Mock(spec=Trimesh)
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
        self.mock_mesh = Mock(spec=Trimesh)
        self.mock_mesh.copy.return_value = self.mock_mesh
        self.mock_mesh.bounds = np.array([[0, 0, 0], [10, 10, 10]])
        
    @patch('your_module.load_mesh_file')
    @patch('your_module.check_valid_for_spherization')
    @patch('your_module.compute_spheres')
    def test_spherize_mesh_with_path(self, mock_compute, mock_check, mock_load):
        # Setup
        mock_load.return_value = self.mock_mesh
        mock_check.return_value = True
        mock_spheres = [Mock(spec=Spherization)]
        mock_compute.return_value = mock_spheres
        
        mesh_path = Path("test_mesh.obj")
        spherization_kwargs = {'method': 'medial'}
        
        # Execute
        result = spherize_mesh("test", mesh_path, spherization_kwargs=spherization_kwargs)
        
        # Assert
        mock_load.assert_called_once_with(mesh_path)
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)
    
    @patch('your_module.check_valid_for_spherization')
    @patch('your_module.compute_spheres')
    def test_spherize_mesh_with_trimesh_object(self, mock_compute, mock_check):
        # Setup
        mock_check.return_value = True
        mock_spheres = [Mock(spec=Spherization)]
        mock_compute.return_value = mock_spheres
        
        spherization_kwargs = {'method': 'medial'}
        
        # Execute
        result = spherize_mesh("test", self.mock_mesh, spherization_kwargs=spherization_kwargs)
        
        # Assert
        mock_compute.assert_called_once_with(self.mock_mesh, method='medial')
        self.assertEqual(result, mock_spheres)
    
    def test_spherize_mesh_with_tm_sphere(self):
        # Setup
        mock_sphere = Mock(spec=TMSphere)
        mock_sphere.center = (1, 2, 3)
        mock_sphere.primitive.radius = 5
        
        spherization_kwargs = {'depth': 2}
        
        # Execute
        result = spherize_mesh("test", mock_sphere, spherization_kwargs=spherization_kwargs)
        
        # Assert
        self.assertEqual(len(result), 3)  # depth + 1
        self.assertIsInstance(result[0], Spherization)
        self.assertEqual(result[0].spheres[0].x, 1)
        self.assertEqual(result[0].spheres[0].y, 2)
        self.assertEqual(result[0].spheres[0].z, 3)
        self.assertEqual(result[0].spheres[0].r, 5)
    
    @patch('your_module.check_valid_for_spherization')
    @patch('your_module.compute_spheres')
    @patch('your_module.smooth_manifold')
    def test_spherize_mesh_requires_smoothing(self, mock_smooth, mock_compute, mock_check):
        # Setup
        mock_check.side_effect = [False, True]  # First check fails, second passes
        mock_smooth.return_value = self.mock_mesh
        mock_spheres = [Mock(spec=Spherization)]
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
    
    @patch('your_module.check_valid_for_spherization')
    def test_spherize_mesh_fails_validation(self, mock_check):
        # Setup
        mock_check.return_value = False
        spherization_kwargs = {'method': 'medial'}
        
        # Execute & Assert
        with self.assertRaises(RuntimeError):
            spherize_mesh("test", self.mock_mesh, spherization_kwargs=spherization_kwargs)


class TestParallelSpherizer(unittest.TestCase):
    
    def setUp(self):
        self.spherizer = ParallelSpherizer(threads=2)
    
    def tearDown(self):
        self.spherizer.executor.shutdown(wait=True)
    
    @patch('your_module.spherize_mesh')
    def test_spherize_mesh_parallel(self, mock_spherize):
        # Setup
        mock_result = [Mock(spec=Spherization)]
        mock_spherize.return_value = mock_result
        mock_mesh = Mock(spec=Trimesh)
        
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
        mock_result = [Mock(spec=Spherization)]
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
    @patch('your_module.jsload')
    def test_database_creation_existing_file(self, mock_jsload):
        # Setup
        mock_data = {"mesh1": {"1": {"0": Mock(spec=Spherization)}}}
        mock_jsload.return_value = mock_data
        existing_path = Path("existing.json")
        existing_path.touch()
        
        # Execute
        db = SpherizationDatabase(existing_path)
        
        # Assert
        self.assertIsNotNone(db.db)
        existing_path.unlink()
    
    def test_add_spherization(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        spherization = Mock(spec=Spherization)
        
        # Execute
        db.add("mesh1", 1, 0, spherization)
        
        # Assert
        self.assertIn("mesh1", db.db)
        self.assertIn(1, db.db["mesh1"])
        self.assertIn(0, db.db["mesh1"][1])
        self.assertEqual(db.db["mesh1"][1][0], spherization)
    
    def test_add_better_spherization(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        worse_sphere = Mock(spec=Spherization)
        better_sphere = Mock(spec=Spherization)
        better_sphere.__lt__ = Mock(return_value=True)
        
        # Execute
        db.add("mesh1", 1, 0, worse_sphere)
        db.add("mesh1", 1, 0, better_sphere)
        
        # Assert
        self.assertEqual(db.db["mesh1"][1][0], better_sphere)
    
    def test_exists(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        db.add("mesh1", 1, 0, Mock(spec=Spherization))
        
        # Execute & Assert
        self.assertTrue(db.exists("mesh1", 1, 0))
        self.assertFalse(db.exists("mesh2", 1, 0))
        self.assertFalse(db.exists("mesh1", 2, 0))
        self.assertFalse(db.exists("mesh1", 1, 1))
    
    def test_get_spherization(self):
        # Setup
        db = SpherizationDatabase(self.db_path)
        spherization = Mock(spec=Spherization)
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
        mock_mesh = Mock(spec=Trimesh)
        
        # Execute
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        # Assert
        self.assertIn("test_mesh", self.helper.ps.waiting)
    
    def test_spherize_mesh_already_in_db(self):
        # Setup
        mock_mesh = Mock(spec=Trimesh)
        spherization = Mock(spec=Spherization)
        self.helper.db.add("test_mesh", 8, 1, spherization)
        
        # Execute
        self.helper.spherize_mesh("test_mesh", mock_mesh)
        
        # Assert - should not be added to waiting queue
        self.assertNotIn("test_mesh", self.helper.ps.waiting)
    
    def test_get_spherization_from_cache(self):
        # Setup
        spherization = Mock(spec=Spherization)
        self.helper.db.add("test_mesh", 8, 1, spherization)
        
        # Execute
        result = self.helper.get_spherization("test_mesh")
        
        # Assert
        self.assertEqual(result, spherization)
    
    @patch('your_module.ParallelSpherizer.get')
    def test_get_spherization_compute_and_cache(self, mock_get):
        # Setup
        spherizations = [Mock(spec=Spherization), Mock(spec=Spherization)]
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