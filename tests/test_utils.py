"""
Unit tests for tools.utils module.

Tests for:
- moded_max: soft/hard/ste max with different parameters
- moded_min: soft/hard/ste min with different parameters  
- _to_serializable: conversion of various types to JSON-serializable format
- hash_dict: deterministic hashing of dictionaries
"""

import unittest
import torch
import numpy as np
from tools.utils import moded_max, moded_min, _to_serializable, hash_dict


class TestModedMax(unittest.TestCase):
    """Test suite for moded_max function."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.scores = torch.tensor([[1.0, 5.0, 3.0], 
                                    [2.0, 1.0, 4.0]])  # (2, 3)
        self.candidates = torch.tensor([[[0.0, 0.0], 
                                         [1.0, 1.0], 
                                         [2.0, 2.0]]])  # (1, 3, 2)
    
    def test_soft_mode_basic(self):
        """Test soft mode returns weighted combination."""
        choice, v, aux = moded_max(self.scores, self.candidates, 
                                   dim=1, temp=1.0, mode="soft")
        
        # Check shapes
        self.assertEqual(choice.shape, (2, 2))
        self.assertEqual(v.shape, (2,))
        
        # Check weights sum to 1
        self.assertTrue(torch.allclose(aux["weights"].sum(dim=1), 
                                      torch.ones(2), atol=1e-5))
        
        # Weights should be positive
        self.assertTrue((aux["weights"] >= 0).all())
        
        # mean_max should be between 0 and 1
        self.assertGreater(aux["mean_max"], 0.0)
        self.assertLessEqual(aux["mean_max"], 1.0)
        
        # eff_temp should exist
        self.assertIsNotNone(aux["eff_temp"])
    
    def test_soft_mode_high_temp_convergence(self):
        """Test that soft mode with high temp approaches uniform weights."""
        _, _, aux = moded_max(self.scores, self.candidates, 
                             dim=1, temp=0.01, mode="soft")
        
        # With very low temp (high effective temp), weights should be more uniform
        weights = aux["weights"]
        # Standard deviation should be smaller than with high temp
        std_low_temp = weights.std(dim=1).mean()
        
        _, _, aux_high = moded_max(self.scores, self.candidates, 
                                   dim=1, temp=100.0, mode="soft")
        std_high_temp = aux_high["weights"].std(dim=1).mean()
        
        self.assertLess(std_low_temp, std_high_temp)
    
    def test_hard_mode_selects_argmax(self):
        """Test hard mode selects the maximum scoring candidate."""
        choice, v, aux = moded_max(self.scores, self.candidates, 
                                   dim=1, mode="hard")
        
        # Check shapes
        self.assertEqual(choice.shape, (2, 2))
        self.assertEqual(v.shape, (2,))
        
        # Check idx is returned
        self.assertIsNotNone(aux["idx"])
        self.assertEqual(aux["idx"].shape, (2,))
        
        # First sample: max score is at index 1 (score=5.0)
        self.assertEqual(aux["idx"][0].item(), 1)
        torch.testing.assert_close(choice[0], self.candidates[0, 1])
        
        # Second sample: max score is at index 2 (score=4.0)
        self.assertEqual(aux["idx"][1].item(), 2)
        torch.testing.assert_close(choice[1], self.candidates[0, 2])
        
        # Values should match max scores
        self.assertAlmostEqual(v[0].item(), 5.0, places=5)
        self.assertAlmostEqual(v[1].item(), 4.0, places=5)
        
        # mean_max should be 1.0 for hard mode
        self.assertAlmostEqual(aux["mean_max"], 1.0, places=5)
    
    def test_ste_mode_combines_hard_and_soft(self):
        """Test STE mode uses hard forward and soft backward."""
        choice, v, aux = moded_max(self.scores, self.candidates, 
                                   dim=1, temp=1.0, mode="ste")
        
        # Should have both weights (from soft) and idx (from hard)
        self.assertIsNotNone(aux["weights"])
        self.assertIsNotNone(aux["idx"])
        
        # Hard part: idx should point to argmax
        self.assertEqual(aux["idx"][0].item(), 1)
        self.assertEqual(aux["idx"][1].item(), 2)
        
        # Soft part: weights should sum to 1
        self.assertTrue(torch.allclose(aux["weights"].sum(dim=1), 
                                      torch.ones(2), atol=1e-5))
    
    def test_effective_temperature_clamping(self):
        """Test that effective temperature is properly clamped."""
        # Small range of scores should lead to high effective temp
        scores_narrow = torch.tensor([[1.0, 1.001, 1.002]])
        candidates_narrow = torch.randn(1, 3, 2)
        
        _, _, aux = moded_max(scores_narrow, candidates_narrow, 
                             dim=1, temp=5.0, max_effective_temp=100.0, 
                             mode="soft")
        
        # eff_temp should be clamped at max_effective_temp
        self.assertTrue((aux["eff_temp"] <= 100.0).all())
        self.assertTrue((aux["eff_temp"] >= 5.0).all())
    
    def test_gradient_flow_soft_mode(self):
        """Test that gradients flow through soft mode."""
        scores = torch.tensor([[1.0, 5.0, 3.0]], requires_grad=True)
        candidates = torch.tensor([[[0.0], [1.0], [2.0]]])
        
        choice, v, _ = moded_max(scores, candidates, dim=1, mode="soft")
        loss = choice.sum() + v.sum()
        loss.backward()
        
        # Gradients should exist and be non-zero
        self.assertIsNotNone(scores.grad)
        self.assertTrue((scores.grad != 0).any())
    
    def test_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError."""
        with self.assertRaises(ValueError):
            moded_max(self.scores, self.candidates, mode="invalid")
    
    def test_dimension_assertion(self):
        """Test that incorrect dimensions raise assertions."""
        # Wrong candidates shape
        bad_candidates = torch.randn(2, 3, 2)  # Should be (1, 3, 2)
        with self.assertRaises(AssertionError):
            moded_max(self.scores, bad_candidates, dim=1)
        
        # Wrong dim parameter
        with self.assertRaises(AssertionError):
            moded_max(self.scores, self.candidates, dim=0)


class TestModedMin(unittest.TestCase):
    """Test suite for moded_min function."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.scores = torch.tensor([[1.0, 5.0, 3.0], 
                                    [2.0, 1.0, 4.0]])  # (2, 3)
        self.candidates = torch.tensor([[[0.0, 0.0], 
                                         [1.0, 1.0], 
                                         [2.0, 2.0]]])  # (1, 3, 2)
    
    def test_soft_mode_basic(self):
        """Test soft mode returns weighted combination."""
        choice, v, aux = moded_min(self.scores, self.candidates, 
                                   dim=1, temp=1.0, mode="soft")
        
        # Check shapes
        self.assertEqual(choice.shape, (2, 2))
        self.assertEqual(v.shape, (2,))
        
        # Check weights sum to 1
        self.assertTrue(torch.allclose(aux["weights"].sum(dim=1), 
                                      torch.ones(2), atol=1e-5))
        
        # Weights should be positive
        self.assertTrue((aux["weights"] >= 0).all())
    
    def test_hard_mode_selects_argmin(self):
        """Test hard mode selects the minimum scoring candidate."""
        choice, v, aux = moded_min(self.scores, self.candidates, 
                                   dim=1, mode="hard")
        
        # Check shapes
        self.assertEqual(choice.shape, (2, 2))
        self.assertEqual(v.shape, (2,))
        
        # First sample: min score is at index 0 (score=1.0)
        self.assertEqual(aux["idx"][0].item(), 0)
        torch.testing.assert_close(choice[0], self.candidates[0, 0])
        
        # Second sample: min score is at index 1 (score=1.0)
        self.assertEqual(aux["idx"][1].item(), 1)
        torch.testing.assert_close(choice[1], self.candidates[0, 1])
        
        # Values should match min scores
        self.assertAlmostEqual(v[0].item(), 1.0, places=5)
        self.assertAlmostEqual(v[1].item(), 1.0, places=5)
    
    def test_min_is_negative_max(self):
        """Test that min(-scores) = -max(scores)."""
        choice_min, v_min, _ = moded_min(self.scores, self.candidates, 
                                         dim=1, mode="hard")
        choice_max, v_max, _ = moded_max(-self.scores, self.candidates, 
                                         dim=1, mode="hard")
        
        # Choices should be the same
        torch.testing.assert_close(choice_min, choice_max)
        
        # Values should be negatives
        torch.testing.assert_close(v_min, -v_max)
    
    def test_gradient_flow_soft_mode(self):
        """Test that gradients flow through soft mode."""
        scores = torch.tensor([[1.0, 5.0, 3.0]], requires_grad=True)
        candidates = torch.tensor([[[0.0], [1.0], [2.0]]])
        
        choice, v, _ = moded_min(scores, candidates, dim=1, mode="soft")
        loss = choice.sum() + v.sum()
        loss.backward()
        
        # Gradients should exist and be non-zero
        self.assertIsNotNone(scores.grad)
        self.assertTrue((scores.grad != 0).any())


class TestToSerializable(unittest.TestCase):
    """Test suite for _to_serializable function."""
    
    def test_scalar_torch_tensor(self):
        """Test conversion of scalar torch tensor."""
        t = torch.tensor(3.14)
        result = _to_serializable(t)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 3.14, places=5)
    
    def test_vector_torch_tensor(self):
        """Test conversion of vector torch tensor."""
        t = torch.tensor([1.0, 2.0, 3.0])
        result = _to_serializable(t)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0, 3.0])
    
    def test_matrix_torch_tensor(self):
        """Test conversion of 2D torch tensor."""
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = _to_serializable(t)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
    
    def test_numpy_scalar(self):
        """Test conversion of numpy scalar."""
        n = np.float32(2.71)
        result = _to_serializable(n)
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 2.71, places=5)
    
    def test_numpy_array(self):
        """Test conversion of numpy array."""
        n = np.array([1, 2, 3])
        result = _to_serializable(n)
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1, 2, 3])
    
    def test_nested_dict(self):
        """Test conversion of nested dictionary with mixed types."""
        d = {
            "a": torch.tensor(1.0),
            "b": np.array([2, 3]),
            "c": {"nested": torch.tensor([4.0, 5.0])},
        }
        result = _to_serializable(d)
        
        self.assertEqual(result["a"], 1.0)
        self.assertEqual(result["b"], [2, 3])
        self.assertIsInstance(result["c"], dict)
        self.assertEqual(result["c"]["nested"], [4.0, 5.0])
    
    def test_list_of_tensors(self):
        """Test conversion of list containing tensors."""
        lst = [torch.tensor(1.0), torch.tensor([2.0, 3.0])]
        result = _to_serializable(lst)
        
        self.assertIsInstance(result, list)
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], [2.0, 3.0])
    
    def test_tuple_of_tensors(self):
        """Test conversion of tuple containing tensors."""
        tup = (torch.tensor(1.0), np.array([2, 3]))
        result = _to_serializable(tup)
        
        self.assertIsInstance(result, list)  # Tuples become lists
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[1], [2, 3])
    
    def test_python_primitives(self):
        """Test that Python primitives pass through unchanged."""
        self.assertEqual(_to_serializable(42), 42)
        self.assertEqual(_to_serializable(3.14), 3.14)
        self.assertEqual(_to_serializable("hello"), "hello")
        self.assertEqual(_to_serializable(True), True)
        self.assertEqual(_to_serializable(None), None)
    
    def test_torch_device(self):
        """Test conversion of torch device."""
        device = torch.device("cpu")
        result = _to_serializable(device)
        self.assertEqual(result, "cpu")
        
        if torch.cuda.is_available():
            device_cuda = torch.device("cuda:0")
            result_cuda = _to_serializable(device_cuda)
            self.assertEqual(result_cuda, "cuda:0")
    
    def test_callable(self):
        """Test conversion of callable objects."""
        def my_func():
            pass
        
        result = _to_serializable(my_func)
        self.assertIsInstance(result, str)
        # Should contain module and class info
        self.assertIn("function", result)
    
    def test_optimizer(self):
        """Test conversion of optimizer."""
        param = torch.nn.Parameter(torch.randn(2, 2))
        optimizer = torch.optim.Adam([param])
        
        result = _to_serializable(optimizer)
        self.assertIsInstance(result, str)
        self.assertIn("OPTIMIZER:Adam", result)
    
    def test_gradient_preserved(self):
        """Test that tensors with gradients are properly converted."""
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        result = _to_serializable(t)
        
        # Should convert to list regardless of grad
        self.assertIsInstance(result, list)
        self.assertEqual(result, [1.0, 2.0])


class TestHashDict(unittest.TestCase):
    """Test suite for hash_dict function."""
    
    def test_deterministic_hashing(self):
        """Test that same dict always produces same hash."""
        d = {"a": 1, "b": 2, "c": 3}
        hash1 = hash_dict(d)
        hash2 = hash_dict(d)
        
        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, str)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex digest length
    
    def test_order_independence(self):
        """Test that dict order doesn't affect hash (sorted keys)."""
        d1 = {"a": 1, "b": 2, "c": 3}
        d2 = {"c": 3, "a": 1, "b": 2}
        
        self.assertEqual(hash_dict(d1), hash_dict(d2))
    
    def test_different_dicts_different_hashes(self):
        """Test that different dicts produce different hashes."""
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 1, "b": 3}
        
        self.assertNotEqual(hash_dict(d1), hash_dict(d2))
    
    def test_torch_tensor_hashing(self):
        """Test hashing of dicts containing torch tensors."""
        d1 = {"tensor": torch.tensor([1.0, 2.0, 3.0])}
        d2 = {"tensor": torch.tensor([1.0, 2.0, 3.0])}
        d3 = {"tensor": torch.tensor([1.0, 2.0, 4.0])}
        
        # Same tensors should produce same hash
        self.assertEqual(hash_dict(d1), hash_dict(d2))
        
        # Different tensors should produce different hash
        self.assertNotEqual(hash_dict(d1), hash_dict(d3))
    
    def test_numpy_array_hashing(self):
        """Test hashing of dicts containing numpy arrays."""
        d1 = {"array": np.array([1, 2, 3])}
        d2 = {"array": np.array([1, 2, 3])}
        
        self.assertEqual(hash_dict(d1), hash_dict(d2))
    
    def test_nested_dict_hashing(self):
        """Test hashing of nested dictionaries."""
        d1 = {
            "a": 1,
            "b": {"c": 2, "d": torch.tensor(3.0)},
        }
        d2 = {
            "a": 1,
            "b": {"c": 2, "d": torch.tensor(3.0)},
        }
        
        self.assertEqual(hash_dict(d1), hash_dict(d2))
    
    def test_empty_dict(self):
        """Test hashing of empty dict."""
        d = {}
        hash_val = hash_dict(d)
        
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)
    
    def test_mixed_types(self):
        """Test hashing of dict with mixed types."""
        d = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "tensor": torch.tensor([1.0]),
            "numpy": np.array([2.0]),
        }
        
        hash1 = hash_dict(d)
        hash2 = hash_dict(d)
        
        self.assertEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
