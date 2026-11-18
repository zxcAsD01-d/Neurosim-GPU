from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from neurosim_gpu.precision import (
    _e2m1_constants, encode_sparse_weights, quantize_dequantize,
)


class PrecisionConstantTests(unittest.TestCase):
    def test_tables_are_reused_and_exact_after_inference_warmup(self):
        device = torch.device("cpu")
        with torch.inference_mode():
            midpoints, levels = _e2m1_constants(device, torch.float32)
        reused = _e2m1_constants(device, torch.float32)
        self.assertIs(midpoints, reused[0])
        self.assertIs(levels, reused[1])
        self.assertFalse(midpoints.is_inference())
        self.assertFalse(levels.is_inference())
        torch.testing.assert_close(
            midpoints, torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]),
            atol=0, rtol=0,
        )
        torch.testing.assert_close(
            levels, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
            atol=0, rtol=0,
        )

    def test_nvfp4_does_not_create_per_call_tensor_constants(self):
        value = torch.linspace(-3, 3, 77).reshape(7, 11).requires_grad_()
        expected = quantize_dequantize(value.detach(), "nvfp4")
        with patch.object(torch.Tensor, "new_tensor", side_effect=AssertionError("new constant")):
            actual = quantize_dequantize(value, "nvfp4")
            encode_sparse_weights(value.detach().flatten(), "nvfp4")
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.sum().backward()
        torch.testing.assert_close(value.grad, torch.ones_like(value), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for graph capture")
    def test_warmed_nvfp4_quantization_replays_with_new_inputs(self):
        # Warm the exact device tables and operations before stream capture.
        value = torch.linspace(-3, 3, 77, device="cuda").reshape(7, 11)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                quantize_dequantize(value, "nvfp4")
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream), torch.no_grad():
            actual = quantize_dequantize(value, "nvfp4")
        with torch.no_grad():
            value.mul_(0.73).add_(0.07)
            expected = quantize_dequantize(value, "nvfp4")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
