from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import torch

from neurosim_gpu import capture_runtime, load_runtime_checkpoint
from neurosim_gpu.simulate import build_runtime, parser, run


class MainlineSimulationTests(unittest.TestCase):
    def args(self, *extra):
        return parser().parse_args(["--neurons", "12", "--edges", "48",
                                    "--batch", "2", "--steps", "5", *extra])

    def test_default_is_complete_portable_matrix(self):
        runtime = build_runtime(self.args())
        self.assertEqual(type(runtime).__name__, "DendriticMatrixRuntime")
        self.assertEqual(runtime.state.shape, (2, 12))
        result = run(self.args())
        self.assertFalse(result["approximate"])
        self.assertEqual(result["backend"], "reference")
        self.assertGreater(result["states"]["dendritic_state"]["mean"], 0)
        self.assertGreater(result["states"]["adaptation_state"]["mean"], 0)

    def test_zero_threshold_pels_matches_matrix(self):
        matrix = build_runtime(self.args())
        pels = build_runtime(self.args("--propagation", "pels"))
        for _ in range(30):
            matrix.step(0.2, dendritic_drive=0.1)
            pels.step(0.2, dendritic_drive=0.1)
        for name in ("state", "dendritic_state", "adaptation_state"):
            torch.testing.assert_close(getattr(matrix, name), getattr(pels, name),
                                       atol=2e-6, rtol=2e-4)

    def test_checkpoint_resume_and_summary_for_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / "states.pt")
            summary = str(Path(tmp) / "summary.json")
            for path in ("matrix", "pels"):
                extra = ["--propagation", path]
                if path == "pels":
                    extra += ["--theta", "0.02"]
                run(self.args(*extra, "--checkpoint", checkpoint))
                result = run(self.args(*extra, "--resume", checkpoint,
                                       "--checkpoint", checkpoint, "--output", summary))
                self.assertEqual(json.loads(Path(summary).read_text()), result)
                self.assertTrue(result["resumed"])
                self.assertEqual(result["approximate"], path == "pels")
                resumed, continuous = build_runtime(self.args(*extra)), build_runtime(self.args(*extra))
                load_runtime_checkpoint(checkpoint, resumed)
                for _ in range(10):
                    continuous.step(0.2, dendritic_drive=0.1)
                expected = capture_runtime(continuous)["tensors"]
                for name, value in capture_runtime(resumed)["tensors"].items():
                    torch.testing.assert_close(value, expected[name], atol=0, rtol=0)

    def test_rejects_invalid_or_misleading_options(self):
        for extra in (("--backend", "native"), ("--theta", "0.1"),
                      ("--theta", "nan"), ("--drive", "inf"),
                      ("--neurons", "0"), ("--edges", "-1"),
                      ("--steps", "0"), ("--batch", "0"),
                      ("--tau-dendrite", "0"), ("--coupling", "2"),
                      ("--propagation", "pels", "--theta", "-1"),
                      ("--precision", "fp16"),
                      ("--device", "cuda", "--backend", "native", "--propagation", "pels",
                       "--precision", "nvfp4")):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_runtime(self.args(*extra))

    def test_cpu_quantized_simulation_reports_execution(self):
        for name in ("fp8", "nvfp4"):
            result = run(self.args("--precision", name, "--propagation", "pels"))
            self.assertEqual(result["precision"], name)
            self.assertIn("qdq-emulation", result["precision_execution"])
            self.assertEqual(result["dtype"], "torch.float32")
            self.assertTrue(result["approximate"])

    def test_summary_cannot_overwrite_checkpoint_or_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "states.pt"
            alias = root / "sub" / ".." / checkpoint.name
            with self.assertRaisesRegex(ValueError, "different file from --checkpoint"):
                run(self.args("--checkpoint", str(checkpoint), "--output", str(alias)))
            self.assertFalse(checkpoint.exists())

            run(self.args("--checkpoint", str(checkpoint)))
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "different file from --resume"):
                run(self.args("--resume", str(checkpoint), "--output", str(checkpoint)))
            self.assertEqual(checkpoint.read_bytes(), before)

    def test_summary_cannot_overwrite_hard_link_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "states.pt"
            alias = Path(tmp) / "alias.json"
            run(self.args("--checkpoint", str(checkpoint)))
            before = checkpoint.read_bytes()
            try:
                os.link(checkpoint, alias)
            except OSError as error:
                self.skipTest(f"hard links unavailable: {error}")
            with self.assertRaisesRegex(ValueError, "different file from --resume"):
                run(self.args("--resume", str(checkpoint), "--output", str(alias)))
            self.assertEqual(checkpoint.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
