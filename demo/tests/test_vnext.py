"""Tests for the vNext causal-control experiment."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

DEMO = Path(__file__).resolve().parents[1]
ROOT = DEMO.parent
for path in (DEMO, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from neurobody.embodied import EmbodiedConfig  # noqa: E402
from neurobody.vnext import (  # noqa: E402
    BrainPolicy,
    PDPolicy,
    ServoTask,
    TrainConfig,
    VNextBrain,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_brain,
)


def _small_config() -> EmbodiedConfig:
    # Keep the functional M1 slice (800..2240) valid while making CPU tests fast.
    return EmbodiedConfig(
        brain_neurons=2240,
        degree=8,
        visual_per_eye=8,
        motor_per_joint=2,
    )


class GraphTests(unittest.TestCase):
    def test_m1_projection_and_error_connectivity(self) -> None:
        brain = VNextBrain(
            _small_config(),
            n_symbols=1,
            seed=7,
            m1_to_motor=8,
            error_to_brain=0,
            error_to_motor=False,
        )
        self.assertEqual((brain.m1.start, brain.m1.stop), (800, 2240))
        src = brain.src.long()
        dst = brain.dst.long()
        motor = brain.motor
        brain_to_motor = ((src < brain.config.brain_neurons) & (dst >= motor.start) & (dst < motor.stop)).sum()
        m1_to_motor = (
            (src >= brain.m1.start)
            & (src < brain.m1.stop)
            & (dst >= motor.start)
            & (dst < motor.stop)
        ).sum()
        self.assertGreater(int(brain_to_motor), 0)
        self.assertGreater(int(m1_to_motor), 0)
        # The error neurons are part of the sensory block, so they already have
        # random projections to the brain; the optional extra path is off here.
        error_start = brain.sensory.start + brain.config.n_joints * 3 + 16
        error_to_brain = (
            (src >= error_start) & (src < error_start + brain.config.n_joints) & (dst < brain.config.brain_neurons)
        ).sum()
        self.assertGreater(int(error_to_brain), 0)
        error_to_motor = (
            (src >= error_start)
            & (src < error_start + brain.config.n_joints)
            & (dst >= motor.start)
            & (dst < motor.stop)
        ).sum()
        self.assertEqual(int(error_to_motor), 0)

    def test_error_to_motor_flag_disables_direct_bypass(self) -> None:
        config = _small_config()
        disabled = VNextBrain(
            config, n_symbols=1, seed=7, error_to_motor=False,
            reflex_gain=0.0, m1_to_motor=0,
        )
        enabled = VNextBrain(
            config, n_symbols=1, seed=7, error_to_motor=True,
            reflex_gain=0.0, m1_to_motor=0,
        )
        state = torch.full((1, config.n_neurons), 0.5)
        state[:, disabled.error_slice] = 1.0
        base = state.clone()
        base[:, disabled.error_slice] = 0.5
        self.assertTrue(torch.equal(disabled.motor_velocity(state), disabled.motor_velocity(base)))
        self.assertFalse(torch.equal(enabled.motor_velocity(state), enabled.motor_velocity(base)))


class TaskTests(unittest.TestCase):
    def test_posture_shapes_and_pd(self) -> None:
        task = ServoTask("posture", batch=2, device=torch.device("cpu"), episode_ticks=8, push_tick=4)
        q, qd, vest, target = task.reset()
        self.assertEqual(q.shape, (2, 41))
        self.assertEqual(vest.shape, (2, 10))
        policy = PDPolicy()
        velocity = policy.act(q, qd, target, task.v_cmd, max_step=task.max_step)
        self.assertEqual(velocity.shape, (2, 41))
        self.assertTrue(torch.isfinite(velocity).all())
        q, qd, vest, target = task.step(velocity)
        self.assertTrue(torch.isfinite(q).all())

    def test_velocity_command_channel(self) -> None:
        task = ServoTask("velocity", batch=2, device=torch.device("cpu"), episode_ticks=8)
        q, qd, vest, target = task.reset()
        self.assertTrue((target[:, 39:] - q[:, 39:]).abs().sum() >= 0)
        self.assertEqual(vest[:, 0].shape, (2,))

    def test_switch_target_changes(self) -> None:
        task = ServoTask(
            "switch",
            batch=2,
            device=torch.device("cpu"),
            episode_ticks=9,
            switch_b_scale=0.5,
        )
        q, qd, vest, target = task.reset()
        self.assertGreater(float(task.target_b.abs().sum()), 0.0)
        for _ in range(4):
            q, qd, vest, target = task.step(torch.zeros_like(q))
        self.assertGreater(float(task.q_goal.abs().sum()), 0.0)
        self.assertEqual(task.q_goal.shape, (41,))

    def test_goal_modes(self) -> None:
        for mode in ("cycle", "continuous", "random"):
            task = ServoTask(
                "goals",
                batch=1,
                device=torch.device("cpu"),
                episode_ticks=9,
                goal_mode=mode,
                goal_dwell=3,
            )
            q, qd, vest, target = task.reset()
            for _ in range(4):
                q, qd, vest, target = task.step(torch.zeros_like(q))
            self.assertEqual(task.q_goal.shape, (41,))
            self.assertTrue(torch.isfinite(target).all())


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.manual_seed(0)
        cls.brain = VNextBrain(_small_config(), n_symbols=1, seed=7, m1_to_motor=4)

    def test_plan_and_counterfactual_shapes(self) -> None:
        brain = VNextBrain(
            _small_config(),
            n_symbols=1,
            seed=7,
            m1_to_motor=4,
            plan_horizon=4,
        )
        state = torch.zeros(2, _small_config().n_neurons)
        goal = torch.zeros(2, 41)
        q = torch.zeros(2, 41)
        plan = brain.plan(state, goal, q)
        self.assertEqual(plan.shape, (2, 4, 41))
        imagined = plan + 0.01
        error = brain.counterfactual_error(state, goal, q, imagined)
        self.assertEqual(error.shape, (2,))

    def test_plain_dynamics(self) -> None:
        brain = VNextBrain(
            _small_config(),
            n_symbols=1,
            seed=7,
            dynamics="plain",
            m1_to_motor=4,
        )
        q = torch.zeros(2, 41)
        qd = torch.zeros(2, 41)
        vest = torch.zeros(2, 10)
        target = torch.zeros(2, 41)
        state = brain.initial_state(brain.encode(q, qd, vest, None, target))
        state = brain.advance(state, brain.encode(q, qd, vest, None, target))
        self.assertTrue(torch.isfinite(state).all())

    def test_dendritic_core_carries_and_detaches_all_states(self) -> None:
        brain = VNextBrain(
            _small_config(), n_symbols=1, seed=7,
            dynamics="dendritic", m1_to_motor=4,
        )
        drive = torch.full((2, brain.config.n_neurons), 0.1, requires_grad=True)
        state = brain.initial_neural_state(drive)
        self.assertIsInstance(state, tuple)
        for _ in range(3):
            state = brain.advance(state, drive)
        self.assertTrue(torch.isfinite(brain.soma_activity(state)).all())
        self.assertGreater(float(state[1].detach().abs().max()), 0.0)
        self.assertGreater(float(state[2].detach().abs().max()), 0.0)
        detached = brain.detach_neural_state(state)
        self.assertTrue(all(value.grad_fn is None for value in detached))

    def test_dendritic_edges_obey_source_sign(self) -> None:
        for brain in (
            VNextBrain(
                _small_config(), n_symbols=1, seed=7,
                dynamics="dendritic", m1_to_motor=4,
            ),
            VNextBrain(
                _small_config(), n_symbols=1, seed=7,
                dynamics="neurosim", dale_constrained=True, m1_to_motor=4,
            ),
        ):
            brain.to(torch.device("cpu"))
            sparse = brain._csr()
            expected = brain.edge_weight.abs()
            source_type = brain.type_ids[brain.src.long()]
            inhibitory = (source_type == 1) | (source_type == 5)
            expected = torch.where(inhibitory, -expected, expected)
            self.assertTrue(torch.equal(
                sparse.values(), expected[brain.in_weight_index.long()]
            ))

    def test_dendritic_training_smoke(self) -> None:
        brain = VNextBrain(
            _small_config(), n_symbols=1, seed=9,
            dynamics="dendritic", m1_to_motor=4,
        )
        task = ServoTask(
            "goals", batch=1, device=torch.device("cpu"),
            episode_ticks=4, goal_dwell=2,
        )
        result = train_brain(
            brain, task,
            TrainConfig(ticks=8, batch=1, lr=1e-3, window=2,
                        episode_ticks=4, log_every=8, self_weight=0.0),
            device=torch.device("cpu"),
        )
        self.assertEqual(result["updates"], 4)
        self.assertTrue(torch.isfinite(brain.edge_weight).all())

    def test_dendritic_checkpoint_roundtrip(self) -> None:
        brain = VNextBrain(
            _small_config(), n_symbols=1, seed=13,
            dynamics="dendritic", m1_to_motor=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dendritic.pt"
            save_checkpoint(path, brain, {"core": "dendritic"})
            loaded, meta = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(meta["core"], "dendritic")
            self.assertEqual(loaded.dynamics, "dendritic")
            q = torch.zeros(1, 41)
            drive = loaded.encode(q, q, torch.zeros(1, 10), None, q)
            state = loaded.initial_neural_state(drive)
            state = loaded.advance(state, drive)
            self.assertTrue(torch.isfinite(loaded.soma_activity(state)).all())

    def test_checkpoint_preserves_independent_dale_flag(self) -> None:
        brain = VNextBrain(
            _small_config(), n_symbols=1, seed=14,
            dynamics="neurosim", dale_constrained=True, m1_to_motor=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "point-dale.pt"
            save_checkpoint(path, brain, {"test": True})
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(loaded.dynamics, "neurosim")
            self.assertTrue(loaded.dale_constrained)
            self.assertTrue(torch.equal(
                loaded._csr().values(),
                brain._csr().values(),
            ))

    def test_train_smoke(self) -> None:
        brain = VNextBrain(_small_config(), n_symbols=1, seed=7, m1_to_motor=4)
        task = ServoTask("posture", batch=2, device=torch.device("cpu"), episode_ticks=8, push_tick=4)
        result = train_brain(
            brain,
            task,
            TrainConfig(ticks=16, batch=2, lr=1e-3, log_every=8, self_weight=0.0),
            device=torch.device("cpu"),
        )
        self.assertGreater(result["updates"], 0)
        self.assertTrue(result["mean_loss"] == result["mean_loss"])  # not NaN
        self.assertGreater(result["mean_loss"], 0.0)

    def test_evaluate_pd(self) -> None:
        metrics = evaluate(
            PDPolicy(),
            ServoTask("posture", batch=1, device=torch.device("cpu"), episode_ticks=32, push_tick=8),
            episodes=2,
            device=torch.device("cpu"),
        )
        self.assertIn("postureError", metrics)
        self.assertTrue(metrics["postureError"] == metrics["postureError"])

    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, self.brain, {"test": True})
            loaded, meta = load_checkpoint(path, torch.device("cpu"))
            self.assertTrue(meta["test"])
            self.assertEqual(loaded.edge_weight.numel(), self.brain.edge_weight.numel())
            self.assertTrue(torch.equal(loaded.src, self.brain.src))
            self.assertTrue(torch.equal(loaded.dst, self.brain.dst))

    def test_legacy_checkpoint_restores_saved_topology_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            save_checkpoint(path, self.brain, {"legacy": True})
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint.pop("seed")
            torch.save(checkpoint, path)
            loaded, meta = load_checkpoint(path, torch.device("cpu"), seed=999)
            self.assertTrue(meta["legacy"])
            self.assertTrue(torch.equal(loaded.src, self.brain.src))
            self.assertTrue(torch.equal(loaded.dst, self.brain.dst))
            self.assertTrue(torch.equal(loaded.in_rowptr, self.brain.in_rowptr))

    def test_pre_pool_checkpoint_infers_6020_style_neuron_count(self) -> None:
        config = _small_config()
        config.touch_neurons = 0
        config.audio_neurons = 0
        config.nocicept_neurons = 0
        config.memory_neurons = 0
        legacy = VNextBrain(config, n_symbols=1, seed=17, m1_to_motor=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pre-pool.pt"
            save_checkpoint(path, legacy, {"format": "pre-pool"})
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            for name in ("touch_neurons", "audio_neurons", "nocicept_neurons", "memory_neurons"):
                checkpoint["config"].pop(name)
            torch.save(checkpoint, path)
            loaded, meta = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(meta["format"], "pre-pool")
            self.assertEqual(loaded.config.n_neurons, legacy.config.n_neurons)
            self.assertEqual(loaded.config.touch_neurons, 0)
            self.assertTrue(torch.equal(loaded.src, legacy.src))


if __name__ == "__main__":
    unittest.main()
