from __future__ import annotations

import unittest

import torch

from neurosim_gpu.communication import DelayedMultiChannelCommunication


class DelayedMultiChannelCommunicationTests(unittest.TestCase):
    @staticmethod
    def devices() -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    @staticmethod
    def build(
        device: torch.device,
        *,
        src: list[int],
        dst: list[int],
        weight: list[float],
        delay: list[int],
        transmitter: list[int],
        gain: list[float] | list[list[float]],
        n_neurons: int = 4,
    ) -> DelayedMultiChannelCommunication:
        return DelayedMultiChannelCommunication(
            n_neurons=n_neurons,
            src=torch.tensor(src, dtype=torch.int64, device=device),
            dst=torch.tensor(dst, dtype=torch.int64, device=device),
            weight=torch.tensor(weight, dtype=torch.float64, device=device),
            delay=torch.tensor(delay, dtype=torch.int64, device=device),
            transmitter=torch.tensor(
                transmitter, dtype=torch.int64, device=device
            ),
            receptor_gain=torch.tensor(gain, dtype=torch.float64, device=device),
        )

    def test_zero_delay_matches_direct_edge_scatter_and_skips_ring(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                module = self.build(
                    device,
                    src=[2, 0, 1, 0],
                    dst=[1, 2, 3, 1],
                    weight=[0.5, -2.0, 1.25, 3.0],
                    delay=[0, 0, 0, 0],
                    transmitter=[1, 0, 1, 1],
                    gain=[2.0, 0.25],
                )
                state = torch.tensor(
                    [[1.0, 2.0, 4.0, 8.0], [-1.0, 3.0, 0.5, 2.0]],
                    dtype=torch.float64,
                    device=device,
                )
                actual = module(state)

                expected = torch.zeros_like(state)
                for edge in range(module.n_edges):
                    expected[:, module.dst[edge]] += (
                        state[:, module.src[edge]]
                        * module.weight[edge]
                        * module.edge_receptor_gain[edge]
                    )
                torch.testing.assert_close(actual, expected)
                self.assertEqual(module.history.numel(), 0)
                self.assertTrue(bool(torch.all(module.src[1:] >= module.src[:-1])))

    def test_exact_delay_timing_has_no_future_or_missing_history_leak(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                module = self.build(
                    device,
                    src=[0],
                    dst=[1],
                    weight=[1.0],
                    delay=[2],
                    transmitter=[0],
                    gain=[1.0],
                    n_neurons=2,
                )
                delivered = []
                for value in (1.0, 2.0, 4.0, 8.0, 16.0):
                    state = torch.tensor(
                        [[value, 1000.0 + value]],
                        dtype=torch.float64,
                        device=device,
                    )
                    delivered.append(float(module.step(state)[0, 1]))
                self.assertEqual(delivered, [0.0, 0.0, 1.0, 2.0, 4.0])

    def test_destination_local_channel_gains_select_receptors(self) -> None:
        device = self.devices()[-1]
        module = self.build(
            device,
            src=[0, 0, 0, 0],
            dst=[1, 1, 2, 2],
            weight=[1.0, 1.0, 1.0, 1.0],
            delay=[0, 0, 0, 0],
            transmitter=[0, 1, 0, 1],
            gain=[
                [1.0, 1.0],
                [2.0, 0.5],
                [0.25, 4.0],
                [1.0, 1.0],
            ],
        )
        state = torch.tensor(
            [[3.0, 0.0, 0.0, 0.0]], dtype=torch.float64, device=device
        )
        actual = module(state)
        expected = torch.tensor(
            [[0.0, 7.5, 12.75, 0.0]], dtype=torch.float64, device=device
        )
        torch.testing.assert_close(actual, expected)

    def test_batches_are_isolated(self) -> None:
        for device in self.devices():
            with self.subTest(device=device):
                kwargs = dict(
                    src=[0, 1, 2],
                    dst=[1, 2, 0],
                    weight=[1.0, -0.5, 2.0],
                    delay=[1, 2, 0],
                    transmitter=[0, 1, 0],
                    gain=[1.0, 0.25],
                    n_neurons=3,
                )
                together = self.build(device, **kwargs)
                separate = [self.build(device, **kwargs) for _ in range(2)]
                sequences = [
                    torch.tensor(
                        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.tensor(
                        [[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]],
                        dtype=torch.float64,
                        device=device,
                    ),
                    torch.tensor(
                        [[7.0, 8.0, 9.0], [70.0, 80.0, 90.0]],
                        dtype=torch.float64,
                        device=device,
                    ),
                ]
                for state in sequences:
                    batched = together(state)
                    expected = torch.cat(
                        [
                            separate[index](state[index : index + 1])
                            for index in range(2)
                        ],
                        dim=0,
                    )
                    torch.testing.assert_close(batched, expected)

    def test_ring_wraparound_preserves_maximum_delay(self) -> None:
        device = self.devices()[-1]
        module = self.build(
            device,
            src=[0, 0],
            dst=[1, 2],
            weight=[1.0, 1.0],
            delay=[1, 3],
            transmitter=[0, 0],
            gain=[1.0],
            n_neurons=3,
        )
        for tick in range(14):
            state = torch.tensor(
                [[float(tick + 1), 0.0, 0.0]],
                dtype=torch.float64,
                device=device,
            )
            current = module(state)
            expected_d1 = 0.0 if tick < 1 else float(tick)
            expected_d3 = 0.0 if tick < 3 else float(tick - 2)
            torch.testing.assert_close(
                current,
                torch.tensor(
                    [[0.0, expected_d1, expected_d3]],
                    dtype=torch.float64,
                    device=device,
                ),
            )
        self.assertEqual(module.history.shape, (4, 1, 3))

    def test_signed_weight_is_independent_of_transmitter_channel(self) -> None:
        device = self.devices()[-1]
        module = self.build(
            device,
            src=[0, 0, 0],
            dst=[1, 1, 2],
            weight=[2.0, -0.75, -3.0],
            delay=[0, 0, 0],
            transmitter=[0, 0, 1],
            gain=[2.0, 2.0],
            n_neurons=3,
        )
        current = module(
            torch.tensor(
                [[4.0, 0.0, 0.0]], dtype=torch.float64, device=device
            )
        )
        torch.testing.assert_close(
            current,
            torch.tensor(
                [[0.0, 10.0, -24.0]], dtype=torch.float64, device=device
            ),
        )
        self.assertEqual(module.transmitter.tolist(), [0, 0, 1])
        self.assertEqual(module.weight.sign().tolist(), [1.0, -1.0, -1.0])

    def test_reset_and_runtime_validation(self) -> None:
        device = self.devices()[-1]
        module = self.build(
            device,
            src=[0],
            dst=[1],
            weight=[1.0],
            delay=[1],
            transmitter=[0],
            gain=[1.0],
            n_neurons=2,
        )
        state = torch.ones((2, 2), dtype=torch.float64, device=device)
        module(state)
        with self.assertRaisesRegex(ValueError, "batch size changed"):
            module(state[:1])
        module.reset_history(batch_size=1)
        torch.testing.assert_close(module(state[:1]), torch.zeros_like(state[:1]))
        self.assertEqual(module.tick, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            module(torch.ones((1, 3), dtype=torch.float64, device=device))
        with self.assertRaisesRegex(ValueError, "dtype"):
            module(torch.ones((1, 2), dtype=torch.float32, device=device))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            module.reset_history(batch_size=0)

    def test_graph_validation(self) -> None:
        device = torch.device("cpu")
        valid = dict(
            n_neurons=2,
            src=torch.tensor([0], dtype=torch.int64, device=device),
            dst=torch.tensor([1], dtype=torch.int64, device=device),
            weight=torch.tensor([1.0], dtype=torch.float64, device=device),
            delay=torch.tensor([0], dtype=torch.int64, device=device),
            transmitter=torch.tensor([0], dtype=torch.int64, device=device),
            receptor_gain=torch.tensor([1.0], dtype=torch.float64, device=device),
        )
        cases = [
            ("positive", {"n_neurons": 0}),
            ("equal length", {"dst": torch.tensor([], dtype=torch.int64)}),
            ("source neuron", {"src": torch.tensor([2])}),
            ("destination neuron", {"dst": torch.tensor([-1])}),
            ("non-negative", {"delay": torch.tensor([-1])}),
            ("channel", {"transmitter": torch.tensor([1])}),
            ("floating point", {"weight": torch.tensor([1])}),
            (
                "same dtype",
                {"receptor_gain": torch.tensor([1.0], dtype=torch.float32)},
            ),
            (
                "n_neurons rows",
                {
                    "receptor_gain": torch.ones(
                        (3, 1), dtype=torch.float64
                    )
                },
            ),
        ]
        for message, replacement in cases:
            with self.subTest(message=message):
                arguments = valid | replacement
                with self.assertRaisesRegex(ValueError, message):
                    DelayedMultiChannelCommunication(**arguments)


if __name__ == "__main__":
    unittest.main()
