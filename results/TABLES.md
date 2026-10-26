# Aggregated experiment tables

Raw runs: **105** — `cartpole_bc`: 6, `cartpole_expert`: 3, `cartpole_hard`: 9, `cartpole_online`: 6, `cartpole_robustness`: 4, `mnist_corruption`: 12, `mnist_data`: 10, `mnist_degree`: 10, `mnist_models`: 15, `mnist_neurons`: 8, `mnist_steps`: 10, `mnist_variants`: 12

### MNIST: dense baselines versus sparse brain (10k train, 10 epochs, 3 seeds)

| model | test accuracy | trainable params | synapses | train s | p50 latency |
|---|---|---|---|---|---|
| logreg | 0.9177 ± 0.0030 | 7850 | — | 1.1 | 7 µs |
| mlp (h=6) | 0.8727 ± 0.0093 | 4780 | — | 0.9 | 14 µs |
| mlp (h=64) | 0.9465 ± 0.0004 | 50890 | — | 1.6 | 36 µs |
| sparse brain (N=256, d=8, T=8) | 0.9110 ± 0.0016 | 4585 | 2015 | 14.4 | 616 µs |
| dense RNN (N=256, T=8) | 0.9498 ± 0.0032 | 269066 | — | 9.2 | 611 µs |

### MNIST: synapses per neuron (N=256, T=8)

| degree | test accuracy | synapses | trainable params |
|---|---|---|---|
| 2 | 0.9035 ± 0.0000 | 511 | 3081 |
| 4 | 0.9098 ± 0.0023 | 1016 | 3586 |
| 8 | 0.9100 ± 0.0010 | 2014 | 4584 |
| 16 | 0.9170 ± 0.0030 | 3968 | 6538 |
| 32 | 0.9243 ± 0.0042 | 7709 | 10279 |

### MNIST: neuron count (d=8, T=8)

| neurons | test accuracy | p50 latency |
|---|---|---|
| 64 | 0.8245 ± 0.0090 | 668 µs |
| 128 | 0.8842 ± 0.0107 | 643 µs |
| 256 | 0.9100 ± 0.0010 | 573 µs |
| 512 | 0.9243 ± 0.0008 | 558 µs |

### MNIST: recurrent ticks (N=256, d=8)

| steps | test accuracy | train s |
|---|---|---|
| 1 | 0.8830 ± 0.0010 | 2.7 |
| 2 | 0.8942 ± 0.0008 | 4.1 |
| 4 | 0.9062 ± 0.0012 | 7.3 |
| 8 | 0.9100 ± 0.0010 | 13.0 |
| 16 | 0.9107 ± 0.0007 | 25.6 |

### MNIST: training-set size (N=256, d=8, T=8)

| train_size | test accuracy |
|---|---|
| 1000 | 0.8297 ± 0.0072 |
| 2500 | 0.8670 ± 0.0020 |
| 5000 | 0.8978 ± 0.0003 |
| 10000 | 0.9100 ± 0.0010 |
| 20000 | 0.9245 ± 0.0045 |

### MNIST: architecture variants (N=256, d=8, 10k train)

| variant | test accuracy | trainable params | train s |
|---|---|---|---|
| leak alpha 0.5 | 0.9120 ± 0.0005 | 4584 | 14.1 |
| leak alpha 0.9 | 0.9010 ± 0.0010 | 4584 | 13.6 |
| learned sensory fanout | 0.9293 ± 0.0018 | 7720 | 14.8 |
| no recurrence (T=0) | 0.8470 ± 0.0060 | 4584 | 1.9 |
| sensory fanout 8 | 0.9200 ± 0.0015 | 4584 | 12.9 |
| weight scale 0.35 | 0.9107 ± 0.0007 | 4584 | 13.2 |

### MNIST: distribution shift (test split, 3 seeds)

| corruption | logreg | mlp (h=6) | mlp (h=64) | sparse brain (N=256, d=8, T=8) |
|---|---|---|---|---|
| gaussian_0.1 | 0.9005 ± 0.0036 | 0.7925 ± 0.0057 | 0.9193 ± 0.0098 | 0.9018 ± 0.0041 |
| gaussian_0.2 | 0.7810 ± 0.0060 | 0.5535 ± 0.0253 | 0.7477 ± 0.0428 | 0.8442 ± 0.0073 |
| gaussian_0.3 | 0.5728 ± 0.0041 | 0.3748 ± 0.0157 | 0.5507 ± 0.0425 | 0.7287 ± 0.0151 |
| occlusion_4x4 | 0.9065 ± 0.0016 | 0.8523 ± 0.0124 | 0.9365 ± 0.0021 | 0.8955 ± 0.0043 |
| occlusion_8x8 | 0.8037 ± 0.0022 | 0.7218 ± 0.0091 | 0.8058 ± 0.0138 | 0.7817 ± 0.0066 |
| shift_2px | 0.6578 ± 0.0195 | 0.5950 ± 0.0302 | 0.7033 ± 0.0313 | 0.6502 ± 0.0095 |

### MNIST: emitted-event threshold (`HybridPELSRuntime`, every brain run)

Threshold on the per-tick state delta. Accuracy is measured through the persistent-current runtime, so `theta > 0` is the bounded-error path.

| theta | test accuracy | Δ vs matrix | active edges | active neurons | runs |
|---|---|---|---|---|---|
| 0 | 0.9034 ± 0.0273 | +0.00 pp | 96.4% | 96.4% | 56 |
| 0.02 | 0.9034 ± 0.0277 | -0.01 pp | 59.9% | 60.0% | 56 |
| 0.05 | 0.9034 ± 0.0265 | -0.00 pp | 37.4% | 37.5% | 56 |
| 0.1 | 0.9022 ± 0.0268 | -0.12 pp | 21.1% | 21.2% | 56 |
| 0.2 | 0.8979 ± 0.0249 | -0.55 pp | 8.3% | 8.3% | 56 |

### Cart-pole: closed-loop stability (100 episodes, 3 seeds)

| policy / protocol | survival steps / 200 | success rate | |angle| RMS (deg) | action switch rate | BC accuracy | recurrent ablation |
|---|---|---|---|---|---|---|
| brain (nominal) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 0.36 ± 0.02 | 1.017 ± 0.062 | 0.8868 ± 0.0103 | 200.0 ± 0.0 |
| brain (hard) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 1.12 ± 0.19 | 1.203 ± 0.026 | 0.8505 ± 0.0116 | 193.8 ± 6.4 |
| lqr (nominal) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 0.36 ± 0.02 | 1.152 ± 0.037 | — | — |
| lqr (hard) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 0.96 ± 0.04 | 1.275 ± 0.015 | — | — |
| mlp (nominal) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 0.36 ± 0.02 | 1.103 ± 0.047 | 0.9667 ± 0.0017 | — |
| mlp (hard) | 200.0 ± 0.0 | 1.0000 ± 0.0000 | 0.97 ± 0.04 | 1.197 ± 0.024 | 0.9625 ± 0.0030 | — |

### Cart-pole: robustness grid (survival steps)

| condition | brain | mlp |
|---|---|---|
| delay=1 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| delay=2 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| delay=4 | 162.8 ± 37.2 | 200.0 ± 0.0 |
| delay=6 | 37.2 ± 6.3 | 29.1 ± 0.3 |
| impulse | 200.0 ± 0.0 | 200.0 ± 0.0 |
| length_scale=0.5 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| length_scale=2 | 193.7 ± 6.3 | 200.0 ± 0.0 |
| length_scale=4 | 198.7 ± 1.3 | 38.8 ± 0.1 |
| mass_scale=0.3 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| mass_scale=3 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| noise=0.02 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| noise=0.05 | 200.0 ± 0.0 | 200.0 ± 0.0 |
| noise=0.1 | 197.3 ± 2.7 | 200.0 ± 0.0 |
| noise=0.2 | 102.2 ± 11.3 | 127.9 ± 1.3 |
| nominal | 200.0 ± 0.0 | 200.0 ± 0.0 |

### Cart-pole: online three-factor plasticity after an unannounced plant change

| variant | survival pre | survival post | adaptation gain | recovery episode | ‖ΔW‖ |
|---|---|---|---|---|---|
| frozen | 200.0 ± 0.0 | 130.3 ± 15.8 | -69.7 ± 15.8 | 10.7 ± 0.9 | 0.000 ± 0.000 |
| online | 200.0 ± 0.0 | 129.1 ± 16.2 | -70.9 ± 16.2 | 10.7 ± 0.9 | 0.800 ± 0.456 |

<details><summary>Raw run inventory</summary>

- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=brain_2dc343e8f5_seed0.json`
- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=brain_2dc343e8f5_seed1.json`
- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=brain_2dc343e8f5_seed2.json`
- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=mlp_e82837fa19_seed0.json`
- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=mlp_e82837fa19_seed1.json`
- `cartpole_bc_cartpole_bc_episodes=100_eval=closed_loop_policy=mlp_e82837fa19_seed2.json`
- `cartpole_expert_cartpole_expert_episodes=100_eval=closed_loop_policy=lqr_0c8f99ed66_seed0.json`
- `cartpole_expert_cartpole_expert_episodes=100_eval=closed_loop_policy=lqr_0c8f99ed66_seed1.json`
- `cartpole_expert_cartpole_expert_episodes=100_eval=closed_loop_policy=lqr_0c8f99ed66_seed2.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=brain_377874e08c_seed0.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=brain_377874e08c_seed1.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=brain_377874e08c_seed2.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=lqr_c1f348f005_seed0.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=lqr_c1f348f005_seed1.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=lqr_c1f348f005_seed2.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=mlp_0e45de17f6_seed0.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=mlp_0e45de17f6_seed1.json`
- `cartpole_hard_cartpole_hard_episodes=100_eval=closed_loop_hard=True_policy=mlp_0e45de17f6_seed2.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=frozen_81d07a4ed8_seed0.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=frozen_81d07a4ed8_seed1.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=frozen_81d07a4ed8_seed2.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=online_ed7226732d_seed0.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=online_ed7226732d_seed1.json`
- `cartpole_online_cartpole_online_episodes=20_eval=online_policy=brain_variant=online_ed7226732d_seed2.json`
- `cartpole_robustness_cartpole_robustness_episodes=40_eval=robustness_hard=True_policy=brain_80c6dea7fd_seed0.json`
- `cartpole_robustness_cartpole_robustness_episodes=40_eval=robustness_hard=True_policy=brain_80c6dea7fd_seed1.json`
- `cartpole_robustness_cartpole_robustness_episodes=40_eval=robustness_hard=True_policy=mlp_fe6b4dce80_seed0.json`
- `cartpole_robustness_cartpole_robustness_episodes=40_eval=robustness_hard=True_policy=mlp_fe6b4dce80_seed1.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_26f1d31403_seed0.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_26f1d31403_seed1.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_26f1d31403_seed2.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_2b4816a495_seed0.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_2b4816a495_seed1.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_2b4816a495_seed2.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_439ad82344_seed0.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_439ad82344_seed1.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_439ad82344_seed2.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_503a6d1da9_seed0.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_503a6d1da9_seed1.json`
- `mnist_corruption_mnist_alpha=0p72_batch=64_corruptions=True_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_503a6d1da9_seed2.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_7507cdc071_seed0.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_7507cdc071_seed1.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_95d85bd67d_seed0.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_95d85bd67d_seed1.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_a705832177_seed0.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_a705832177_seed1.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_a8521b0b95_seed0.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_a8521b0b95_seed1.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed0.json`
- `mnist_data_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed1.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=16_epochs=10_eval_every=0_fanout=4_gain=1p05__ce0ffea613_seed0.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=16_epochs=10_eval_every=0_fanout=4_gain=1p05__ce0ffea613_seed1.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=2_epochs=10_eval_every=0_fanout=4_gain=1p05_h_0f9e97c2b2_seed0.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=2_epochs=10_eval_every=0_fanout=4_gain=1p05_h_0f9e97c2b2_seed1.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=32_epochs=10_eval_every=0_fanout=4_gain=1p05__5cbd8cb23e_seed0.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=32_epochs=10_eval_every=0_fanout=4_gain=1p05__5cbd8cb23e_seed1.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=4_epochs=10_eval_every=0_fanout=4_gain=1p05_h_21962e91e7_seed0.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=4_epochs=10_eval_every=0_fanout=4_gain=1p05_h_21962e91e7_seed1.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed0.json`
- `mnist_degree_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_c73c226fe6_seed0.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_c73c226fe6_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_c73c226fe6_seed2.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_ce80e0731f_seed0.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_ce80e0731f_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_ce80e0731f_seed2.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_d02836b969_seed0.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_d02836b969_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_d02836b969_seed2.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_f7c6c28393_seed0.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_f7c6c28393_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_f7c6c28393_seed2.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=6_learn_be3fefd59c_seed0.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=6_learn_be3fefd59c_seed1.json`
- `mnist_models_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=6_learn_be3fefd59c_seed2.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_788be9d7fa_seed0.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_788be9d7fa_seed1.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed0.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_ab4f55d796_seed1.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_bea7131b27_seed0.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_bea7131b27_seed1.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_f6d4849dd6_seed0.json`
- `mnist_neurons_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_f6d4849dd6_seed1.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_44164ed58a_seed0.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_44164ed58a_seed1.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_7b80a8228e_seed0.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_7b80a8228e_seed1.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_b5f52d31f0_seed0.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_b5f52d31f0_seed1.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_d02836b969_seed0.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_d02836b969_seed1.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_fcaf496bff_seed0.json`
- `mnist_steps_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_fanout=4_gain=1p05_hidden=64_lear_fcaf496bff_seed1.json`
- `mnist_variants_mnist_alpha=0p5_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_dad96c7b73_seed0.json`
- `mnist_variants_mnist_alpha=0p5_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_dad96c7b73_seed1.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_0e9600dff8_seed0.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_0e9600dff8_seed1.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_8ef04b738c_seed0.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_8ef04b738c_seed1.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_9cce5ce64a_seed0.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_h_9cce5ce64a_seed1.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=8_gain=1p05_h_3829708654_seed0.json`
- `mnist_variants_mnist_alpha=0p72_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=8_gain=1p05_h_3829708654_seed1.json`
- `mnist_variants_mnist_alpha=0p9_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_9ca5b58d8a_seed0.json`
- `mnist_variants_mnist_alpha=0p9_batch=64_corruptions=False_degree=8_epochs=10_eval_every=0_fanout=4_gain=1p05_hi_9ca5b58d8a_seed1.json`

</details>
