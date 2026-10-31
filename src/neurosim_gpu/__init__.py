from pathlib import Path as _Path

# The LLM subproject lives under ``llm/`` so the repository keeps the core
# simulator and the Cortex-LM pipeline separate while both share one importable
# package namespace.
_LLM_PACKAGE_ROOT = _Path(__file__).resolve().parents[2] / "llm" / "src" / "neurosim_gpu"
if _LLM_PACKAGE_ROOT.is_dir():
    __path__.append(str(_LLM_PACKAGE_ROOT))

from .budget import UniformErrorBudget, uniform_error_budget
from .chemistry import RegionalDopamineField
from .cost_model import CostModel, estimate_cost
from .continual import (
    ContinualLearningConfig,
    ContinualLearningController,
    ReplayBatch,
    ReplayBuffer,
)
from .checkpoint import (
    capture_runtime,
    load_runtime_checkpoint,
    restore_runtime,
    save_runtime_checkpoint,
)
from .communication import DelayedMultiChannelCommunication
from .development import (
    CandidateEdgeScores,
    ConnectomeUpdate,
    MigrationConfig,
    StructuralPlasticityConfig,
    migrate_neurons,
    score_candidate_edges,
    update_structural_connectome,
)
from .dopamine import (
    DopamineReceptors,
    DopamineSignals,
    PlasticityUpdate,
    TemporalDifferenceRPE,
    dopamine_gated_weight_update,
    dopamine_gated_batch_weight_update,
    modulate_neuron_dynamics,
    update_eligibility_trace,
)
from .dynamics import Dynamics
from .gomoku import (
    BLACK,
    EMPTY,
    WHITE,
    Gomoku,
    GomokuConfig,
    GomokuEnv,
    MoveResult,
    RandomOpponent,
    RolloutResult,
    apply_move,
    canonical_board,
    canonical_encoding,
    canonical_move,
    canonical_symmetry_index,
    dihedral_symmetries,
    empty_board,
    encode_board,
    has_won,
    inverse_transform_move,
    is_draw,
    legal_move_mask,
    legal_moves,
    random_game,
    random_legal_move,
    random_rollout,
    transform_move,
    winner,
)
from .neurochem import (
    DopamineMatrixRuntime,
    DopaminePersistentRuntime,
    DopaminePersistentStats,
    NativeDopamineMatrixRuntime,
    NativeDopamineEventRuntime,
    DopaminePlasticityConfig,
    DopamineStepStats,
)
from .runtime import DenseOracle, HybridPELSRuntime, MatrixRuntime, RuntimeStats
from .scheduler import MultiRateScheduler, SchedulerConfig, SchedulerStep
from .topology import SparseTopology, random_topology
from .topology_io import (
    SingleWeightInferenceView,
    load_topology,
    memory_accounting,
    save_topology,
    single_weight_inference_view,
)

__all__ = [
    "DenseOracle",
    "SingleWeightInferenceView",
    "DelayedMultiChannelCommunication",
    "DopamineMatrixRuntime",
    "DopaminePersistentRuntime",
    "DopaminePersistentStats",
    "NativeDopamineMatrixRuntime",
    "NativeDopamineEventRuntime",
    "DopaminePlasticityConfig",
    "DopamineReceptors",
    "DopamineSignals",
    "DopamineStepStats",
    "Dynamics",
    "BLACK",
    "EMPTY",
    "WHITE",
    "Gomoku",
    "GomokuConfig",
    "GomokuEnv",
    "MoveResult",
    "RandomOpponent",
    "RolloutResult",
    "apply_move",
    "canonical_board",
    "canonical_encoding",
    "canonical_move",
    "canonical_symmetry_index",
    "dihedral_symmetries",
    "empty_board",
    "encode_board",
    "has_won",
    "inverse_transform_move",
    "is_draw",
    "legal_move_mask",
    "legal_moves",
    "random_game",
    "random_legal_move",
    "random_rollout",
    "transform_move",
    "winner",
    "HybridPELSRuntime",
    "MatrixRuntime",
    "MultiRateScheduler",
    "MigrationConfig",
    "RuntimeStats",
    "SchedulerConfig",
    "SchedulerStep",
    "PlasticityUpdate",
    "CandidateEdgeScores",
    "ConnectomeUpdate",
    "CostModel",
    "ContinualLearningConfig",
    "ContinualLearningController",
    "RegionalDopamineField",
    "ReplayBatch",
    "ReplayBuffer",
    "SparseTopology",
    "StructuralPlasticityConfig",
    "TemporalDifferenceRPE",
    "UniformErrorBudget",
    "random_topology",
    "dopamine_gated_weight_update",
    "dopamine_gated_batch_weight_update",
    "modulate_neuron_dynamics",
    "capture_runtime",
    "load_runtime_checkpoint",
    "restore_runtime",
    "save_runtime_checkpoint",
    "load_topology",
    "memory_accounting",
    "save_topology",
    "single_weight_inference_view",
    "migrate_neurons",
    "score_candidate_edges",
    "update_structural_connectome",
    "update_eligibility_trace",
    "uniform_error_budget",
    "estimate_cost",
]
