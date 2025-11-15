"""
Orchestry MARL - Multi-Agent Reinforcement Learning for LLMs.

API-based MARL implementation using:
- Group Relative Policy Optimization (GRPO)
- Multi-sample trajectory search
- Centralized value estimation
- Behavior pattern extraction
"""

from .trajectory import MultiTurnTrajectory, Turn, TrajectoryBeam
from .value_estimator import CentralizedValueEstimator
from .api_grpo import APIGroupRelativePolicyOptimizer, Agent, ResponseCache
from .behavior_library import BehaviorLibrary
from .trainer import MARLTrainer

__version__ = "1.0.0-marl"

__all__ = [
    'MultiTurnTrajectory',
    'Turn',
    'TrajectoryBeam',
    'CentralizedValueEstimator',
    'APIGroupRelativePolicyOptimizer',
    'Agent',
    'ResponseCache',
    'BehaviorLibrary',
    'MARLTrainer',
]
