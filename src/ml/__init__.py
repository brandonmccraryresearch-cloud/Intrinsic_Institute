"""
Machine Learning Surrogate Models for IRH

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §4.3 (Roadmap Phase 4.3)

This module provides neural network surrogate models for accelerating
IRH computations, including:
- RG flow trajectory approximation
- Uncertainty quantification via ensemble methods
- Bayesian parameter optimization

These surrogates enable real-time exploration of the parameter space
while maintaining theoretical accuracy through uncertainty tracking.

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from __future__ import annotations

__version__ = "21.1.0"
__theoretical_foundation__ = "IRH v21.1 Manuscript, Phase 4.3 (ML Surrogate Models)"

# Import main classes and functions
from .rg_flow_surrogate import (
    RGFlowSurrogate,
    create_rg_flow_surrogate,
    train_rg_flow_surrogate,
    predict_rg_trajectory,
    SurrogateConfig,
)

from .uncertainty_quantification import (
    UncertaintyEstimator,
    EnsembleUncertainty,
    MCDropoutUncertainty,
    compute_uncertainty,
    calibrate_uncertainty,
)

from .parameter_optimizer import (
    ParameterOptimizer,
    BayesianOptimizer,
    ActiveLearningOptimizer,
    optimize_parameters,
    suggest_next_point,
)

__all__ = [
    # Surrogate model
    'RGFlowSurrogate',
    'create_rg_flow_surrogate',
    'train_rg_flow_surrogate',
    'predict_rg_trajectory',
    'SurrogateConfig',
    
    # Uncertainty
    'UncertaintyEstimator',
    'EnsembleUncertainty',
    'MCDropoutUncertainty',
    'compute_uncertainty',
    'calibrate_uncertainty',
    
    # Optimization
    'ParameterOptimizer',
    'BayesianOptimizer',
    'ActiveLearningOptimizer',
    'optimize_parameters',
    'suggest_next_point',
]
