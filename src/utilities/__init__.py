"""
Cross-Cutting Computational Utilities for Intrinsic Resonance Holography v21.0

This module provides shared numerical infrastructure used across all layers
of the computational framework. These utilities are theory-agnostic and can
be used independently of the IRH-specific modules.

Modules:
    instrumentation: Theoretical logging and traceability (Phase II) ✓ COMPLETE
    output_contextualization: Standardized outputs with provenance (Phase III) ✓ COMPLETE
    integration: Numerical quadrature on group manifolds ✓ COMPLETE
    optimization: Fixed-point solvers, minimizers ✓ COMPLETE
    special_functions: Bessel, hypergeometric, Wigner D-matrices ✓ COMPLETE
    lattice_discretization: Finite-volume approximations ✓ COMPLETE
    parallel_computing: Thread/process parallelization ✓ COMPLETE

Design Principles:
    1. No IRH-specific assumptions
    2. High numerical precision (configurable)
    3. Parallelizable where appropriate
    4. Well-documented numerical methods

Dependencies:
    - NumPy, SciPy (numerical computing)
    - Optional: JAX (GPU acceleration)
    - Optional: mpi4py (distributed computing)

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)
"""

__version__ = "21.0.0"

# Import instrumentation module (Phase II)
from .instrumentation import (
    IRHLogLevel,
    TheoreticalReference,
    ComputationContext,
    IRHLogger,
    instrumented,
    get_logger,
    configure_logging,
)

# Import output contextualization module (Phase III)
from .output_contextualization import (
    ComputationType,
    TheoreticalContext,
    ComputationalProvenance,
    ObservableResult,
    UncertaintyTracker,
    IRHOutputWriter,
    create_output_writer,
    format_observable,
)

# Import integration module
from .integration import (
    integrate_SU2,
    integrate_U1,
    integrate_G_inf,
    monte_carlo_integrate,
)

# Import optimization module
from .optimization import (
    find_fixed_point_newton,
    minimize_functional,
    root_find,
)

# Import special functions module
from .special_functions import (
    bessel_j,
    hypergeometric_2f1,
    wigner_d_matrix,
    wigner_D_matrix,
    clebsch_gordan,
    spherical_harmonic,
)

# Import lattice discretization module
from .lattice_discretization import (
    discretize_SU2,
    discretize_U1,
    laplacian_matrix,
    lattice_volume,
)

# Import parallel computing module
from .parallel_computing import (
    parallel_map,
    distributed_sum,
    batch_compute,
    get_optimal_workers,
)

__all__ = [
    # instrumentation exports (Phase II)
    'IRHLogLevel',
    'TheoreticalReference',
    'ComputationContext',
    'IRHLogger',
    'instrumented',
    'get_logger',
    'configure_logging',
    
    # output_contextualization exports (Phase III)
    'ComputationType',
    'TheoreticalContext',
    'ComputationalProvenance',
    'ObservableResult',
    'UncertaintyTracker',
    'IRHOutputWriter',
    'create_output_writer',
    'format_observable',
    
    # integration exports
    'integrate_SU2',
    'integrate_U1',
    'integrate_G_inf',
    'monte_carlo_integrate',
    
    # optimization exports
    'find_fixed_point_newton',
    'minimize_functional',
    'root_find',
    
    # special_functions exports
    'bessel_j',
    'hypergeometric_2f1',
    'wigner_d_matrix',
    'wigner_D_matrix',
    'clebsch_gordan',
    'spherical_harmonic',
    
    # lattice_discretization exports
    'discretize_SU2',
    'discretize_U1',
    'laplacian_matrix',
    'lattice_volume',
    
    # parallel_computing exports
    'parallel_map',
    'distributed_sum',
    'batch_compute',
    'get_optimal_workers',
]
