"""
IRH v21.0 Performance Optimization System

THEORETICAL FOUNDATION: IRH21.md §1.6, docs/ROADMAP.md §3 (Tier 3)

This module provides comprehensive performance optimization capabilities:
    - Caching and memoization for expensive computations
    - Vectorized numerical routines for large-scale operations
    - Performance profiling and benchmarking utilities
    - Memory optimization tools (array pooling, sparse arrays, GC tuning)
    - MPI parallelization for distributed computing
    - GPU acceleration using JAX/CuPy
    - Distributed computing with Dask/Ray for cluster-scale operations

The optimization layer maintains theoretical fidelity while achieving
significant speedups for exascale-ready computations.

Authors: IRH Computational Framework Team
Last Updated: December 2025 (synchronized with IRH21.md v21.0)

Implementation Timeline:
    Phase 3.1: NumPy Vectorization (Q1 2026) ✅
    Phase 3.2: Caching & Memoization (Q1 2026) ✅
    Phase 3.3: Memory Optimization (Q1 2026) ✅
    Phase 3.4: MPI Parallelization (Q2 2026) ✅
    Phase 3.5: GPU Acceleration (Q3 2026) ✅
    Phase 3.6: Distributed Computing (Q4 2025) ✅
"""

from __future__ import annotations

__version__ = "21.0.0"
__theoretical_foundation__ = "IRH21.md §1.6, docs/ROADMAP.md §3"

from .cache_manager import (
    CacheManager,
    LRUCache,
    DiskCache,
    create_cache,
    get_cache,
    cached,
    clear_all_caches,
    get_cache_stats,
)

from .numerical_opts import (
    vectorized_beta_functions,
    vectorized_qncd_distance,
    optimized_matrix_operations,
    batch_quaternion_multiply,
    parallel_fixed_point_search,
    VectorizedOperations,
)

from .profiling import (
    Profiler,
    profile,
    time_function,
    memory_profile,
    get_profiling_stats,
    ProfileReport,
    create_profiler,
)

from .memory_optimization import (
    ArrayPool,
    SparseFieldArray,
    MemoryMonitor,
    MemoryOptimizer,
    memory_efficient,
    get_memory_stats,
    optimize_gc,
    create_memory_mapped_array,
    estimate_memory_usage,
)

from .mpi_parallel import (
    MPIContext,
    MPIBackend,
    distributed_rg_flow,
    scatter_initial_conditions,
    gather_results,
    parallel_fixed_point_search,
    parallel_qncd_matrix,
    domain_decomposition,
    is_mpi_available,
    get_mpi_info,
)

from .gpu_acceleration import (
    GPUBackend,
    GPUContext,
    gpu_beta_functions,
    gpu_rg_flow_integration,
    gpu_qncd_matrix,
    gpu_quaternion_multiply,
    is_gpu_available,
    get_gpu_info,
    get_available_backends,
    set_default_backend,
    benchmark_gpu_performance,
)

from .distributed import (
    DistributedBackend,
    DistributedContext,
    dask_rg_flow,
    ray_parameter_scan,
    distributed_monte_carlo,
    distributed_qncd_matrix,
    distributed_map,
    is_dask_available,
    is_ray_available,
    get_distributed_info,
    get_available_distributed_backends,
    create_local_cluster,
    shutdown_cluster,
)

__all__ = [
    # Cache Management
    'CacheManager',
    'LRUCache',
    'DiskCache',
    'create_cache',
    'get_cache',
    'cached',
    'clear_all_caches',
    'get_cache_stats',
    
    # Vectorized Operations
    'vectorized_beta_functions',
    'vectorized_qncd_distance',
    'optimized_matrix_operations',
    'batch_quaternion_multiply',
    'parallel_fixed_point_search',
    'VectorizedOperations',
    
    # Profiling
    'Profiler',
    'profile',
    'time_function',
    'memory_profile',
    'get_profiling_stats',
    'ProfileReport',
    'create_profiler',
    
    # Memory Optimization
    'ArrayPool',
    'SparseFieldArray',
    'MemoryMonitor',
    'MemoryOptimizer',
    'memory_efficient',
    'get_memory_stats',
    'optimize_gc',
    'create_memory_mapped_array',
    'estimate_memory_usage',
    
    # MPI Parallelization
    'MPIContext',
    'MPIBackend',
    'distributed_rg_flow',
    'scatter_initial_conditions',
    'gather_results',
    'parallel_fixed_point_search',
    'parallel_qncd_matrix',
    'domain_decomposition',
    'is_mpi_available',
    'get_mpi_info',
    
    # GPU Acceleration
    'GPUBackend',
    'GPUContext',
    'gpu_beta_functions',
    'gpu_rg_flow_integration',
    'gpu_qncd_matrix',
    'gpu_quaternion_multiply',
    'is_gpu_available',
    'get_gpu_info',
    'get_available_backends',
    'set_default_backend',
    'benchmark_gpu_performance',
    
    # Distributed Computing (Dask/Ray)
    'DistributedBackend',
    'DistributedContext',
    'dask_rg_flow',
    'ray_parameter_scan',
    'distributed_monte_carlo',
    'distributed_qncd_matrix',
    'distributed_map',
    'is_dask_available',
    'is_ray_available',
    'get_distributed_info',
    'get_available_distributed_backends',
    'create_local_cluster',
    'shutdown_cluster',
]
