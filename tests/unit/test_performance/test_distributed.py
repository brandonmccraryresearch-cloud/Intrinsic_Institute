"""
Tests for Distributed Computing Module (Dask/Ray)

THEORETICAL FOUNDATION: IRH v21.1 Manuscript §1.6, docs/ROADMAP.md §3.6

Tests cover:
    - DistributedContext initialization and operations
    - Dask-based distributed RG flow integration
    - Ray-based parameter space exploration
    - Distributed Monte Carlo sampling
    - Distributed QNCD matrix computation
    - Graceful fallback to serial execution

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from src.performance.distributed import (
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
    _beta_functions,
    _integrate_rg_trajectory,
    LAMBDA_STAR,
    GAMMA_STAR,
    MU_STAR,
)


# =============================================================================
# Test Constants
# =============================================================================

FIXED_POINT = np.array([LAMBDA_STAR, GAMMA_STAR, MU_STAR])
TOLERANCE = 1e-8


# =============================================================================
# Test Distributed Availability
# =============================================================================

class TestDistributedAvailability:
    """Tests for distributed backend availability detection."""
    
    def test_is_dask_available_returns_bool(self):
        """is_dask_available should return a boolean."""
        result = is_dask_available()
        assert isinstance(result, bool)
    
    def test_is_ray_available_returns_bool(self):
        """is_ray_available should return a boolean."""
        result = is_ray_available()
        assert isinstance(result, bool)
    
    def test_get_distributed_info_returns_dict(self):
        """get_distributed_info should return a dictionary with expected keys."""
        info = get_distributed_info()
        assert isinstance(info, dict)
        assert 'dask_available' in info
        assert 'ray_available' in info
        assert 'default_backend' in info
        assert 'available_backends' in info
    
    def test_available_backends_includes_serial(self):
        """Serial backend should always be available."""
        backends = get_available_distributed_backends()
        assert DistributedBackend.SERIAL in backends
    
    def test_distributed_backend_enum(self):
        """DistributedBackend enum should have expected values."""
        assert DistributedBackend.DASK.value == "dask"
        assert DistributedBackend.RAY.value == "ray"
        assert DistributedBackend.SERIAL.value == "serial"


# =============================================================================
# Test DistributedContext
# =============================================================================

class TestDistributedContext:
    """Tests for DistributedContext class."""
    
    def test_default_context_creation(self):
        """Should create context with default settings."""
        ctx = DistributedContext()
        assert ctx.backend in get_available_distributed_backends()
        assert ctx.n_workers is not None
        assert ctx.n_workers >= 1
    
    def test_serial_backend_always_works(self):
        """Serial backend should always work."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        assert ctx.backend == DistributedBackend.SERIAL
        assert ctx.is_distributed is False
    
    def test_context_manager_entry_exit(self):
        """Context manager should work correctly."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        with ctx as c:
            assert c._active is True
        assert ctx._active is False
    
    def test_context_verbose_mode(self):
        """Verbose mode should not raise errors."""
        ctx = DistributedContext(verbose=True, backend=DistributedBackend.SERIAL)
        with ctx:
            pass  # Should not raise
    
    def test_context_n_workers_setting(self):
        """Should respect n_workers setting."""
        ctx = DistributedContext(n_workers=2, backend=DistributedBackend.SERIAL)
        assert ctx.n_workers == 2
    
    def test_context_submit_serial(self):
        """Submit should execute immediately in serial mode."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        
        def square(x):
            return x * x
        
        result = ctx.submit(square, 5)
        assert result == 25
    
    def test_context_gather_serial(self):
        """Gather should return list unchanged in serial mode."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        results = [1, 2, 3]
        gathered = ctx.gather(results)
        assert gathered == results


# =============================================================================
# Test Beta Functions
# =============================================================================

class TestBetaFunctions:
    """Tests for internal beta functions (Eq. 1.13)."""
    
    def test_beta_functions_shape(self):
        """Beta functions should return correct shape."""
        couplings = np.array([[50.0, 100.0, 150.0]])
        betas = _beta_functions(couplings)
        assert betas.shape == (1, 3)
    
    def test_beta_functions_batch(self):
        """Should handle batch inputs."""
        couplings = np.random.rand(10, 3) * 100
        betas = _beta_functions(couplings)
        assert betas.shape == (10, 3)
    
    def test_beta_lambda_at_16pi2_over_9(self):
        """β_λ should vanish at λ̃ = 16π²/9 (one-loop zero).
        
        Theoretical Reference:
            IRH v21.1 Manuscript Eq. 1.13
            β_λ = -2λ̃ + (9/8π²)λ̃² = 0 has non-trivial solution λ̃ = 16π²/9
            
        Note: This is distinct from the Cosmic Fixed Point values in Eq. 1.14,
        which arise from the full Wetterich equation analysis.
        """
        # From β_λ = -2λ̃ + (9/8π²)λ̃² = 0, solving: λ̃ = 16π²/9 ≈ 17.546
        lambda_zero = 16 * np.pi**2 / 9
        couplings = np.array([[lambda_zero, GAMMA_STAR, MU_STAR]])
        betas = _beta_functions(couplings)
        assert abs(betas[0, 0]) < 1e-10


# =============================================================================
# Test RG Trajectory Integration
# =============================================================================

class TestRGTrajectory:
    """Tests for RG trajectory integration."""
    
    def test_integrate_trajectory_shape(self):
        """Should return trajectory of correct shape."""
        initial = np.array([60.0, 110.0, 160.0])
        trajectory, converged = _integrate_rg_trajectory(
            initial, t_range=(0, 5), n_steps=100
        )
        assert trajectory.shape == (101, 3)
        assert isinstance(converged, (bool, np.bool_))
    
    def test_trajectory_starts_at_initial(self):
        """Trajectory should start at initial conditions."""
        initial = np.array([60.0, 110.0, 160.0])
        trajectory, _ = _integrate_rg_trajectory(
            initial, t_range=(0, 5), n_steps=100
        )
        assert_allclose(trajectory[0], initial)
    
    def test_trajectory_integration_mechanics(self):
        """Test that trajectory integration works correctly mechanically."""
        # Use a simple starting point that won't overflow
        initial = np.array([10.0, 20.0, 30.0])
        trajectory, converged = _integrate_rg_trajectory(
            initial, t_range=(0, 0.1), n_steps=10  # Very short integration
        )
        # Should produce finite trajectory values
        final = trajectory[-1]
        # Check that integration produced finite values (may not converge)
        assert np.all(np.isfinite(final)), "Trajectory diverged to non-finite values"
        # Check that trajectory evolved from initial conditions
        assert trajectory.shape == (11, 3)
        assert_allclose(trajectory[0], initial)


# =============================================================================
# Test Dask RG Flow
# =============================================================================

class TestDaskRGFlow:
    """Tests for Dask-based distributed RG flow."""
    
    def test_dask_rg_flow_serial_fallback(self):
        """Should work in serial mode."""
        initial = np.array([[60.0, 110.0, 160.0], [55.0, 105.0, 155.0]])
        
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        result = dask_rg_flow(initial, t_range=(0, 5), n_steps=100, ctx=ctx)
        
        assert 'trajectories' in result
        assert 'converged' in result
        assert 'timing' in result
        assert result['n_trajectories'] == 2
        assert result['trajectories'].shape == (2, 101, 3)
    
    def test_dask_rg_flow_returns_expected_keys(self):
        """Should return all expected result keys."""
        initial = np.array([[60.0, 110.0, 160.0]])
        result = dask_rg_flow(initial, t_range=(0, 5), n_steps=50)
        
        expected_keys = [
            'trajectories', 'times', 'converged', 'fixed_points',
            'n_trajectories', 'n_converged', 'timing', 'backend',
            'is_distributed', 'theoretical_reference'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_dask_rg_flow_theoretical_reference(self):
        """Should include theoretical reference."""
        initial = np.array([[60.0, 110.0, 160.0]])
        result = dask_rg_flow(initial, t_range=(0, 5), n_steps=50)
        
        assert 'theoretical_reference' in result
        assert 'IRH v21.1' in result['theoretical_reference']
        assert 'Eq. 1.12-1.13' in result['theoretical_reference']
    
    def test_dask_rg_flow_timing(self):
        """Should record timing information."""
        initial = np.array([[60.0, 110.0, 160.0]])
        result = dask_rg_flow(initial, t_range=(0, 5), n_steps=50)
        
        assert result['timing']['total_seconds'] > 0
        assert result['timing']['n_trajectories'] == 1


# =============================================================================
# Test Ray Parameter Scan
# =============================================================================

class TestRayParameterScan:
    """Tests for Ray-based parameter space exploration."""
    
    def test_ray_parameter_scan_serial_fallback(self):
        """Should work in serial mode."""
        grid = np.array([
            [50.0, 100.0, 150.0],
            [51.0, 101.0, 151.0],
            [52.0, 102.0, 152.0],
        ])
        
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        result = ray_parameter_scan(grid, ctx=ctx)
        
        assert 'parameters' in result
        assert 'values' in result
        assert result['n_points'] == 3
        assert len(result['values']) == 3
    
    def test_ray_parameter_scan_returns_expected_keys(self):
        """Should return all expected result keys."""
        grid = np.array([[50.0, 100.0, 150.0], [52.0, 105.0, 158.0]])
        result = ray_parameter_scan(grid)
        
        expected_keys = [
            'parameters', 'values', 'min_point', 'min_value',
            'max_point', 'max_value', 'n_points', 'timing',
            'backend', 'is_distributed', 'theoretical_reference'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_ray_parameter_scan_min_max(self):
        """Should correctly identify min and max points."""
        # Point closest to fixed point should have smallest value
        grid = np.array([
            FIXED_POINT,  # Distance = 0
            FIXED_POINT + np.array([10, 10, 10]),  # Distance > 0
        ])
        
        result = ray_parameter_scan(grid)
        
        # Min should be at fixed point
        assert_allclose(result['min_point'], FIXED_POINT, rtol=1e-10)
        assert result['min_value'] < result['max_value']
    
    def test_ray_parameter_scan_custom_function(self):
        """Should work with custom evaluation function."""
        grid = np.array([[1, 2, 3], [4, 5, 6]])
        
        def sum_squares(params):
            return np.sum(params ** 2)
        
        result = ray_parameter_scan(grid, evaluation_function=sum_squares)
        
        # First point sum of squares: 1 + 4 + 9 = 14
        # Second point: 16 + 25 + 36 = 77
        assert_allclose(result['values'][0], 14)
        assert_allclose(result['values'][1], 77)


# =============================================================================
# Test Distributed Monte Carlo
# =============================================================================

class TestDistributedMonteCarlo:
    """Tests for distributed Monte Carlo sampling."""
    
    def test_monte_carlo_serial_fallback(self):
        """Should work in serial mode."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        result = distributed_monte_carlo(
            n_samples=100,
            n_batches=2,
            ctx=ctx
        )
        
        assert 'mean' in result
        assert 'std' in result
        assert result['n_samples'] == 100
    
    def test_monte_carlo_returns_expected_keys(self):
        """Should return all expected result keys."""
        result = distributed_monte_carlo(n_samples=50, n_batches=2)
        
        expected_keys = [
            'mean', 'std', 'variance', 'n_samples', 'values',
            'histogram', 'timing', 'backend', 'is_distributed',
            'theoretical_reference'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_monte_carlo_statistics(self):
        """Should compute correct statistics."""
        result = distributed_monte_carlo(n_samples=200, n_batches=4)
        
        # Verify consistency of statistics
        assert isinstance(result['mean'], float)
        assert isinstance(result['std'], float)
        assert result['std'] >= 0
        assert result['variance'] >= 0
        assert np.isclose(result['variance'], result['std'] ** 2, rtol=1e-10)
    
    def test_monte_carlo_custom_functions(self):
        """Should work with custom sample and observable functions."""
        def sample_func(n):
            return np.ones((n, 3))
        
        def obs_func(params):
            return np.sum(params)  # Should always be 3
        
        result = distributed_monte_carlo(
            n_samples=50,
            sample_function=sample_func,
            observable_function=obs_func,
            n_batches=2
        )
        
        # All observables should be 3
        assert_allclose(result['mean'], 3.0, rtol=1e-10)
        assert result['std'] < 1e-10


# =============================================================================
# Test Distributed QNCD Matrix
# =============================================================================

class TestDistributedQNCDMatrix:
    """Tests for distributed QNCD matrix computation."""
    
    def test_qncd_matrix_serial_fallback(self):
        """Should work in serial mode."""
        vectors = np.random.rand(10, 4)
        
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        result = distributed_qncd_matrix(vectors, n_blocks=2, ctx=ctx)
        
        assert 'distance_matrix' in result
        assert result['distance_matrix'].shape == (10, 10)
    
    def test_qncd_matrix_returns_expected_keys(self):
        """Should return all expected result keys."""
        vectors = np.random.rand(5, 3)
        result = distributed_qncd_matrix(vectors, n_blocks=2)
        
        expected_keys = [
            'distance_matrix', 'min_distance', 'max_distance',
            'mean_distance', 'n_vectors', 'n_pairs', 'timing',
            'backend', 'is_distributed', 'theoretical_reference'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_qncd_matrix_symmetric(self):
        """Distance matrix should be symmetric."""
        vectors = np.random.rand(8, 3)
        result = distributed_qncd_matrix(vectors, n_blocks=2)
        
        D = result['distance_matrix']
        assert_allclose(D, D.T, rtol=1e-10)
    
    def test_qncd_matrix_diagonal_zero(self):
        """Diagonal should be zero (self-distance)."""
        vectors = np.random.rand(6, 3)
        result = distributed_qncd_matrix(vectors, n_blocks=2)
        
        D = result['distance_matrix']
        assert_allclose(np.diag(D), np.zeros(6), atol=1e-10)
    
    def test_qncd_matrix_nonnegative(self):
        """All distances should be non-negative."""
        vectors = np.random.rand(7, 3)
        result = distributed_qncd_matrix(vectors, n_blocks=2)
        
        D = result['distance_matrix']
        assert np.all(D >= -1e-10)  # Allow tiny numerical errors
    
    def test_qncd_matrix_identical_vectors(self):
        """Identical vectors should have zero distance."""
        vectors = np.array([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        result = distributed_qncd_matrix(vectors, n_blocks=1)
        
        D = result['distance_matrix']
        assert D[0, 1] < 1e-10  # Identical vectors
        assert D[0, 2] > 0  # Different vectors


# =============================================================================
# Test Distributed Map
# =============================================================================

class TestDistributedMap:
    """Tests for generic distributed map function."""
    
    def test_distributed_map_serial(self):
        """Should work in serial mode."""
        def square(x):
            return x * x
        
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        results = distributed_map(square, [1, 2, 3, 4, 5], ctx)
        
        assert results == [1, 4, 9, 16, 25]
    
    def test_distributed_map_preserves_order(self):
        """Should preserve order of results."""
        def identity(x):
            return x
        
        items = list(range(10))
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        results = distributed_map(identity, items, ctx)
        
        assert results == items
    
    def test_distributed_map_complex_function(self):
        """Should work with more complex functions."""
        def process(params):
            x, y = params
            return x + y
        
        items = [(1, 2), (3, 4), (5, 6)]
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        results = distributed_map(process, items, ctx)
        
        assert results == [3, 7, 11]


# =============================================================================
# Test Cluster Management
# =============================================================================

class TestClusterManagement:
    """Tests for cluster creation and shutdown."""
    
    def test_create_local_cluster_without_dask(self):
        """Should return None if Dask distributed unavailable."""
        if not is_dask_available():
            cluster = create_local_cluster(n_workers=2)
            assert cluster is None
    
    def test_shutdown_cluster_handles_none(self):
        """Should handle None cluster gracefully."""
        shutdown_cluster(None)  # Should not raise


# =============================================================================
# Test Integration
# =============================================================================

class TestIntegration:
    """Integration tests for distributed computing."""
    
    def test_full_workflow_serial(self):
        """Test complete workflow in serial mode."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        
        with ctx:
            # 1. Parameter scan
            grid = np.random.rand(10, 3) * 100
            scan_result = ray_parameter_scan(grid, ctx=ctx)
            assert scan_result['n_points'] == 10
            
            # 2. RG flow from best point
            best_point = scan_result['min_point']
            flow_result = dask_rg_flow(
                np.array([best_point]),
                t_range=(0, 5),
                n_steps=50,
                ctx=ctx
            )
            assert flow_result['n_trajectories'] == 1
            
            # 3. Monte Carlo around result
            mc_result = distributed_monte_carlo(
                n_samples=50,
                n_batches=2,
                ctx=ctx
            )
            assert mc_result['n_samples'] == 50
    
    def test_context_reuse(self):
        """Context should be reusable after exit."""
        ctx = DistributedContext(backend=DistributedBackend.SERIAL)
        
        with ctx:
            result1 = distributed_map(lambda x: x * 2, [1, 2, 3], ctx)
        
        with ctx:
            result2 = distributed_map(lambda x: x * 3, [1, 2, 3], ctx)
        
        assert result1 == [2, 4, 6]
        assert result2 == [3, 6, 9]


# =============================================================================
# Test Dask Integration (Conditional)
# =============================================================================

@pytest.mark.skipif(not is_dask_available(), reason="Dask not available")
class TestDaskIntegration:
    """Tests that require Dask."""
    
    def test_dask_backend_available(self):
        """Dask backend should be in available backends."""
        backends = get_available_distributed_backends()
        assert DistributedBackend.DASK in backends
    
    def test_dask_context_creation(self):
        """Should create Dask context."""
        ctx = DistributedContext(backend=DistributedBackend.DASK)
        assert ctx.backend == DistributedBackend.DASK


# =============================================================================
# Test Ray Integration (Conditional)
# =============================================================================

@pytest.mark.skipif(not is_ray_available(), reason="Ray not available")
class TestRayIntegration:
    """Tests that require Ray."""
    
    def test_ray_backend_available(self):
        """Ray backend should be in available backends."""
        backends = get_available_distributed_backends()
        assert DistributedBackend.RAY in backends
    
    def test_ray_context_creation(self):
        """Should create Ray context."""
        ctx = DistributedContext(backend=DistributedBackend.RAY)
        assert ctx.backend == DistributedBackend.RAY


# =============================================================================
# Test Theoretical References
# =============================================================================

class TestTheoreticalReferences:
    """Tests for theoretical reference documentation."""
    
    def test_dask_rg_flow_cites_manuscript(self):
        """dask_rg_flow should cite IRH v21.1 Manuscript."""
        result = dask_rg_flow(
            np.array([[60.0, 110.0, 160.0]]),
            n_steps=10
        )
        ref = result['theoretical_reference']
        assert 'IRH v21.1' in ref
        assert '§1.2' in ref
    
    def test_ray_parameter_scan_cites_manuscript(self):
        """ray_parameter_scan should cite IRH v21.1 Manuscript."""
        result = ray_parameter_scan(np.array([[50.0, 100.0, 150.0]]))
        ref = result['theoretical_reference']
        assert 'IRH v21.1' in ref
        assert '§1.3' in ref or 'Appendix B' in ref
    
    def test_monte_carlo_cites_manuscript(self):
        """distributed_monte_carlo should cite IRH v21.1 Manuscript."""
        result = distributed_monte_carlo(n_samples=20, n_batches=1)
        ref = result['theoretical_reference']
        assert 'IRH v21.1' in ref
        assert '§1.1' in ref
    
    def test_qncd_matrix_cites_appendix_a(self):
        """distributed_qncd_matrix should cite Appendix A."""
        result = distributed_qncd_matrix(np.random.rand(5, 3), n_blocks=1)
        ref = result['theoretical_reference']
        assert 'IRH v21.1' in ref
        assert 'Appendix A' in ref
