"""
Tests for SQDDiagnostics.

Unit tests construct real SCIResult/SCIState objects (not loose mocks) so
that a future qiskit-addon-sqd API change would break these tests rather
than silently passing against a stale mock shape.

The integration test runs the actual diagonalize_fermionic_hamiltonian
workflow against a real (tiny) H2 molecule, to confirm the callback wiring
works against the genuine library, not just our assumptions about its
output shape.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit_addon_sqd.fermion import SCIResult, SCIState

from qiskit_sqd_dashboard import SQDDiagnostics


def _make_sci_result(energy: float, occ_a: np.ndarray, occ_b: np.ndarray) -> SCIResult:
    """Build a real (not mocked) SCIResult with a tiny 2x2 subspace."""
    ci_strs_a = np.array([0b01, 0b10])
    ci_strs_b = np.array([0b01, 0b10])
    amplitudes = np.array([[1.0, 0.0], [0.0, 0.0]])
    sci_state = SCIState(
        amplitudes=amplitudes, ci_strs_a=ci_strs_a, ci_strs_b=ci_strs_b, norb=2, nelec=(1, 1)
    )
    return SCIResult(
        energy=energy,
        sci_state=sci_state,
        orbital_occupancies=(occ_a, occ_b),
    )


class TestCallback:
    def test_first_call_has_no_occupancy_delta(self):
        diag = SQDDiagnostics()
        result = _make_sci_result(-1.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        diag.callback([result])

        assert diag.iterations == [0]
        assert diag.best_energy == [-1.0]
        assert diag.occupancy_deltas == [None]

    def test_second_call_computes_occupancy_delta(self):
        diag = SQDDiagnostics()
        r1 = _make_sci_result(-1.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        r2 = _make_sci_result(-1.5, np.array([0.9, 0.1]), np.array([1.0, 0.0]))
        diag.callback([r1])
        diag.callback([r2])

        assert diag.iterations == [0, 1]
        assert diag.occupancy_deltas[0] is None
        # occ changed by 0.1 in one component, 0 in the others -> max abs change = 0.1
        assert diag.occupancy_deltas[1] == pytest.approx(0.1)

    def test_best_energy_picks_minimum_across_batches(self):
        diag = SQDDiagnostics()
        r_worse = _make_sci_result(-1.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        r_better = _make_sci_result(-2.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        diag.callback([r_worse, r_better])

        assert diag.best_energy == [-2.0]
        assert diag.energy_min == [-2.0]
        assert diag.energy_max == [-1.0]

    def test_subspace_dim_matches_ci_strs_product(self):
        diag = SQDDiagnostics()
        result = _make_sci_result(-1.0, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        diag.callback([result])

        # ci_strs_a and ci_strs_b each have length 2 in our fixture -> dim = 4
        assert diag.subspace_dims == [[4]]

    def test_empty_results_raises(self):
        diag = SQDDiagnostics()
        with pytest.raises(ValueError, match="empty"):
            diag.callback([])


class TestSummary:
    def test_summary_before_any_callback(self):
        diag = SQDDiagnostics()
        summary = diag.summary()
        assert "0 iterations" in summary

    def test_summary_after_callback(self):
        diag = SQDDiagnostics()
        result = _make_sci_result(-1.23456789, np.array([1.0, 0.0]), np.array([1.0, 0.0]))
        diag.callback([result])
        summary = diag.summary()
        assert "1 iterations" in summary
        assert "-1.23456789" in summary


class TestRealSQDIntegration:
    """
    Runs the actual qiskit_addon_sqd workflow end-to-end against a tiny real
    H2 molecule. This is the test that would catch a breaking API change in
    qiskit-addon-sqd itself (e.g. SCIResult gaining/losing a field).
    """

    def test_callback_wiring_against_real_diagonalize_fermionic_hamiltonian(self):
        pyscf = pytest.importorskip("pyscf")
        from pyscf import ao2mo, gto, scf
        from qiskit.primitives import BitArray
        from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

        mol = gto.M(atom="H 0 0 0; H 0 0 0.735", basis="sto-3g")
        mf = scf.RHF(mol).run()
        norb = mf.mo_coeff.shape[1]
        nelec = mol.nelec

        hcore = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
        eri = ao2mo.restore(1, ao2mo.full(mol, mf.mo_coeff), norb)

        rng = np.random.default_rng(0)
        good_configs = ["0110", "0101", "1001", "1010"]
        samples = [rng.choice(good_configs) for _ in range(200)]
        bit_array = BitArray.from_samples(samples, num_bits=2 * norb)

        diag = SQDDiagnostics()
        result = diagonalize_fermionic_hamiltonian(
            hcore, eri, bit_array,
            samples_per_batch=50, norb=norb, nelec=nelec,
            num_batches=2, max_iterations=3,
            callback=diag.callback, seed=0,
        )

        assert len(diag.iterations) >= 1
        assert diag.best_energy[-1] == pytest.approx(result.energy, abs=1e-9)
        # HF energy for H2/STO-3G is about -1.117 Ha electronic-only value
        # differs (this is the active-space electronic energy, no nuclear
        # repulsion added), but it should be a real negative number in a
        # sane range, not NaN/zero/wildly off from physical expectations.
        assert -3.0 < diag.best_energy[-1] < 0.0
