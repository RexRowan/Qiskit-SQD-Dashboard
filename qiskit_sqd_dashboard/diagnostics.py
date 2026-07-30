"""
Live diagnostics dashboard for qiskit-addon-sqd's self-consistent
configuration recovery loop.

Usage (inside a Jupyter/JupyterLab/Colab notebook):

    from qiskit_sqd_dashboard import SQDDiagnostics
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

    diag = SQDDiagnostics()
    diag.display()  # renders the live figure; subsequent cells update it in place

    result = diagonalize_fermionic_hamiltonian(
        hcore, eri, bit_array, samples_per_batch=..., norb=norb, nelec=nelec,
        callback=diag.callback,
    )

The callback signature matches qiskit_addon_sqd.fermion's
`callback: Callable[[list[SCIResult]], None]` parameter exactly, so no
adapter code is needed on the user's side.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from qiskit_addon_sqd.fermion import SCIResult


class SQDDiagnostics:
    """
    Tracks per-iteration SQD diagnostics and renders them as a live-updating
    Plotly figure inside a notebook.

    Uses a clear_output()+redraw pattern rather than go.FigureWidget, since
    FigureWidget (which depends on anywidget as of Plotly 6.0+) is broken in
    Google Colab specifically (see plotly/plotly.py#5027) — this approach
    works reliably across Jupyter, JupyterLab, Colab, and VS Code notebooks.

    Tracked per iteration (aggregated across all batches in that iteration):
        - best energy found (minimum across batches)
        - energy spread across batches (min/max), to show batch variance
        - subspace dimension per batch (len(ci_strs_a) * len(ci_strs_b))
        - orbital occupancy convergence: max abs change vs previous iteration
    """

    def __init__(self) -> None:
        self.iterations: list[int] = []
        self.best_energy: list[float] = []
        self.energy_min: list[float] = []
        self.energy_max: list[float] = []
        self.subspace_dims: list[list[int]] = []  # one list of dims per iteration (per batch)
        self.occupancy_deltas: list[float | None] = []  # None on first iteration (nothing to compare to)

        self._prev_occupancies: np.ndarray | None = None
        self._displaying: bool = False

    def callback(self, results: Sequence[SCIResult]) -> None:
        """
        Matches qiskit_addon_sqd.fermion's callback signature:
        Callable[[list[SCIResult]], None]. Pass this method directly as the
        `callback=` argument to diagonalize_fermionic_hamiltonian.

        Raises:
            ValueError: if `results` is empty. diagonalize_fermionic_hamiltonian
                is not expected to call the callback with an empty list, but
                this fails loudly rather than silently corrupting internal
                state if it ever does (e.g. from a custom caller).
        """
        if not results:
            raise ValueError("SQDDiagnostics.callback received an empty results list.")

        iteration = len(self.iterations)
        energies = [r.energy for r in results]
        best_idx = int(np.argmin(energies))
        best_result = results[best_idx]

        dims = [
            len(r.sci_state.ci_strs_a) * len(r.sci_state.ci_strs_b) for r in results
        ]

        occ_a, occ_b = best_result.orbital_occupancies
        current_occ = np.concatenate([occ_a, occ_b])
        if self._prev_occupancies is None:
            occ_delta = None
        else:
            occ_delta = float(np.max(np.abs(current_occ - self._prev_occupancies)))
        self._prev_occupancies = current_occ

        self.iterations.append(iteration)
        self.best_energy.append(float(energies[best_idx]))
        self.energy_min.append(float(min(energies)))
        self.energy_max.append(float(max(energies)))
        self.subspace_dims.append(dims)
        self.occupancy_deltas.append(occ_delta)

        if self._displaying:
            self._redraw()

    def display(self) -> None:
        """
        Start live display. Call this before starting the SQD run; the
        figure will redraw itself after every callback invocation.
        """
        self._displaying = True
        self._redraw()

    def _build_figure(self):
        import plotly.graph_objects as go
        from plotly.graph_objects import Figure
        from plotly.subplots import make_subplots

        fig: Figure = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                "Energy convergence", "Subspace dimension per batch", "Max orbital-occupancy change"
            ),
            vertical_spacing=0.12,
        )
        fig.add_trace(
            go.Scatter(x=self.iterations, y=self.best_energy, mode="lines+markers", name="best energy"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.iterations, y=self.energy_min, mode="lines", name="energy min", line=dict(dash="dot")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=self.iterations, y=self.energy_max, mode="lines", name="energy max", line=dict(dash="dot")),
            row=1, col=1,
        )

        batch_x: list[int] = []
        batch_y: list[int] = []
        for it, dims in zip(self.iterations, self.subspace_dims):
            batch_x.extend([it] * len(dims))
            batch_y.extend(dims)
        fig.add_trace(
            go.Scatter(x=batch_x, y=batch_y, mode="markers", name="subspace dim (per batch)"),
            row=2, col=1,
        )

        occ_x = [it for it, d in zip(self.iterations, self.occupancy_deltas) if d is not None]
        occ_y = [d for d in self.occupancy_deltas if d is not None]
        fig.add_trace(
            go.Scatter(x=occ_x, y=occ_y, mode="lines+markers", name="max |occupancy change|"),
            row=3, col=1,
        )

        fig.update_layout(height=700, showlegend=True, margin=dict(t=60, b=40))
        fig.update_xaxes(title_text="iteration", row=3, col=1)
        fig.update_yaxes(title_text="energy (Ha)", row=1, col=1)
        fig.update_yaxes(title_text="dimension", row=2, col=1)
        fig.update_yaxes(title_text="\u0394 occupancy", row=3, col=1)
        return fig

    def _redraw(self) -> None:
        from IPython.display import clear_output
        clear_output(wait=True)
        self._build_figure().show()

    def summary(self) -> str:
        """A plain-text summary, useful outside a notebook or for logging."""
        lines = [f"SQD run: {len(self.iterations)} iterations"]
        if self.best_energy:
            lines.append(f"Final best energy: {self.best_energy[-1]:.8f} Ha")
            lines.append(f"Final subspace dims (per batch): {self.subspace_dims[-1]}")
            if self.occupancy_deltas[-1] is not None:
                lines.append(f"Final max |occupancy change|: {self.occupancy_deltas[-1]:.2e}")
        return "\n".join(lines)
