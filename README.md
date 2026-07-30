# qiskit-sqd-dashboard

Live convergence diagnostics for [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd)'s self-consistent configuration recovery loop, rendered as an in-notebook Plotly widget.

`qiskit-addon-sqd` is a pure compute library with no visualization layer — this package fills that gap without modifying or wrapping the addon itself.

## Why a notebook widget, not a standalone app

SQD workflows are already Jupyter-native: the whole configuration-recovery loop is a Python call inside a notebook cell. This package hooks directly into that loop via the `callback` parameter that `diagonalize_fermionic_hamiltonian` already exposes — no separate server, no data hand-off between processes.

## Install

```bash
pip install -e .
```

No extra system dependencies needed beyond the packages above — earlier versions of this package used Plotly's `FigureWidget` (which requires `anywidget`), but that approach was dropped in favor of a `clear_output()` + redraw pattern after discovering `FigureWidget` is currently broken in Google Colab ([plotly/plotly.py#5027](https://github.com/plotly/plotly.py/issues/5027)). See the note in `diagnostics.py` for details.

## Usage

```python
from qiskit_sqd_dashboard import SQDDiagnostics
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian

diag = SQDDiagnostics()
diag.display()  # renders the live figure in the notebook cell

result = diagonalize_fermionic_hamiltonian(
    hcore, eri, bit_array,
    samples_per_batch=samples_per_batch,
    norb=norb, nelec=nelec,
    callback=diag.callback,   # <-- this is the whole integration
)

print(diag.summary())
```

That's it — `diag.callback` matches `diagonalize_fermionic_hamiltonian`'s own `callback: Callable[[list[SCIResult]], None]` signature exactly, so there's no adapter code to write.

See `sqd_dashboard_colab.ipynb` for a complete, self-contained, Colab-ready example against a real N2 active-space problem (LUCJ ansatz from CCSD amplitudes, via `ffsim`, with realistic hardware noise injected). Everything needed is inlined in the notebook — no separate package install required to try it.

## What it tracks, per iteration

- **Energy convergence** — best energy found (min across batches), plus min/max spread across batches
- **Subspace dimension per batch** — `len(ci_strs_a) * len(ci_strs_b)` for each batch's `SCIResult`
- **Orbital occupancy convergence** — max absolute change in average orbital occupancies vs. the previous iteration (this is the quantity configuration recovery uses internally to correct noisy samples, so watching it flatten is a direct convergence signal)

## Project layout

```
qiskit_sqd_dashboard/
└── diagnostics.py            # SQDDiagnostics: the callback + live Plotly redraw logic
tests/
└── test_diagnostics.py       # unit tests (real SCIResult/SCIState objects) + one real-SQD integration test
.github/workflows/tests.yml   # CI: runs the test suite on Python 3.10/3.11/3.12
sqd_dashboard_colab.ipynb     # self-contained, executed, working example against a real N2 active-space run
```

## Development

```bash
pip install -e ".[test]"
pytest tests/ -v
```

The test suite includes a real integration test that runs the actual `qiskit_addon_sqd.fermion.diagonalize_fermionic_hamiltonian` workflow end-to-end against a tiny H2 molecule — not just mocked unit tests — so a breaking change in `qiskit-addon-sqd`'s own API would be caught here.

To build distribution artifacts (sdist + wheel):
```bash
pip install build twine
python -m build
twine check dist/*
```

## Current limitations (v1)

- **Validated against a real N2 active-space run** (8 orbitals, 10 electrons, STO-3G, LUCJ ansatz built from CCSD amplitudes via `ffsim`, with realistic 2% per-bit hardware noise injected) — this showed genuine multi-iteration convergence dynamics (4-6 iterations, energy improving monotonically, subspace dimension growing from ~380 to ~700, occupancy deltas rising then falling toward zero). See `sqd_dashboard_colab.ipynb`. Not yet validated at the full scale of IBM's own N2/6-31g tutorial (59 qubits) — this was a smaller but still genuinely correlated multi-orbital system, not a 2-orbital toy.
- No persistence: diagnostics live only for the notebook session. A "save iteration history to disk" option would be needed to support the planned Streamlit comparison-across-runs view.
- Single-run view only — comparing multiple runs (e.g. different `samples_per_batch` settings) side by side is out of scope for v1; see Roadmap.
- Not yet published to PyPI — `python -m build` + `twine check` both pass, so the package builds cleanly and could be published, but this hasn't been done yet (`pip install -e .` from source is the only install path today).

## Roadmap

1. Validate against the full-scale N2/6-31g tutorial workflow (59 qubits)
2. Optional history persistence (save/load iteration logs)
3. Secondary Streamlit app for comparing multiple saved runs side by side

## License

Apache 2.0 (recommended for Qiskit Ecosystem submission).
