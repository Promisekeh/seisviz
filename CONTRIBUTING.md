# Contributing to seisviz

Pull requests and issues are welcome — especially around interactivity,
mask/horizon overlays, or ML-facing features (see [ROADMAP.md](ROADMAP.md)).

## Setup

```bash
git clone https://github.com/Promisekeh/seisviz.git
cd seisviz
pip install -e ".[dev]"
```

`segyio` (needed for SEG-Y support) is included in the `dev` extra so the
full test suite can run; it's an optional runtime dependency otherwise (see
`[segy]` in `pyproject.toml`).

## Running the tests

```bash
pytest
```

The suite runs headless (`matplotlib.use("Agg")`, set in `tests/conftest.py`)
and needs no real SEG-Y file or external dataset.

## Before opening a PR

- Add a test for any bug fix or new behavior. `tests/conftest.py`'s `cube`
  fixture uses three distinct axis lengths specifically so axis-order bugs
  don't slip through silently — reuse it rather than a cube with equal
  dimensions.
- Run `pytest -q` and confirm it's green.
- If you're changing plotting output, sanity-check a rendered figure by eye
  (`fig.savefig(...)` and open it) — the test suite checks shapes, labels,
  and color limits, not that a plot looks right.
- Keep the axis convention (`inline, xline, depth`) consistent across any
  function you touch; `test_axis_convention.py` is the single source of
  truth for what that means in practice.

## Reporting a bug

Include the `seisviz`/matplotlib/Python versions, a minimal repro, and — if
it's a rendering issue — the figure itself. A screenshot together with the
exact call that produced it (as in [#2](https://github.com/Promisekeh/seisviz/issues/2))
is usually enough to diagnose without back-and-forth.
