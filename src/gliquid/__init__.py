"""gliquid — thermodynamic modeling of binary/ternary liquid phases from DFT + experimental data.

MIT License

Copyright (c) 2025 Joshua Willwerth

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from importlib import import_module, metadata

# Public API façade (PEP 562): names resolve lazily so `import gliquid` stays light —
# no matplotlib/plotly/pymatgen import until the consuming submodule is actually needed.
# Changing this mapping changes the public surface and requires explicit sign-off.
_EXPORTS = {
    "BinaryLiquid": "gliquid.binary",
    "BLPlotter": "gliquid.binary",
    "TernaryLiquidInterpolation": "gliquid.ternary",
    "TLIPlotter": "gliquid.ternary",
    "HSX": "gliquid.hsx",
    "lower_convex_hull": "gliquid.hsx",
    "ConvexHullEditor": "gliquid.hull_editor",
    "SolutionModel": "gliquid.solution",
    "RKPolyExp": "gliquid.solution",
    "UNARY": "gliquid.phase",
    "ComponentRef": "gliquid.phase",
    "Phase": "gliquid.phase",
    "identify_invariant_points": "gliquid.mpds",
    "load_mpds_data": "gliquid.mpds",
    # Added by spec 07 (dev/plans/gliquid-cache-layout/07-portable-ml-bundle.md), which is
    # the recorded sign-off this mapping asks for. Additive: the pickle-free bundle now
    # ships inside the wheel at gliquid/models/, so the runner is usable straight from
    # `import gliquid` with no bundle path to find.
    "ProductionModelRunner": "gliquid.production_model_runner",
    "set_cache_dir": "gliquid.config",
    "set_data_dir": "gliquid.config",  # deprecated alias of set_cache_dir; still public
    "ConfigError": "gliquid.config",
}

__all__ = [*_EXPORTS, "__version__"]

# Single-sourced from the git tag. Installed distributions carry it in their metadata; a
# built-but-uninstalled tree carries it in _version.py, written by the hatch-vcs build hook.
# A source tree with neither is genuinely unversioned, and says so rather than asserting a
# stale number -- no version literal belongs in this package.
try:
    __version__ = metadata.version("gliquid")
except metadata.PackageNotFoundError:
    try:
        from gliquid._version import version as __version__
    except ImportError:
        __version__ = "0.0.0+unknown"


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'gliquid' has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *_EXPORTS})
