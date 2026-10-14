"""The released names that break the package's naming rules, renamed with no alias.

``input`` is ``system`` on the system-taking functions and ``params`` on
``BinaryLiquid.update_params``. Passing ``input=`` raises ``TypeError`` naming the new name,
before any store is read.
"""

import inspect

import pytest

from gliquid import api, mpds, phase
from gliquid.binary import BinaryLiquid


def _first_parameter(func):
    names = [name for name in inspect.signature(func).parameters if name not in ("self", "cls")]
    return names[0]


# --------------------------------------------------------------------------- signatures
@pytest.mark.parametrize(
    "func, expected",
    [
        pytest.param(api.get_dft_convexhull, "system", id="api.get_dft_convexhull"),
        pytest.param(api.get_dft_structure_entries, "system", id="api.get_dft_structure_entries"),
        pytest.param(api.cache_imputed_entries, "system", id="api.cache_imputed_entries"),
        pytest.param(api.read_imputed_entries, "system", id="api.read_imputed_entries"),
        pytest.param(api.drop_imputed_entries, "system", id="api.drop_imputed_entries"),
        pytest.param(BinaryLiquid.from_cache, "system", id="BinaryLiquid.from_cache"),
        pytest.param(BinaryLiquid.update_params, "params", id="BinaryLiquid.update_params"),
        pytest.param(mpds.load_mpds_data, "system", id="mpds.load_mpds_data"),
        pytest.param(
            phase.validate_and_format_system, "system", id="phase.validate_and_format_system"
        ),
    ],
)
def test_first_parameter_has_the_new_name(func, expected):
    assert _first_parameter(func) == expected


# --------------------------------------------------------------------------- input= keyword
@pytest.mark.parametrize(
    "call, new",
    [
        pytest.param(api.get_dft_convexhull, "system", id="api.get_dft_convexhull"),
        pytest.param(api.get_dft_structure_entries, "system", id="api.get_dft_structure_entries"),
        pytest.param(
            lambda **kw: api.cache_imputed_entries(imputed_entry_dicts=[], **kw),
            "system",
            id="api.cache_imputed_entries",
        ),
        pytest.param(BinaryLiquid.from_cache, "system", id="BinaryLiquid.from_cache"),
        pytest.param(mpds.load_mpds_data, "system", id="mpds.load_mpds_data"),
        pytest.param(
            phase.validate_and_format_system, "system", id="phase.validate_and_format_system"
        ),
        pytest.param(
            lambda **kw: BinaryLiquid.update_params(None, **kw),
            "params",
            id="BinaryLiquid.update_params",
        ),
    ],
)
def test_released_input_keyword_raises_naming_the_new_name(call, new):
    with pytest.raises(TypeError, match=rf"(?s)(?=.*\binput\b)(?=.*\b{new}\b)"):
        call(input="Cu-Mg")


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(api.read_imputed_entries, id="api.read_imputed_entries"),
        pytest.param(api.drop_imputed_entries, id="api.drop_imputed_entries"),
    ],
)
def test_unreleased_functions_reject_the_input_keyword(func):
    with pytest.raises(TypeError, match="input"):
        func(input="Xx-Qq")
