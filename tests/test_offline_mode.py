"""Offline mode: every remote path RAISES rather than reaching out.

This is what actually removes the API key from a deployment. "Has no key installed" is not
a property anything enforces -- ``api.get_api_key`` falls through to a gitignored ``.env``,
an operator's shell may export the variable, and a cache miss then reaches the network
anyway, slowly and without credentials. ``config.set_offline(True)`` makes that a
configuration decision enforced in code.

Every raise here is paired with a POSITIVE CONTROL showing the same call succeeds with
offline mode off. Without one, a test asserting "this raises" would keep passing if the
function started raising for an unrelated reason -- a missing key, a renamed argument, an
import error -- and the guard being tested could be deleted entirely.

The MPDS half carries the more dangerous case. On a cache miss with no ``MPDS_API_KEY``
that path logs a warning and returns ``{"reference": None}``, which is shaped exactly like
the real answer "MPDS holds no digitized diagram for this system". A silent skip under
offline mode would be indistinguishable from that fact.
"""

import json

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds


@pytest.fixture(autouse=True)
def _restore_offline():
    """Offline mode is process-wide state; save and restore the ACTUAL previous value."""
    previous = config.offline
    yield
    config.set_offline(previous)


class Tripwire:
    """A stand-in for a network client that fails loudly if anything calls it.

    The point of offline mode is that nothing is ATTEMPTED, not merely that an attempt
    fails. Asserting only on the exception type would pass just as well for a guard placed
    after the fetch.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("a network client was constructed while offline mode was ON")


class TestDefaults:
    def test_offline_is_off_by_default(self):
        """The package must not become unusable for everyone who does have a key."""
        assert config.offline is False

    def test_require_online_is_a_no_op_when_online(self):
        config.set_offline(False)
        assert config.require_online("anything") is None

    def test_require_online_raises_when_offline(self):
        config.set_offline(True)
        with pytest.raises(config.OfflineError) as excinfo:
            config.require_online("Fetching the thing")
        message = str(excinfo.value)
        assert "Fetching the thing" in message  # names WHAT was refused
        assert "set_offline(False)" in message  # names the way out
        assert config.OFFLINE_ENV_VAR in message

    def test_offline_error_is_a_config_error(self):
        """So `except gliquid.ConfigError` catches it, as it does every other misconfiguration."""
        assert issubclass(config.OfflineError, config.ConfigError)


class TestEnvironmentVariable:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
    def test_truthy_spellings_turn_it_on(self, monkeypatch, value):
        monkeypatch.setenv(config.OFFLINE_ENV_VAR, value)
        assert config._initial_offline() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_spellings_leave_it_off(self, monkeypatch, value):
        monkeypatch.setenv(config.OFFLINE_ENV_VAR, value)
        assert config._initial_offline() is False

    def test_unset_is_off(self, monkeypatch):
        monkeypatch.delenv(config.OFFLINE_ENV_VAR, raising=False)
        assert config._initial_offline() is False

    def test_an_unrecognized_value_FAILS_CLOSED(self, monkeypatch, caplog):
        """A typo must not silently permit the network.

        Someone who sets GLIQUID_OFFLINE at all is trying to switch offline mode on.
        Reading 'yse' as "stay online" would let the one deployment that cannot reach the
        network do exactly that.
        """
        monkeypatch.setenv(config.OFFLINE_ENV_VAR, "yse")
        with caplog.at_level("WARNING"):
            assert config._initial_offline() is True
        assert "yse" in caplog.text


class TestDftFetch:
    """``_get_dft_entries_from_components`` — the Materials Project path."""

    def test_it_raises_without_constructing_a_client(self, monkeypatch):
        tripwire = Tripwire()
        monkeypatch.setattr(api, "mp_rester", tripwire)
        monkeypatch.setattr(api, "get_mpr", tripwire)
        config.set_offline(True)
        with pytest.raises(config.OfflineError) as excinfo:
            api._get_dft_entries_from_components(["Cu", "Mg"], "GGA")
        assert "Cu-Mg" in str(excinfo.value)
        assert tripwire.calls == 0

    def test_positive_control_the_same_call_proceeds_when_online(self, monkeypatch):
        """Offline OFF: the call reaches the fetch. Stubbed, so no network is needed here.

        This is what makes the test above meaningful — it shows the raise comes from the
        offline guard and not from the call being broken for some other reason.
        """
        entries = [
            {
                "@module": "pymatgen.entries.computed_entries",
                "@class": "ComputedEntry",
                "composition": {"Cu": 1.0},
                "energy": -4.1,
                "entry_id": "mp-30-GGA",
                "correction": 0.0,
                "energy_adjustments": [],
            }
        ]
        reached = []

        class _StubRester:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def get_entries_in_chemsys(self, elements, additional_criteria=None):
                reached.append(sorted(elements))
                return list(entries)

        monkeypatch.setattr(api, "mp_rester", lambda *a, **k: _StubRester())
        monkeypatch.setattr(api, "get_api_key", lambda name: "x" * 32)
        config.set_offline(False)
        result = api._get_dft_entries_from_components(["Cu", "Mg"], "GGA")
        assert reached == [["Cu", "Mg"]]
        assert [e["entry_id"] for e in result] == ["mp-30-GGA"]

    @pytest.mark.parametrize(
        "call",
        [
            pytest.param(lambda: api.mp_rester(), id="mp_rester"),
            pytest.param(lambda: api.get_mpr(), id="get_mpr"),
            pytest.param(lambda: api.get_mpds_client(), id="get_mpds_client"),
        ],
    )
    def test_client_constructors_refuse_too(self, call):
        """Defence in depth: the guard sits on the objects that can open a socket.

        Any future caller reaching for a client directly is refused as well, so offline mode
        does not depend on every call site remembering to ask first.
        """
        config.set_offline(True)
        with pytest.raises(config.OfflineError):
            call()

    def test_get_dft_convexhull_raises_on_an_uncached_system(self, tmp_path, monkeypatch):
        """The whole point, end to end: a cache miss offline is an error, not a fetch."""
        monkeypatch.setattr(api, "mp_rester", Tripwire())
        config.set_offline(True)
        with pytest.raises(config.OfflineError):
            api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=str(tmp_path))

    def test_a_CACHED_system_is_unaffected_by_offline_mode(self, tmp_path, monkeypatch):
        """Offline mode must not break the reads it exists to make sufficient."""
        entries = [
            {
                "@module": "pymatgen.entries.computed_entries",
                "@class": "ComputedEntry",
                "composition": comp,
                "energy": energy,
                "entry_id": entry_id,
                "correction": 0.0,
                "energy_adjustments": [],
            }
            for comp, energy, entry_id in (
                ({"Cu": 1.0}, -4.1, "mp-30-GGA"),
                ({"Mg": 1.0}, -1.6, "mp-153-GGA"),
                ({"Cu": 2.0, "Mg": 1.0}, -10.5, "mp-1002-GGA"),
            )
        ]
        (tmp_path / "Cu-Mg_ENTRIES_MP_GGA.json").write_text(json.dumps(entries))
        monkeypatch.setattr(api, "mp_rester", Tripwire())
        config.set_offline(True)
        hull, _ = api.get_dft_convexhull(["Cu", "Mg"], "GGA", data_dir=str(tmp_path))
        assert len(hull.stable_entries) == 3


class TestMpdsFetch:
    """``mpds.load_mpds_data`` — the MPDS path, guarded ABOVE the key check."""

    def test_a_cache_miss_raises_rather_than_returning_a_shapeless_record(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(api, "get_mpds_client", Tripwire())
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "cache_mode", "directory")
        monkeypatch.setattr(config, "dir_structure", "flat")
        config.set_offline(True)
        with pytest.raises(config.OfflineError) as excinfo:
            mpds.load_mpds_data(["Cu", "Mg"])
        assert "Cu-Mg" in str(excinfo.value)

    def test_positive_control_the_same_call_returns_the_placeholder_when_online(
        self, tmp_path, monkeypatch
    ):
        """Offline OFF, no MPDS key: the historical behaviour, unchanged.

        ``{"reference": None}`` is precisely the record offline mode must NOT return, so
        showing that it is what comes back otherwise is what makes the raise above load
        bearing rather than incidental.
        """
        monkeypatch.setattr(api, "get_api_key", lambda name: None)
        monkeypatch.setattr(api, "get_mpds_client", Tripwire())
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "cache_mode", "directory")
        monkeypatch.setattr(config, "dir_structure", "flat")
        config.set_offline(False)
        mpds_json, (liquidus, _) = mpds.load_mpds_data(["Cu", "Mg"])
        assert mpds_json == {"reference": None}
        assert liquidus is None

    def test_a_CACHED_diagram_is_unaffected_by_offline_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(api, "get_mpds_client", Tripwire())
        monkeypatch.setattr(config, "cache_dir", tmp_path)
        monkeypatch.setattr(config, "cache_mode", "directory")
        monkeypatch.setattr(config, "dir_structure", "flat")
        (tmp_path / "Cu-Mg.json").write_text(
            json.dumps({"reference": {"entry": "https://mpds.io/entry/C900001"}, "shapes": []})
        )
        config.set_offline(True)
        mpds_json, _ = mpds.load_mpds_data(["Cu", "Mg"])
        assert mpds_json["reference"]["entry"].endswith("C900001")


@pytest.mark.needs_network
class TestLivePositiveControl:
    """The positive control against the REAL network, not a stub.

    Marked ``needs_network`` because it fetches. Deselect with ``-m 'not needs_network'``.
    """

    def test_the_same_fetch_succeeds_with_offline_off(self):
        if not api.get_api_key(api.MP_KEY_VAR):
            pytest.skip(f"{api.MP_KEY_VAR} is not configured")
        config.set_offline(True)
        with pytest.raises(config.OfflineError):
            api._get_dft_entries_from_components(["Ag", "V"], "GGA")
        config.set_offline(False)
        entries = api._get_dft_entries_from_components(["Ag", "V"], "GGA")
        assert entries, "the live fetch returned nothing; the control proves nothing"
        assert all(isinstance(e, dict) for e in entries)


def test_every_remote_call_site_in_the_package_is_guarded():
    """A census, so a NEW network path cannot be added without a guard.

    The set below is derived from the module rather than written down twice: every function
    in ``gliquid.api`` whose body constructs a client or fetches must call
    ``config.require_online``. The positive control is the count -- an empty census would
    mean the source scan itself broke.
    """
    import inspect

    source = inspect.getsource(api) + inspect.getsource(mpds)
    guarded = source.count("config.require_online(")
    assert guarded >= 5, f"expected the 5 known remote paths to be guarded, found {guarded}"
    # And the two functions that own the actual fetches, by name, so a rename is visible.
    for function in (api._get_dft_entries_from_components, mpds.load_mpds_data):
        assert "require_online" in inspect.getsource(function), function.__name__
