import numpy as np
import MistrasDTA


def test_MistrasDTA(dta_file, ref_file):
    rec, wfm = MistrasDTA.read_bin(dta_file)
    ref = np.load(ref_file, allow_pickle=True)

    # Do not compare TIMESTAMP, it's affected by timezone
    if hasattr(rec, 'dtype'):
        rec = np.lib.recfunctions.drop_fields(rec, "TIMESTAMP")
    rec_ref = ref["rec"]
    if hasattr(rec_ref, 'dtype') and rec_ref.dtype.names and "TIMESTAMP" in rec_ref.dtype.names:
        rec_ref = np.lib.recfunctions.drop_fields(rec_ref, "TIMESTAMP")

    np.testing.assert_array_equal(rec, rec_ref)
    np.testing.assert_array_equal(wfm, ref["wfm"])


def test_include_config(dta_file, ref_file):
    """include_config=True returns a 3-tuple with a well-formed config dict."""
    result = MistrasDTA.read_bin(dta_file, include_config=True)
    assert len(result) == 3
    rec, wfm, config = result

    # All expected keys present
    expected_keys = {
        "test_start_time", "product_name", "user_comment", "chid_list",
        "gain", "threshold", "hdt", "hlt", "pdt",
        "sampling_interval_ms", "demand_rate_ms",
        "demand_chid_list", "demand_pid_list",
        "partial_power_segments", "waveform_hardware",
    }
    assert set(config.keys()) == expected_keys

    # Gain, threshold, timing dicts are per-channel
    assert isinstance(config["gain"], dict)
    assert isinstance(config["threshold"], dict)
    assert isinstance(config["hdt"], dict)

    # waveform_hardware is a recarray with expected columns
    hw = config["waveform_hardware"]
    assert hasattr(hw, 'dtype')
    assert set(hw.dtype.names) >= {"CH", "SRATE", "TDLY"}


def test_include_td(dta_file, ref_file):
    """include_td=True returns a 3-tuple; td is a recarray when TD data exists."""
    rec, wfm, td = MistrasDTA.read_bin(dta_file, include_td=True)

    if hasattr(td, 'dtype'):
        # File has time-driven data
        assert len(td) > 0
        assert 'SSSSSSSS.mmmuuun' in td.dtype.names
    else:
        # File has no time-driven data — td is an empty list
        assert td == []


def test_multi_file(cont_files):
    """read_bin() accepts a list of continuation files."""
    assert len(cont_files) >= 2, "need at least 2 continuation files"

    rec, wfm, td = MistrasDTA.read_bin(cont_files, include_td=True)

    # Should produce combined results from all files
    assert hasattr(td, 'dtype')
    assert len(td) > 0
    assert hasattr(wfm, 'dtype')
    assert len(wfm) > 0


def test_iter_bin(dta_file, ref_file):
    """iter_bin() yields (type, data) tuples consistent with read_bin()."""
    ref = np.load(ref_file, allow_pickle=True)
    rec_ref = ref["rec"]

    events = list(MistrasDTA.iter_bin(dta_file))
    types = {t for t, _ in events}

    # Must contain at least hits or other data events
    assert len(events) > 0

    hits = [d for t, d in events if t is MistrasDTA.EventType.HIT]
    assert len(hits) == len(rec_ref)

    # Verify first hit's channel matches reference
    if hits and hasattr(rec_ref, 'dtype'):
        assert hits[0][1] == rec_ref["CH"][0]  # cid is second element
