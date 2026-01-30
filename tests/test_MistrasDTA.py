import numpy as np
import MistrasDTA


def test_MistrasDTA(dta_file, ref_file):
    rec, wfm, td = MistrasDTA.read_bin(dta_file, include_td=True)
    ref = np.load(ref_file)

    # Do not compare TIMESTAMP, it's affected by timezone
    # rec/wfm may be empty lists if no data, or recarrays if data exists
    if hasattr(rec, 'dtype') and rec.dtype.names and 'TIMESTAMP' in rec.dtype.names:
        rec = np.lib.recfunctions.drop_fields(rec, "TIMESTAMP")
    if hasattr(ref["rec"], 'dtype') and ref["rec"].dtype.names and 'TIMESTAMP' in ref["rec"].dtype.names:
        rec_ref = np.lib.recfunctions.drop_fields(ref["rec"], "TIMESTAMP")
    else:
        rec_ref = ref["rec"]

    np.testing.assert_array_equal(rec, rec_ref)
    np.testing.assert_array_equal(wfm, ref["wfm"])
