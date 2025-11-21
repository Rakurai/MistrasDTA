import numpy as np
import MistrasDTA


def test_MistrasDTA(dta_file, ref_file):
    rec, wfm = MistrasDTA.read_bin(dta_file)
    ref = np.load(ref_file)

    # Do not compare TIMESTAMP, it's affected by timezone
    rec = np.lib.recfunctions.drop_fields(rec, "TIMESTAMP")
    rec_ref = np.lib.recfunctions.drop_fields(ref["rec"], "TIMESTAMP")

    np.testing.assert_array_equal(rec, rec_ref)
    np.testing.assert_array_equal(wfm, ref["wfm"])
