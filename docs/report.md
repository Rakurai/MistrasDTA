# MistrasDTA Parser Analysis Report

**Date:** January 29, 2026  
**File Analyzed:** `01-14-2026_16-12-09__0.dta`  
**Parser Version:** Current HEAD with waveform fix

---

## Part 1: Waveform Parsing Bug (Fixed)

### Summary

During a spec audit of the DTA binary reader, a critical bug was discovered in waveform message parsing (MID 173, SubID 1). The parser was returning only 17 samples per waveform instead of the expected 3072 samples, causing a ~99.4% data loss in waveform records.

---

## Bug Description

### Symptom

Test comparison against reference data failed with a waveform dtype mismatch:

| Field | Expected | Actual |
|-------|----------|--------|
| WAVEFORM dtype | `S24576` (3072 float64s) | `S136` (17 float64s) |

### Root Cause

The parser incorrectly used the `SLEN` field from the TRA hardware setup message (MID 42, SubID 173/42) as the waveform sample count.

**What `SLEN` actually represents:**  
`SLEN` is the "size of each setup struct in bytes" (value: 17), **not** the number of waveform samples.

**Actual waveform format:**  
The waveform message (MID 173, SubID 1) contains **no explicit sample count field**. All remaining bytes after the header are int16_le samples.

---

## What the Spec Says

### GUIDE.md §4.1 (Incorrect Documentation)

The guide documented the waveform format as:

```
SUBID:uint8==1, TOT:bytes[6], CID:uint8, ALB:uint8, N:uint16_le, SAMPLES:int16_le[N]
```

This implied an `N:uint16_le` field specifying sample count — **this field does not exist in actual data**.

### SPEC.md (PAC Message Definitions)

The spec does not provide byte-level detail for MID 173 waveform payloads. It references "AE waveform data" without explicit structure.

### Reference Implementation (`archive/MistrasDTA.working.py`)

The working reference implementation (lines 276-283) correctly reads all remaining bytes:

```python
remaining = len(buf) - self.cursor
n = remaining // 2  # 2 bytes per int16 sample
samples = struct.unpack_from(f"<{n}h", buf, self.cursor)
```

This confirms the actual format: **no N field; all remaining bytes are samples**.

---

## Testing Methodology

### Test File

- **DTA file:** `tests/dta/210527-CH1-15.DTA`
- **Reference:** `tests/reference/210527-CH1-15.npz`

### Diagnostic Steps

1. **Ran pytest** — test failed with waveform array mismatch
2. **Inspected reference npz** — confirmed expected WAVEFORM dtype `S24576` (3072 × 8 bytes)
3. **Inspected actual output** — found WAVEFORM dtype `S136` (17 × 8 bytes)
4. **Traced hardware config** — `SLEN=17` in all channels from MID 42/173,42
5. **Examined raw message bytes:**
   - Total message length: 6154 bytes
   - Header consumed: 10 bytes (SUBID + TOT + CID + ALB)
   - Remaining for samples: 6144 bytes = 3072 int16 samples ✓
6. **Verified against reference implementation** — confirmed it reads all remaining bytes

### Calculations

```
Message body length: 6154 bytes
- MID byte:           1 byte
- SUBID:              1 byte  
- TOT:                6 bytes
- CID:                1 byte
- ALB:                1 byte
─────────────────────────────
Remaining (samples): 6144 bytes ÷ 2 = 3072 samples ✓
```

---

## The Fix

### Code Changes

**File:** `MistrasDTA/MistrasDTA.py`

1. **`_decode_waveform_payload()` (lines 542-602):**
   - Removed incorrect `N:uint16_le` parsing
   - Changed to read `p.remaining()` bytes as samples
   - Updated docstring to reflect actual format

2. **`_ensure_hardware_recarray()` (lines 221-235):**
   - Removed `SLEN` column from hardware state (not needed)

3. **Hardware setup parsing (lines 378-404):**
   - Removed `SLEN` storage
   - Added comment clarifying `SLEN` is "setup struct size", not sample count

### Before (Incorrect)

```python
# Read N samples based on SLEN from hardware config
n = channel["SLEN"][0]
sample_bytes = p.read_bytes(2 * n)
```

### After (Correct)

```python
# All remaining bytes are int16_le samples (no N field)
sample_bytes = p.read_bytes(p.remaining())
```

---

## Test Results

### Before Fix

```
FAILED tests/test_MistrasDTA.py::test_dta_matches_reference[210527-CH1-15]
  - wfm dtype mismatch: expected S24576, got S136
```

### After Fix

```
PASSED tests/test_MistrasDTA.py::test_dta_matches_reference[210527-CH1-15]
```

Full test output:

```
tests/test_MistrasDTA.py::test_dta_matches_reference[210527-CH1-15] PASSED
tests/test_MistrasDTA.py::test_dta_matches_reference[01-14-2026_11-11-58__0] FAILED
  (missing reference file - pre-existing issue)
```

---

## Lessons Learned

1. **Documentation can be wrong.** GUIDE.md documented an `N:uint16_le` field that doesn't exist in actual data.

2. **Reference implementations are authoritative.** The working `archive/MistrasDTA.working.py` correctly parsed the format.

3. **Byte-level verification is essential.** Calculating expected vs actual byte consumption revealed the discrepancy immediately.

4. **Field names can be misleading.** `SLEN` ("setup length") was misinterpreted as "sample length".

---

## Recommendations

1. **Update GUIDE.md §4.1** to reflect the actual waveform format:
   ```
   SUBID:uint8==1, TOT:bytes[6], CID:uint8, ALB:uint8, SAMPLES:int16_le[]
   (all remaining bytes are samples; no explicit N field)
   ```

2. **Add regression test** specifically for waveform sample count verification.

3. **Document SLEN meaning** in hardware setup comments to prevent future confusion.
