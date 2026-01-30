# Feature Gap Analysis: MistrasDTA Parser

**Date:** January 29, 2026  
**File Analyzed:** `01-14-2026_16-12-09__0.dta`  
**Result:** 475 hits, 475 waveforms, 3718 time-driven records

---

## Summary of Discarded Data

The parser output shows several categories of unhandled data:

| Category | Count | Impact |
|----------|-------|--------|
| Unknown MID 116 | 6 messages | Unknown data lost |
| Unknown MID 11 | 1 message | Unknown data lost |
| Unhandled HW subrecords | 20 unique subIDs | Possible config missing |
| Hit parametrics | 9 bytes/hit (first 5 logged) | Parametric data not decoded |
| Control message tail | 1 byte after RTOT | Minor |

---

## 1. Unknown Message IDs

### MID 116 (6 occurrences, 12–7187 bytes each)

**Status:** Not documented in DTA_SPEC.md  
**Action Required:** Reverse-engineer or obtain spec

```
unknown msg_id=116 skipping payload=12 bytes
unknown msg_id=116 skipping payload=9 bytes
unknown msg_id=116 skipping payload=712 bytes
unknown msg_id=116 skipping payload=7187 bytes
unknown msg_id=116 skipping payload=175 bytes
```

**Speculation:** Large payloads (7187 bytes) suggest this could be:
- Diagnostic/status data
- Extended waveform or spectrum data
- Location/clustering results

### MID 11 (1 occurrence, 0 bytes)

**Status:** Not documented  
**Action Required:** Likely a marker/flag message with no payload

---

## 2. Unhandled Hardware Setup Subrecords (MID 42)

The following subIDs within MID 42 are skipped:

| SubID | Length | Spec Reference | Likely Purpose |
|-------|--------|----------------|----------------|
| 100 | 1 byte | — | Unknown |
| 102 | 5 bytes | — | Unknown |
| 103 | 3 bytes | — | Unknown |
| 106 | 7 bytes | — | Unknown |
| 109 | 36 bytes | DTA_SPEC §4.9 | **Partial Power Setup** (embedded in MID 42) |
| 110 | 3 bytes | — | Unknown |
| 124 | 1 byte | — | Unknown |
| 136 | 12 bytes | — | Unknown |
| 172 | 15 bytes | — | TRA-related (see MID 173) |
| 173 | 2 bytes | DTA_SPEC §4.2.3 | TRA container (short variant) |
| 19 | 8 bytes | — | Unknown |
| 20 | 3 bytes | — | Unknown |
| 22 | 4 bytes | — | Unknown |
| 24 | 4 bytes | — | Unknown |
| 25 | 4 bytes | — | Unknown |
| 26 | 4 bytes | — | Unknown |
| 28 | 2 bytes | — | Unknown |
| 29 | 26 bytes | — | Unknown |
| 30 | 2 bytes | — | Unknown |
| 33 | 10 bytes | — | Unknown |
| 146 | ~10 bytes | — | Unknown |

### Priority Items

**SubID 109 (36 bytes) — Partial Power Setup**

Per DTA_SPEC.md §4.9, this defines segment count for CHID 22 (Partial Power). The current parser handles MID 109 as a top-level message but NOT when embedded as a subrecord within MID 42.

**Current code:**
```python
elif msg_id == 109:
    _ = p.u8()  # SEGMENT_TYPE
    state.partial_power_segments = p.u16()  # N_SEG
```

**Gap:** When subID 109 appears inside MID 42, it's skipped. This may cause CHID 22 decoding to fail.

**SubID 23 tail (1 byte unconsumed)**

The Set Gain handler reads CID + GAIN_DB (2 bytes) but leaves 1 byte. Per spec, this may be padding or an additional field.

---

## 3. Hit Parametrics (MID 1)

**Current behavior:** Dropped as trailing bytes

```
msg_id=1 dropping 9 trailing bytes (hit parametrics: PID+u16 pairs)
```

**Spec says (DTA_SPEC §4.1.4):**
> Parametric data follows the feature vector. Format: `PID:uint8 + VALUE:uint16_le` repeated.

**9 bytes = 3 parametric values** (3 × 3 bytes each)

**To implement:**
```python
# After feature vector decoding:
parametrics = {}
while p.remaining() >= 3:
    pid = p.u8()
    val = p.u16()
    parametrics[pid] = val
```

**Impact:** Parametric channels (external sensors, load cells, etc.) are currently lost.

---

## 4. Time-Driven Feature Vector Decoding

**Current output:**
```
Time-driven event columns: ('SSSSSSSS.mmmuuun', 'PID_1', 'CID_1_FV', 'CID_2_FV', 'CID_3_FV', 'CID_4_FV')
```

The `CID_*_FV` columns contain raw bytes. These should be decoded using the demand CHID list (from MID 42 SubID 6).

**Current code stores raw bytes:**
```python
per_channel[cid] = fv  # raw bytes
```

**Gap:** `_decode_td_fv_bytes()` exists but isn't called in the output path.

---

## 5. Control Message Tail (MID 128)

```
msg_id=128 dropping 1 bytes after RTOT
```

**Spec says (DTA_SPEC §4.8):**
> Layout: `RTOT (6 bytes) [additional bytes?]`

The extra byte is undocumented. Possibly a status flag or padding.

---

## Implementation Priority

### High Priority (Data Loss)

1. **Hit Parametrics** — 9 bytes per hit being dropped
2. **SubID 109 in MID 42** — May affect CHID 22 decoding
3. **MID 116** — Large payloads (up to 7KB) being skipped

### Medium Priority (Incomplete Decode)

4. **Time-driven FV decode** — Raw bytes instead of named fields
5. **SubID 23 tail byte** — Gain message has extra byte

### Low Priority (Informational)

6. **MID 11** — Empty payload, likely marker
7. **Other HW subrecords** — Unknown purpose, may be informational

---

## Recommended Next Steps

1. **Investigate MID 116** — Dump raw bytes, look for structure
2. **Handle SubID 109 inside MID 42** — Extract partial power segment count
3. **Decode hit parametrics** — Add to HitEvent model
4. **Wire up TD feature decode** — Use existing `_decode_td_fv_bytes()`
5. **Document unknown subIDs** — Capture raw bytes for future analysis

---

## Appendix: Full Discard Log

```
[discard msg41_ascii #1] msg_id=41 dropping 31 bytes (ASCII: product name + version string)
[discard msg7_comment #1] msg_id=7 dropping 48 bytes (TEXT: ASCII user comment)
[discard hw_subrecord #1-#20] Various unhandled subrecords
[discard unknown_msg #1-#6] MID 116 (×5) + MID 11 (×1)
[discard control_tail #1] msg_id=128 dropping 1 bytes after RTOT
[discard hit_parametrics #1-#5] 9 bytes each (continues for all 475 hits)
```
