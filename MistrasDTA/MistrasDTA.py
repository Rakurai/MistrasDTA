import logging
from pathlib import Path
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple, Any, Iterable, Union

import numpy as np
from numpy.lib.recfunctions import append_fields

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# --- Discard logging (rate-limited) ------------------------------------------

_DISCARD_COUNTS: Dict[str, int] = {}


def _log_discard(key: str, msg: str, *, every: int = 1000, first: int = 5) -> None:
    """
    Log discards at INFO without flooding:
      - log the first `first` occurrences
      - then log every `every` occurrences
    """
    n = _DISCARD_COUNTS.get(key, 0) + 1
    _DISCARD_COUNTS[key] = n
    if n <= first or (every > 0 and n % every == 0):
        logger.info(f"[discard {key} #{n}] {msg}")


# --- Feature metadata (SPEC: §2.1.1 Fixed-width CHID mapping) -----------------
# Maps CHID (AE Characteristic ID) to human-readable names.
# Source: GUIDE.md §2.1.1, SPEC.md ID 5 — Event Data Set Definition

CHID_to_str = {
    1: "RISE",           # Rise time (uint16)
    2: "PCNTS",          # Counts to peak (uint16)
    3: "COUN",           # Counts (uint16)
    4: "ENER",           # Energy counts (uint16)
    5: "DURATION",       # Duration (uint32)
    6: "AMP",            # Amplitude dB (uint8)
    8: "ASL",            # ASL dB (uint8)
    10: "THR",           # Threshold (uint8)
    13: "A-FRQ",         # Avg frequency kHz (uint16)
    17: "RMS",           # RMS16 scaled (uint16)
    18: "R-FRQ",         # Reverberation freq kHz (uint16)
    19: "I-FRQ",         # Initiation freq kHz (uint16)
    20: "SIG STRENGTH",  # Signal strength scaled (uint32)
    21: "ABS-ENERGY",    # Absolute energy scaled (float32)
    22: "PARTIAL POWER", # Partial power (variable-length, N_SEG bytes from MID 109)
    23: "FRQ-C",         # Frequency centroid kHz (uint16)
    24: "P-FRQ",         # Peak frequency kHz (uint16)
    31: "UNKNOWN",       # Placeholder for unknown CHIDs
}

# CHID byte lengths per GUIDE.md §2.1.1
# CHID 22 is special: variable-length = partial_power_segment_count bytes (from MID 109)
CHID_byte_len = {
    1: 2,   # RISE: uint16_le
    2: 2,   # PCNTS: uint16_le
    3: 2,   # COUN: uint16_le
    4: 2,   # ENER: uint16_le
    5: 4,   # DURATION: uint32_le
    6: 1,   # AMP: uint8
    8: 1,   # ASL: uint8
    10: 1,  # THR: uint8
    13: 2,  # A-FRQ: uint16_le
    17: 2,  # RMS16: uint16_le (scaled by 1/5000)
    18: 2,  # R-FRQ: uint16_le
    19: 2,  # I-FRQ: uint16_le
    20: 4,  # SIG STRENGTH: uint32_le (scaled by 3.05)
    21: 4,  # ABS-ENERGY: float32_le (scaled by 9.31e-4)
    22: 0,  # PARTIAL POWER: variable-length = N_SEG bytes (from MID 109 §3.19)
    23: 2,  # FRQ-C: uint16_le
    24: 2,  # P-FRQ: uint16_le
    31: 2,  # UNKNOWN: placeholder
}

# --- Known unimplemented message IDs -----------------------------------------
# These MIDs appear in real files but have no available spec documentation.
# They are silently skipped (no log spam) since we can't decode them anyway.

KNOWN_UNIMPLEMENTED_MIDS = {
    11,   # Unknown marker/flag (0 bytes payload)
    116,  # Unknown extended data (variable payload, up to 7KB observed)
}

# --- Known unimplemented hardware setup subrecord IDs ------------------------
# These SubIDs appear in MID 42 but have no available spec or are metadata-only.
# They are silently skipped to reduce log noise.

KNOWN_UNIMPLEMENTED_SUBIDS = {
    19,   # Reserved (no spec)
    20,   # Reserved for Pre-amp Gain (no detail)
    28,   # Alarm Definition (have spec but not needed for data decode)
    29,   # AE Filter Definition (have spec but not needed for data decode)
    30,   # Delta-T Filter Definition (have spec but not needed for data decode)
    33,   # Reserved (no spec)
    100,  # Start of Test Setup (marker only)
    101,  # End of Setup (marker only)
    103,  # Unknown (no spec)
    106,  # Group Definition (have spec but not needed for data decode)
    110,  # Group Parametric Assignment (have spec but not needed for data decode)
    115,  # Unknown (appears with 39 bytes, no spec)
    124,  # End of Group Settings (marker only)
    136,  # Unknown (no spec)
    137,  # Unknown (no spec)
    138,  # Unknown (no spec)
    139,  # Unknown (appears with 6 bytes, no spec)
    146,  # Unknown (no spec)
    148,  # Unknown (appears with 4370 bytes, large block, no spec)
    151,  # Unknown (appears with 135 bytes, no spec)
    154,  # Unknown (appears with 41 bytes, no spec)
    172,  # Digital Filter Setup (mentioned but no detail)
    176,  # Unknown (no spec)
}


# --- Low-level formats (GUIDE.md §0.1, §1) -----------------------------------
# All multi-byte integers are little-endian (LSB first) per GUIDE.md §0.1

FMT_U8 = "<B"      # uint8: 1 byte
FMT_U16 = "<H"     # uint16_le: 2 bytes
FMT_I16 = "<h"     # int16_le: 2 bytes
FMT_I32 = "<i"     # int32_le: 4 bytes
FMT_F32 = "<f"     # float32_le: 4 bytes IEEE-754
FMT_RTOT = "<IH"   # RTOT/TOT: 6 bytes = uint32_le + uint16_le (GUIDE.md §1.1.1)


def _rtot_to_seconds(u32: int, u16: int) -> float:
    """Convert 6-byte RTOT (uint48_le) to seconds. Unit = 0.25 µs per tick."""
    return ((u32 + (2**32) * u16) * 0.25e-6)


# --- Consuming readers --------------------------------------------------------

class StreamReader:
    """Reads bytes from a file-like object and unpacks fixed layouts by format."""

    __slots__ = ("fp",)

    def __init__(self, fp: BinaryIO):
        self.fp = fp

    def read_exact(self, n: int) -> bytes:
        b = self.fp.read(n)
        if b is None or len(b) != n:
            raise EOFError(f"Expected {n} bytes, got {0 if b is None else len(b)}")
        return b

    def read_u16(self) -> int:
        return struct.unpack(FMT_U16, self.read_exact(2))[0]

    def read_u8(self) -> int:
        return struct.unpack(FMT_U8, self.read_exact(1))[0]


class PayloadReader:
    """
    A consuming reader over an in-memory payload.
    Uses memoryview + struct.unpack_from for speed; no explicit offsets in user code.
    """

    __slots__ = ("_buf", "_off")

    def __init__(self, payload: bytes | memoryview):
        self._buf = payload if isinstance(payload, memoryview) else memoryview(payload)
        self._off = 0

    def remaining(self) -> int:
        return len(self._buf) - self._off

    def read_bytes(self, n: int) -> bytes:
        end = self._off + n
        b = self._buf[self._off:end]
        self._off = end
        return b.tobytes()

    def unpack(self, fmt: str) -> tuple:
        size = struct.calcsize(fmt)
        out = struct.unpack_from(fmt, self._buf, self._off)
        self._off += size
        return out

    def u8(self) -> int:
        return self.unpack(FMT_U8)[0]

    def u16(self) -> int:
        return self.unpack(FMT_U16)[0]

    def i16(self) -> int:
        return self.unpack(FMT_I16)[0]

    def i32(self) -> int:
        return self.unpack(FMT_I32)[0]

    def f32(self) -> float:
        return self.unpack(FMT_F32)[0]

    def rtot_seconds(self) -> float:
        lo, hi = self.unpack(FMT_RTOT)
        return _rtot_to_seconds(lo, hi)


# --- Pydantic models ----------------------------------------------------------

class HitEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rtot_s: float
    cid: int
    features: Dict[str, Any]
    parametrics: Dict[int, int] = {}  # PID -> value (from hit parametric channels)


class WaveformEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tot_s: float
    cid: int
    srate: int
    tdly: int
    waveform_bytes: bytes  # float64 bytes


class TimeDrivenEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rtot_s: float
    parametrics: Dict[int, int]              # PID -> value (u16 in current decode)
    per_channel: Dict[int, Dict[str, Any]]   # CID -> decoded feature dict (named fields)


class TestStartEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_start_time: datetime


# --- Parser state -------------------------------------------------------------

@dataclass
class _State:
    chid_list: Tuple[int, ...] = ()
    gain: Dict[int, int] = None
    hardware_rows: List[List[int]] = None
    hardware: Optional[np.recarray] = None
    test_start_time: Optional[datetime] = None
    partial_power_segments: int = 0
    demand_chid_list: Tuple[int, ...] = ()
    demand_pid_list: Tuple[int, ...] = ()
    # Additional hardware config collected from MID 42 subrecords
    threshold: Dict[int, int] = None       # CID -> threshold dB (SubID 22)
    hdt: Dict[int, int] = None             # CID -> hit definition time µs (SubID 24)
    hlt: Dict[int, int] = None             # CID -> hit lockout time µs (SubID 25)
    pdt: Dict[int, int] = None             # CID -> peak definition time µs (SubID 26)
    sampling_interval_ms: Optional[int] = None  # global sampling interval (SubID 27)
    demand_rate_ms: Optional[int] = None   # demand sampling rate (SubID 102)
    # ASCII metadata from MID 7 and MID 41
    product_name: Optional[str] = None     # product name + version (MID 41)
    user_comment: Optional[str] = None     # user comment / test label (MID 7)

    def __post_init__(self) -> None:
        if self.gain is None:
            self.gain = {}
        if self.hardware_rows is None:
            self.hardware_rows = []
        if self.threshold is None:
            self.threshold = {}
        if self.hdt is None:
            self.hdt = {}
        if self.hlt is None:
            self.hlt = {}
        if self.pdt is None:
            self.pdt = {}


def _ensure_hardware_recarray(state: _State) -> None:
    if state.hardware is not None:
        return
    if state.hardware_rows:
        state.hardware = np.rec.fromrecords(
            state.hardware_rows, names=["CH", "SRATE", "TDLY"]
        )
    else:
        state.hardware = np.recarray(
            (0,),
            dtype=[
                ("CH", np.int32),
                ("SRATE", np.int32),
                ("TDLY", np.int32),
            ],
        )


# --- Message decoders ---------------------------------------------------------

def _decode_hit_payload(p: PayloadReader, state: _State) -> HitEvent:
    """
    Decode MID 1 — AE Hit / Event Data (GUIDE.md §3.1, SPEC.md ID 1)
    
    Layout:
      MID:uint8 == 1           (already consumed by caller)
      RTOT:bytes[6]            relative time of test (uint48_le, 0.25µs ticks)
      CID:uint8                AE channel number
      AE_FEATURES:             for each CHID in event_chids (from MID 5):
                                 - fixed width per CHID_byte_len, OR
                                 - CHID 22: partial_power_segment_count bytes
      HIT_PARAMETRICS:         repeat until end of message:
                                 PID:uint8 + VALUE:uint16_le [+ V3:uint8 if configured]
    """
    rtot_s = p.rtot_seconds()  # bytes[0:6]: RTOT (6 bytes)
    cid = p.u8()               # byte[6]: CID (1 byte)

    features: Dict[str, Any] = {}

    for chid in state.chid_list:
        if chid not in CHID_to_str:
            _log_discard(
                "unknown_chid",
                f"CHID list contains unknown ID {chid}; decode will likely fail without a spec",
                every=1,
                first=50,
            )
            raise KeyError(chid)

        name = CHID_to_str[chid]
        b = CHID_byte_len[chid]

        if name == "RMS":
            v = p.u16() / 5000.0
        elif name == "DURATION":
            v = p.i32()
        elif name == "SIG STRENGTH":
            v = p.i32() * 3.05
        elif name == "ABS-ENERGY":
            v = p.f32() * 9.31e-4
        elif name == "PARTIAL POWER":
            n = state.partial_power_segments
            v = p.read_bytes(n)
        else:
            if b == 1:
                v = p.u8()
            elif b == 2:
                v = p.u16()
            elif b == 4:
                v = p.read_bytes(4)
            else:
                v = p.read_bytes(b)

        features[name] = v

    # --- HIT_PARAMETRICS: remaining bytes (PID:u8 + VALUE:u16 per parametric) ---
    # Per DTA_SPEC §4.1.4: repeat until end of message
    # Format: PID:uint8, VALUE:uint16_le per parametric channel
    # Note: Messages end with 2 undocumented bytes (observed: varies, possibly checksum)
    parametrics: Dict[int, int] = {}
    while p.remaining() >= 5:  # need PID(1) + VALUE(2) + trailing(2)
        pid = p.u8()    # PID: parametric channel ID
        val = p.u16()   # VALUE: parametric value
        parametrics[pid] = val
    
    # Consume trailing 2 bytes (undocumented, possibly checksum or padding)
    if p.remaining() >= 2:
        _ = p.u16()

    return HitEvent(rtot_s=rtot_s, cid=cid, features=features, parametrics=parametrics)


def _decode_hw_setup_payload(p: PayloadReader, state: _State) -> None:
    """
    Decode MID 42 — Hardware Setup (GUIDE.md §3.18, SPEC.md ID 42)
    
    Layout (after MID byte and 0x00 pad consumed by caller):
      MVERN:uint16_le          message version number
      Submessages:             repeat until end:
                                 LSUB:uint16_le (length of submessage)
                                 SUBID:uint8 + SUBBODY:bytes[LSUB-1]
    
    Submessage IDs we handle:
      5   Event Data Set Definition (CHID list for MID 1)
      6   Demand Data Set Definition (CHID/PID lists for MID 2/3)
      23  Set Gain
      173 TRA setup container (subid2=42 for waveform hardware config)
    """
    _ = p.u16()  # bytes[0:2]: MVERN (message version number, ignored)

    while p.remaining() > 0:
        lsub = p.u16()
        sub_len = lsub
        raw_sub = p.read_bytes(lsub)
        subpayload = PayloadReader(raw_sub)

        subid = subpayload.u8()
        handled = False

        if subid == 5:
            # --- SUBID 5: Event Data Set Definition (GUIDE.md §3.5, SPEC.md ID 5) ---
            # Layout: N_CHID:uint8, CHIDS:uint8[N_CHID], MAX_HIT_PID:uint8
            handled = True
            n = subpayload.u8()                      # byte[0]: N_CHID
            chids = subpayload.unpack(f"<{n}B")     # bytes[1:1+n]: CHID values
            state.chid_list = tuple(int(x) for x in chids)
            _ = subpayload.u8()                      # MAX_HIT_PID

            unknown = [c for c in state.chid_list if c not in CHID_to_str]
            if unknown:
                _log_discard(
                    "unknown_chid_list",
                    f"Event Data Set Definition contains unknown CHIDs: {unknown}",
                    every=1,
                    first=50,
                )

        elif subid == 6:
            # --- SUBID 6: Demand/Time-Driven Data Set Definition (GUIDE.md §3.6, SPEC.md ID 6) ---
            # Layout: N_CHID:uint8, CHIDS:uint8[N_CHID], N_PID:uint8, PIDS:uint8[N_PID]
            handled = True
            n_chid = subpayload.u8()                     # byte[0]: N_CHID
            chids = subpayload.unpack(f"<{n_chid}B")    # bytes[1:1+n_chid]: CHID values
            state.demand_chid_list = tuple(int(x) for x in chids)

            n_pid = subpayload.u8()                      # byte[1+n_chid]: N_PID
            pids = subpayload.unpack(f"<{n_pid}B")      # bytes[2+n_chid:2+n_chid+n_pid]: PID values
            state.demand_pid_list = tuple(int(x) for x in pids)

        elif subid == 23:
            # --- SUBID 23: Set Gain (GUIDE.md §3.12, SPEC.md ID 23) ---
            # Layout: CID:uint8, GAIN:uint8, FLAGS:uint8
            handled = True
            cid = subpayload.u8()       # byte[0]: CID (channel number)
            gain_db = subpayload.u8()   # byte[1]: gain in dB
            _ = subpayload.u8()         # byte[2]: flags/mode (observed: 0x14)
            state.gain[cid] = gain_db

        elif subid == 22:
            # --- SUBID 22: Set Threshold (SPEC.md ID 22) ---
            # Layout: CID:uint8, V:uint8 (threshold dB, MSB=1 float, =0 fix), FLAGS:uint8
            handled = True
            cid = subpayload.u8()
            thr = subpayload.u8()
            _ = subpayload.u8()         # byte[2]: flags/mode (observed: 0x06)
            state.threshold[cid] = thr

        elif subid == 24:
            # --- SUBID 24: Set Hit Definition Time (SPEC.md ID 24) ---
            # Layout: CID:uint8, V:uint16_le (steps of 2 µs)
            handled = True
            cid = subpayload.u8()
            val = subpayload.u16()
            state.hdt[cid] = val * 2  # convert to µs

        elif subid == 25:
            # --- SUBID 25: Set Hit Lockout Time (SPEC.md ID 25) ---
            # Layout: CID:uint8, V:uint16_le (steps of 2 µs)
            handled = True
            cid = subpayload.u8()
            val = subpayload.u16()
            state.hlt[cid] = val * 2  # convert to µs

        elif subid == 26:
            # --- SUBID 26: Set Peak Definition Time (SPEC.md ID 26) ---
            # Layout: CID:uint8, V:uint16_le (steps of 1 µs)
            handled = True
            cid = subpayload.u8()
            val = subpayload.u16()
            state.pdt[cid] = val  # already in µs

        elif subid == 27:
            # --- SUBID 27: Set Sampling Interval (SPEC.md ID 27) ---
            # Layout: V:uint16_le (steps of 1 ms)
            handled = True
            val = subpayload.u16()
            state.sampling_interval_ms = val

        elif subid == 102:
            # --- SUBID 102: Set Demand Sampling Rate (SPEC.md) ---
            # Layout: V:uint16_le (ms), PAD:uint16_le
            handled = True
            val = subpayload.u16()
            state.demand_rate_ms = val
            _ = subpayload.u16()  # padding/reserved

        elif subid == 109:
            # --- SUBID 109: Partial Power Setup (embedded in MID 42) ---
            # Layout: SEGMENT_TYPE:uint8, N_SEG:uint16_le, PAD:3 bytes,
            #         then N_SEG frequency band definitions (8 bytes each)
            handled = True
            _ = subpayload.u8()   # SEGMENT_TYPE (always 0)
            n_seg = subpayload.u16()
            state.partial_power_segments = n_seg
            _ = subpayload.read_bytes(3)           # 3 bytes padding
            _ = subpayload.read_bytes(n_seg * 8)   # frequency band definitions

        elif subid == 173:
            # --- SUBID 173: TRA setup container (nested submessage) ---
            # Contains another SUBID2 byte; we handle subid2=42.
            if sub_len < 2:
                # Too short for subid2, skip silently
                handled = True
            else:
                subid2 = subpayload.u8()  # byte[0]: SUBID2
                if subid2 == 42:
                    # --- SUBID 173, SUBID2 42: TRA Hardware Setup (GUIDE.md §4.4) ---
                    # Layout per GUIDE.md §4.4 (MID 172,42 / 173,42):
                    #   MVERN:uint16_le, ADT:uint8, SETS:uint8, SETS_PAD:uint8, SLEN:uint16_le
                    #   Then SETS repetitions of channel setup struct.
                    # NOTE: SLEN here is "size of each setup struct in bytes", NOT sample count!
                    #       Waveform sample count N is embedded in MID 173 subid 1 message itself.
                    handled = True
                    _ = subpayload.read_bytes(2)  # bytes[1:3]: MVERN (uint16_le, e.g. 100 = v1.00)
                    _ = subpayload.read_bytes(1)  # byte[3]: ADT (A/D data type; 2=16-bit signed)
                    _ = subpayload.read_bytes(2)  # bytes[4:6]: SETS + SETS_PAD (uint8 each)
                    _ = subpayload.read_bytes(2)  # bytes[6:8]: SLEN (setup struct size, NOT sample count)
                    # Per-channel setup block:
                    ch = subpayload.u8()          # byte[8]: CHID (channel, 0=all channels)
                    _ = subpayload.read_bytes(2)  # bytes[9:11]: HLK (hit lockout, uint16_le)
                    _ = subpayload.read_bytes(2)  # bytes[11:13]: HITS (uint16_le)
                    srate = subpayload.u16()      # bytes[13:15]: SRATE (sample rate, uint16_le)
                    _ = subpayload.read_bytes(2)  # bytes[15:17]: TMODE (trigger mode, uint16_le)
                    _ = subpayload.read_bytes(2)  # bytes[17:19]: TSRC (trigger source, uint16_le)
                    tdly = subpayload.i16()       # bytes[19:21]: TDLY (trigger delay, int16_le, negative=pretrigger)
                    _ = subpayload.read_bytes(2)  # bytes[21:23]: MXIN (max input, uint16_le)
                    _ = subpayload.read_bytes(2)  # bytes[23:25]: THRD (threshold, uint16_le)

                    # Store: SRATE is in kHz, multiply by 1000 for Hz
                    state.hardware_rows.append([ch, 1000 * srate, tdly])
                else:
                    # Unknown subid2 variant (e.g., short 2-byte payload with subid2 != 42)
                    # These appear to be flags/markers, silently skip
                    handled = True

        # --- Skip known-unimplemented SubIDs silently ---
        if subid in KNOWN_UNIMPLEMENTED_SUBIDS:
            handled = True  # mark as handled to suppress log

        if not handled:
            _log_discard(
                "hw_subrecord",
                f"msg_id=42 skipping unhandled subrecord subid={subid} len={sub_len} bytes",
                every=200,
                first=20,
            )

        left = subpayload.remaining()
        if left > 0:
            # Only log tail bytes for SubIDs we actively decode (not silently skipped ones)
            if subid not in KNOWN_UNIMPLEMENTED_SUBIDS:
                _log_discard(
                    "hw_subrecord_tail",
                    f"msg_id=42 subid={subid} leaving {left} unconsumed bytes (fields not decoded)",
                    every=200,
                    first=20,
                )
            _ = subpayload.read_bytes(left)

    state.hardware = None
    _ensure_hardware_recarray(state)


def _decode_test_start_payload(p: PayloadReader, state: _State) -> TestStartEvent:
    """
    Decode MID 99 — Time and Date of Test Start (GUIDE.md §3.18 note, SPEC.md ID 99)
    
    Layout (after MID byte consumed by caller):
      TEXT:bytes[LEN-1]        ASCII date string, e.g. "Sun Jul 03 08:49:55 1988\n"
    """
    raw = p.read_bytes(p.remaining())  # entire remaining payload is ASCII text
    s = raw.decode("ascii").strip("\x00")
    dt = datetime.strptime(s, "%a %b %d %H:%M:%S %Y\n")
    state.test_start_time = dt
    return TestStartEvent(test_start_time=dt)


def _decode_td_fv_bytes(fv: bytes, demand_chids: Tuple[int, ...], state: _State) -> Dict[str, Any]:
    """
    Decode time-driven feature vector (FV) bytes into named fields.
    
    The FV is a concatenation of values for each CHID in demand_chids (from MID 42 subid 6).
    Each value uses the byte length from CHID_byte_len, except CHID 22 which uses
    partial_power_segments bytes (from MID 109).
    
    Scaling factors applied per GUIDE.md §2.1.1:
      - CHID 17 (RMS): uint16 / 5000.0
      - CHID 21 (ABS-ENERGY): float32 * 9.31e-4
    """
    pr = PayloadReader(fv)
    out: Dict[str, Any] = {}

    for chid in demand_chids:
        name = CHID_to_str.get(chid, f"CHID_{chid}")

        if chid == 8:    # ASL: uint8
            out[name] = pr.u8()
        elif chid == 17: # RMS16: uint16_le, scaled
            out[name] = pr.u16() / 5000.0
        elif chid == 21: # ABS-ENERGY: float32_le, scaled
            out[name] = pr.f32() * 9.31e-4
        elif chid == 22: # PARTIAL POWER: N_SEG bytes (variable from MID 109)
            n = int(state.partial_power_segments or 0)
            out[name] = pr.read_bytes(n)
        else:
            # Default: read CHID_byte_len[chid] bytes
            bl = int(CHID_byte_len.get(chid, 0))
            out[name] = pr.read_bytes(bl)

    return out


def _decode_time_driven_payload(p: PayloadReader, state: _State) -> Optional[TimeDrivenEvent]:
    """
    Decode MID 2/3 — Time-Driven / User-Forced Sample Data (GUIDE.md §3.2/§3.3, SPEC.md ID 2/3)
    
    Layout (after MID byte consumed by caller):
      RTOT:bytes[6]            relative time of test (uint48_le, 0.25µs ticks)
      PARAMETRICS:             for each PID in demand_pids (from MID 42 subid 6):
                                 PID:uint8 + VALUE:uint16_le [+ V3:uint8 if configured]
      CHANNEL_BLOCKS:          repeat until end:
                                 CID:uint8
                                 FV:bytes[fv_len]  (feature vector for this channel)
    
    FV length = sum of CHID_byte_len for each CHID in demand_chids,
                with CHID 22 contributing partial_power_segments bytes (from MID 109).
    """
    rtot_s = p.rtot_seconds()  # bytes[0:6]: RTOT (6 bytes)

    # --- PARAMETRICS: PID(u8) + VALUE(u16) for each PID in demand_pid_list ---
    # Per GUIDE.md §3.2 step 3: framed as PID:uint8 + VALUE:uint16_le
    parametrics: Dict[int, int] = {}
    for _ in range(len(state.demand_pid_list)):
        if p.remaining() < 3:  # need PID(1) + VALUE(2)
            _log_discard(
                "time_driven_trunc_param",
                f"msg_id=2/3 tot={rtot_s:.6f} truncated while reading parametrics; remaining={p.remaining()}",
                every=50,
                first=50,
            )
            break
        pid = p.u8()   # PID: uint8 (parametric channel ID)
        val = p.u16()  # VALUE: uint16_le (parametric value)
        parametrics[pid] = val

    # --- Compute FV length from demand_chid_list ---
    # Per GUIDE.md §3.2 step 4: FV is concatenated values for each demand CHID.
    # CHID 22 is special: uses partial_power_segments bytes (from MID 109 §3.19).
    fv_len = 0
    for chid in state.demand_chid_list:
        if chid == 22:
            # CHID 22 (PARTIAL POWER): variable length from MID 109
            fv_len += int(state.partial_power_segments or 0)
        else:
            fv_len += int(CHID_byte_len.get(chid, 0))

    # --- CHANNEL_BLOCKS: repeat CID(u8) + FV(fv_len bytes) until end ---
    per_channel: Dict[int, Dict[str, Any]] = {}

    # Need at least CID(1) + FV(fv_len) bytes
    while p.remaining() >= 1 + fv_len and fv_len > 0:
        cid = p.u8()                 # CID: uint8 (channel ID)
        fv = p.read_bytes(fv_len)    # FV: feature vector bytes
        # Decode FV into named fields using demand CHID list
        per_channel[cid] = _decode_td_fv_bytes(fv, state.demand_chid_list, state)

    # Tail guard
    if p.remaining() > 0:
        _log_discard(
            "time_driven_tail",
            f"msg_id=2/3 tot={rtot_s:.6f} dropping {p.remaining()} bytes after TD blocks (fv_len={fv_len})",
            every=100,
            first=20,
        )
        _ = p.read_bytes(p.remaining())

    return TimeDrivenEvent(rtot_s=rtot_s, parametrics=parametrics, per_channel=per_channel)


def _decode_waveform_payload(p: PayloadReader, state: _State, skip_wfm: bool) -> Optional[WaveformEvent]:
    """
    Decode MID 173, SubID 1 — Digital AE Waveform Data (GUIDE.md §4.1)
    
    Layout (after MID=173 byte consumed by caller):
      SUBID:uint8 == 1         sub-message ID
      TOT:bytes[6]             time of test (uint48_le, 0.25µs ticks)
      CID:uint8                AE channel number
      ALB:uint8                alignment/dummy byte
      SAMPLES:int16_le[]       remaining bytes are all waveform samples
    
    NOTE: Unlike GUIDE.md §4.1, there is NO N:uint16_le field in the observed format.
          All remaining bytes after ALB are samples. This matches the working reference.
    """
    subid = p.u8()  # byte[0]: SUBID
    if subid != 1:
        # SubID != 1 (e.g., SubID 3 = power spectrum): not decoded, skip
        if p.remaining() > 0:
            _ = p.read_bytes(p.remaining())  # skip unknown sub-message payload
        return None

    tot_s = p.rtot_seconds()  # bytes[1:7]: TOT (6 bytes)
    cid = p.u8()              # byte[7]: CID (channel number)
    _ = p.u8()                # byte[8]: ALB (alignment/dummy byte)

    # Look up SRATE and TDLY from TRA hardware config
    _ensure_hardware_recarray(state)
    channel = state.hardware[state.hardware["CH"] == cid]
    if len(channel) == 0:
        raise KeyError(f"Missing hardware config for channel {cid} (need SRATE/TDLY)")

    sr = int(channel["SRATE"][0])   # sample rate in Hz
    td = int(channel["TDLY"][0])    # trigger delay (negative = pretrigger)

    # --- SAMPLES: all remaining bytes are int16_le samples ---
    sample_bytes = p.read_bytes(p.remaining())  # SAMPLES: remaining bytes as int16_le

    if skip_wfm:
        return None

    # --- Convert raw int16 samples to voltage (volts) ---
    # Formula: V = sample * (max_input / (gain_linear * max_counts))
    # Where:
    #   max_input = 10.0 V (full-scale input voltage)
    #   max_counts = 32768 (2^15, max value of signed 16-bit)
    #   gain_linear = 10^(gain_dB / 20)
    samples = np.frombuffer(sample_bytes, dtype="<i2")  # int16_le samples

    max_input = 10.0  # full-scale input voltage (V)
    gain_db = state.gain.get(cid, 0)  # gain from MID 42 subid 23
    if cid not in state.gain:
        _log_discard(
            "missing_gain",
            f"cid={cid} missing gain (MID 42 subid 23); assuming 0 dB",
            every=200,
            first=20,
        )
    gain_lin = 10 ** (gain_db / 20.0)  # convert dB to linear
    max_counts = 32768.0               # 2^15 for signed 16-bit
    amp_scale = max_input / (gain_lin * max_counts)

    v_bytes = (amp_scale * samples.astype(np.float64)).tobytes()
    return WaveformEvent(tot_s=tot_s, cid=cid, srate=sr, tdly=td, waveform_bytes=v_bytes)


# --- Public API -----------------------------------------------------------

PathOrPaths = Union[Path, Sequence[Path]]


def iter_bin(files: PathOrPaths, skip_wfm: bool = False, *, validate: bool = True, _state: Optional[_State] = None) -> Iterable[BaseModel]:
    """
    Stream parsed events from one or more .DTA files.
    
    Per GUIDE.md §0.2, each message is encoded as:
      LEN:uint16_le           number of bytes in BODY
      BODY:bytes[LEN]         message body (BODY[0] = MID)
    
    Total bytes per message = 2 + LEN.
    
    For MIDs 40-49, the ID is followed by an extra 0x00 byte (GUIDE.md §0.2.2).
    
    Args:
        files: Path or sequence of paths to .DTA files
        skip_wfm: If True, skip waveform sample conversion (faster)
        validate: If True, use Pydantic validation; False for max throughput
        _state: Internal state object (for capturing config in read_bin)
    
    Yields:
        HitEvent, WaveformEvent, TimeDrivenEvent, or TestStartEvent
    """
    files = [files] if isinstance(files, (str, Path)) else list(files)
    files = [Path(f) for f in files]

    state = _state if _state is not None else _State()

    for path in files:
        with path.open("rb") as fp:
            sr = StreamReader(fp)

            while True:
                # --- Message envelope: LEN(2) + BODY(LEN) ---
                len_bytes = fp.read(2)  # LEN: uint16_le
                if not len_bytes:
                    break  # EOF

                if len(len_bytes) != 2:
                    raise EOFError("Truncated LEN field")

                msg_len = struct.unpack(FMT_U16, len_bytes)[0]  # LEN value
                msg_id = sr.read_u8()  # BODY[0]: MID (message ID)

                payload_len = msg_len - 1  # remaining bytes in BODY after MID
                payload = sr.read_exact(payload_len)  # BODY[1:LEN]

                # --- IDs 40-49: extra 0x00 pad byte after MID (GUIDE.md §0.2.2) ---
                if 40 <= msg_id <= 49:
                    p0 = PayloadReader(payload)
                    _ = p0.u8()  # MID_PAD: should be 0x00
                    payload = p0.read_bytes(p0.remaining())

                p = PayloadReader(payload)

                # --- MID 1: AE Hit / Event Data (GUIDE.md §3.1) ---
                if msg_id == 1:
                    ev = _decode_hit_payload(p, state)
                    yield ev if validate else ev.model_construct(**ev.model_dump())

                # --- MID 2/3: Time-Driven / User-Forced Sample Data (GUIDE.md §3.2/§3.3) ---
                elif msg_id in (2, 3):
                    ev = _decode_time_driven_payload(p, state)
                    yield ev if validate else ev.model_construct(**ev.model_dump())

                # --- MID 7: User Comments / Test Label (GUIDE.md §3.7, SPEC.md ID 7) ---
                elif msg_id == 7:
                    # Layout: TEXT:bytes[LEN-1] (ASCII, may contain NUL)
                    rem = p.remaining()
                    if rem:
                        raw = p.read_bytes(rem)
                        state.user_comment = raw.rstrip(b'\x00').decode('ascii', errors='replace')

                # --- MID 8: Continued File Marker (GUIDE.md §3.8, SPEC.md ID 8) ---
                elif msg_id == 8:
                    # Layout: DOS_TIME_DATE:bytes[8] + REMAINDER:bytes[...]
                    # Type A (end of file): REMAINDER empty
                    # Type B (start of continued file): REMAINDER contains embedded setup messages
                    rem = p.remaining()
                    _ = p.read_bytes(min(8, rem))  # DOS_TIME_DATE: 8 bytes
                    rem2 = p.remaining()
                    if rem2 > 0:
                        _log_discard(
                            "msg8_tail",
                            f"msg_id=8 dropping {rem2} bytes (REMAINDER: embedded setup messages)",
                            every=50,
                            first=50,
                        )
                        _ = p.read_bytes(rem2)  # REMAINDER: not decoded

                # --- MID 41: ASCII Product Definition (GUIDE.md §3.18, SPEC.md ID 41) ---
                elif msg_id == 41:
                    # Layout (after 0x00 pad consumed): PVERN:uint16_le + ASCII:remaining
                    _ = p.u16()  # PVERN: product version number
                    rem = p.remaining()
                    if rem:
                        raw = p.read_bytes(rem)
                        state.product_name = raw.rstrip(b'\x00').decode('ascii', errors='replace')

                # --- MID 42: Hardware Setup (GUIDE.md §3.18, SPEC.md ID 42) ---
                elif msg_id == 42:
                    _decode_hw_setup_payload(p, state)

                # --- MID 99: Time and Date of Test Start (GUIDE.md §3.18 note, SPEC.md ID 99) ---
                elif msg_id == 99:
                    ev = _decode_test_start_payload(p, state)
                    yield ev if validate else ev.model_construct(**ev.model_dump())

                # --- MID 109: Partial Power Setup (GUIDE.md §3.19) ---
                elif msg_id == 109:
                    # Layout: SEGMENT_TYPE:uint8 + N_SEG:uint16_le + RSV1 + RSV2 + segments...
                    _ = p.u8()  # SEGMENT_TYPE: typically 0
                    state.partial_power_segments = p.u16()  # N_SEG: number of segments
                    rem = p.remaining()
                    if rem:
                        _log_discard(
                            "msg109_tail",
                            f"msg_id=109 dropping {rem} bytes (RSV + per-segment metadata)",
                            every=50,
                            first=50,
                        )
                        _ = p.read_bytes(rem)  # per-segment definitions: not decoded

                # --- MID 128/129/130: Control Messages (GUIDE.md mentions Resume/Stop/Pause) ---
                elif msg_id in (128, 129, 130):
                    # Layout: RTOT:bytes[6], STATUS:uint8 (undocumented, observed: 0x01)
                    # 128 = Resume/Start, 129 = Stop, 130 = Pause
                    _ = p.rtot_seconds()  # RTOT: 6 bytes (time of control event)
                    if p.remaining() >= 1:
                        _ = p.u8()  # STATUS: undocumented flag (observed: 0x01)

                # --- MID 173: AEDSP/TRA Extended Messages (GUIDE.md §4) ---
                elif msg_id == 173:
                    # Sub-ID determines payload: 1=Waveform, 3=Power spectrum, 29=Filter, 42=HW setup
                    ev = _decode_waveform_payload(p, state, skip_wfm=skip_wfm)
                    if ev is not None:
                        yield ev if validate else ev.model_construct(**ev.model_dump())

                # --- Unknown MID: skip entire payload ---
                else:
                    rem = p.remaining()
                    # Only log if this is a truly unknown MID (not in known-unimplemented set)
                    if msg_id not in KNOWN_UNIMPLEMENTED_MIDS:
                        _log_discard(
                            "unknown_msg",
                            f"unknown msg_id={msg_id} skipping payload={rem} bytes",
                            every=500,
                            first=50,
                        )
                    _ = p.read_bytes(rem)  # unknown: not decoded


def read_bin(files: PathOrPaths, skip_wfm: bool = False):
    """
    Read AE hit summary, (optionally) waveform data, and time-driven (demand) data
    from one or more Mistras .DTA files.

    Returns:
        rec, wfm, td, config
        
    Where config is a dict containing hardware setup information:
        - test_start_time: datetime of test start
        - chid_list: tuple of CHID values for hit features
        - demand_chid_list: tuple of CHID values for time-driven features
        - demand_pid_list: tuple of PID values for time-driven parametrics
        - gain: dict of CID -> gain_dB
        - threshold: dict of CID -> threshold_dB
        - hdt: dict of CID -> hit_definition_time_us
        - hlt: dict of CID -> hit_lockout_time_us
        - pdt: dict of CID -> peak_definition_time_us
        - sampling_interval_ms: global sampling interval
        - demand_rate_ms: demand sampling rate
        - partial_power_segments: number of partial power segments
        - waveform_hardware: recarray with CH, SRATE, TDLY
    """
    logger.info(f"Starting to read DTA file(s): {files}")
    logger.info(f"skip_wfm={skip_wfm}")

    rec_rows: List[List[Any]] = []
    wfm_rows: List[List[Any]] = []
    td_rows: List[List[Any]] = []

    # Create shared state to capture hardware config
    state = _State()

    hit_count = 0
    wfm_count = 0
    td_count = 0

    # We will build stable schemas from the first events we see.
    td_pid_order: Optional[Tuple[int, ...]] = None
    td_cid_order: Optional[Tuple[int, ...]] = None
    td_fv_keys: Optional[Tuple[str, ...]] = None  # feature names within each CID's FV
    hit_param_order: Optional[Tuple[int, ...]] = None  # parametric PIDs for hits

    for ev in iter_bin(files, skip_wfm=skip_wfm, validate=True, _state=state):
        if isinstance(ev, TestStartEvent):
            state.test_start_time = ev.test_start_time

        elif isinstance(ev, HitEvent):
            hit_count += 1
            # Capture parametric schema from first hit with parametrics
            if hit_param_order is None and ev.parametrics:
                hit_param_order = tuple(ev.parametrics.keys())
            # Build row: features + parametrics (aligned to first-seen schema)
            feat_vals = [ev.features[k] for k in ev.features.keys()]
            param_vals = [ev.parametrics.get(pid, None) for pid in (hit_param_order or ())]
            row = [ev.rtot_s, ev.cid] + feat_vals + param_vals
            rec_rows.append(row)

        elif isinstance(ev, WaveformEvent):
            wfm_count += 1
            wfm_rows.append([ev.tot_s, ev.cid, ev.srate, ev.tdly, ev.waveform_bytes])

        elif isinstance(ev, TimeDrivenEvent):
            td_count += 1

            if td_pid_order is None:
                td_pid_order = tuple(ev.parametrics.keys())
            if td_cid_order is None:
                td_cid_order = tuple(sorted(ev.per_channel.keys()))
            # Capture FV field names from first channel's decoded dict
            if td_fv_keys is None and ev.per_channel:
                first_cid = next(iter(ev.per_channel))
                td_fv_keys = tuple(ev.per_channel[first_cid].keys())

            # Align to first-seen schema; log drift but keep going.
            if td_pid_order != tuple(ev.parametrics.keys()):
                _log_discard(
                    "td_pid_drift",
                    f"time-driven PID keys changed: first={td_pid_order} now={tuple(ev.parametrics.keys())}",
                    every=50,
                    first=20,
                )
            if td_cid_order != tuple(sorted(ev.per_channel.keys())):
                _log_discard(
                    "td_cid_drift",
                    f"time-driven CID keys changed: first={td_cid_order} now={tuple(sorted(ev.per_channel.keys()))}",
                    every=50,
                    first=20,
                )

            pid_vals = [ev.parametrics.get(pid, None) for pid in (td_pid_order or ())]
            # Flatten per-channel decoded dicts: for each CID, extract values in td_fv_keys order
            cid_vals = []
            for cid in (td_cid_order or ()):
                fv_dict = ev.per_channel.get(cid, {})
                for key in (td_fv_keys or ()):
                    cid_vals.append(fv_dict.get(key, None))
            td_rows.append([ev.rtot_s] + pid_vals + cid_vals)

        if hit_count and hit_count % 1000 == 0:
            logger.debug(f"Processed {hit_count} hit records")
        if wfm_count and wfm_count % 100 == 0:
            logger.debug(f"Processed {wfm_count} waveform records")
        if td_count and td_count % 1000 == 0:
            logger.debug(f"Processed {td_count} time-driven records")

    logger.info(
        f"Finished reading file(s). Found {len(rec_rows)} hit records, "
        f"{len(wfm_rows)} waveform records, and {len(td_rows)} time-driven records"
    )

    rec = rec_rows
    wfm = wfm_rows
    td = td_rows

    if rec_rows:
        schema_chids: Tuple[int, ...] = ()
        name_to_chid = {v: k for k, v in CHID_to_str.items()}

        for ev2 in iter_bin(files, skip_wfm=True, validate=False):
            if isinstance(ev2, HitEvent):
                schema_chids = tuple(name_to_chid[n] for n in ev2.features.keys())
                break

        # Build column names: time, channel, features, then parametrics
        feat_names = [CHID_to_str[i] for i in schema_chids]
        param_names = [f"PARAM_{pid}" for pid in (hit_param_order or ())]
        names = ["SSSSSSSS.mmmuuun", "CH"] + feat_names + param_names
        rec = np.rec.fromrecords(rec_rows, names=names)

        if state.test_start_time is None:
            raise ValueError("Missing test start time (msg 99); cannot compute TIMESTAMP")

        ts = [(state.test_start_time + timedelta(seconds=t)).timestamp() for t in rec["SSSSSSSS.mmmuuun"]]
        rec = append_fields(rec, "TIMESTAMP", ts, usemask=False, asrecarray=True)

    if wfm_rows:
        wfm = np.rec.fromrecords(
            wfm_rows, names=["SSSSSSSS.mmmuuun", "CH", "SRATE", "TDLY", "WAVEFORM"]
        )

    if td_rows:
        pid_cols = []
        cid_cols = []
        if td_pid_order is not None:
            pid_cols = [f"PID_{pid}" for pid in td_pid_order]
        # Build flattened column names: CID{n}_{feature_name} for each CID and feature
        if td_cid_order is not None and td_fv_keys is not None:
            for cid in td_cid_order:
                for key in td_fv_keys:
                    cid_cols.append(f"CID{cid}_{key}")

        td_names = ["SSSSSSSS.mmmuuun"] + pid_cols + cid_cols
        td = np.rec.fromrecords(td_rows, names=td_names)

    # Build config dict from collected hardware state
    _ensure_hardware_recarray(state)
    config = {
        "test_start_time": state.test_start_time,
        "product_name": state.product_name,
        "user_comment": state.user_comment,
        "chid_list": state.chid_list,
        "demand_chid_list": state.demand_chid_list,
        "demand_pid_list": state.demand_pid_list,
        "gain": dict(state.gain) if state.gain else {},
        "threshold": dict(state.threshold) if state.threshold else {},
        "hdt": dict(state.hdt) if state.hdt else {},
        "hlt": dict(state.hlt) if state.hlt else {},
        "pdt": dict(state.pdt) if state.pdt else {},
        "sampling_interval_ms": state.sampling_interval_ms,
        "demand_rate_ms": state.demand_rate_ms,
        "partial_power_segments": state.partial_power_segments,
        "waveform_hardware": state.hardware,
    }

    return rec, wfm, td, config


def get_waveform_data(wfm_row):
    """Return time (microseconds) and voltage arrays from a waveform record."""
    v = np.frombuffer(wfm_row["WAVEFORM"])
    t = 1e6 * (np.arange(0, len(v)) + wfm_row["TDLY"]) / wfm_row["SRATE"]
    return t, v
