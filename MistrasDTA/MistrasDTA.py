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


# --- Feature metadata (kept identical to original behavior) -------------------

CHID_to_str = {
    1: "RISE",
    2: "PCNTS",
    3: "COUN",
    4: "ENER",
    5: "DURATION",
    6: "AMP",
    8: "ASL",
    10: "THR",
    13: "A-FRQ",
    17: "RMS",
    18: "R-FRQ",
    19: "I-FRQ",
    20: "SIG STRENGTH",
    21: "ABS-ENERGY",
    22: "PARTIAL POWER",
    23: "FRQ-C",
    24: "P-FRQ",
    31: "UNKNOWN",
}

CHID_byte_len = {
    1: 2,
    2: 2,
    3: 2,
    4: 2,
    5: 4,
    6: 1,
    8: 1,
    10: 1,
    13: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 4,
    21: 4,
    22: 0,  # variable-length; number of bytes comes from msg 109
    23: 2,
    24: 2,
    31: 2,
}


# --- Low-level formats --------------------------------------------------------

FMT_U8 = "<B"
FMT_U16 = "<H"
FMT_I16 = "<h"
FMT_I32 = "<i"
FMT_F32 = "<f"
FMT_RTOT = "<IH"  # 6 bytes: u32 + u16


def _rtot_to_seconds(u32: int, u16: int) -> float:
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
            # fail fast: caller expects well-formed input
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


# --- Pydantic models (validation/normalization layer) --------------------------

class HitEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rtot_s: float
    cid: int
    features: Dict[str, Any]


class WaveformEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tot_s: float
    cid: int
    srate: int
    tdly: int
    waveform_bytes: bytes  # float64 bytes (same as passing version)


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

    def __post_init__(self) -> None:
        if self.gain is None:
            self.gain = {}
        if self.hardware_rows is None:
            self.hardware_rows = []


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
            dtype=[("CH", np.int32), ("SRATE", np.int32), ("TDLY", np.int32)],
        )


# --- Message decoders (payload -> event or state mutation) ---------------------

def _decode_hit_payload(p: PayloadReader, state: _State) -> HitEvent:
    # msg 1: RTOT(6), CID(1), then feature vector per state.chid_list, then optional parametrics
    rtot_s = p.rtot_seconds()
    cid = p.u8()

    features: Dict[str, Any] = {}

    for chid in state.chid_list:
        if chid not in CHID_to_str:
            # If we ever get here, we will likely desync; log loudly and fail naturally.
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
            v = p.read_bytes(n)  # bytes length n (or b'' if n==0)
        else:
            if b == 1:
                v = p.u8()
            elif b == 2:
                v = p.u16()
            elif b == 4:
                # Minimal: fail fast for unknown 4-byte types not explicitly handled above
                v = p.read_bytes(4)
            else:
                v = p.read_bytes(b)

        features[name] = v

    # Ignore any remaining (parametrics)
    rem = p.remaining()
    if rem > 0:
        _log_discard(
            "hit_parametrics",
            f"msg_id=1 dropping {rem} trailing bytes after feature vector (hit parametrics not decoded)",
        )
        _ = p.read_bytes(rem)

    return HitEvent(rtot_s=rtot_s, cid=cid, features=features)


def _decode_hw_setup_payload(p: PayloadReader, state: _State) -> None:
    # msg 42: MVERN(u16) then subrecords until payload end
    _ = p.u16()  # MVERN

    while p.remaining() > 0:
        lsub = p.u16()
        sub_len = lsub
        subpayload = PayloadReader(p.read_bytes(lsub))

        subid = subpayload.u8()

        handled = False

        if subid == 5:
            handled = True
            # Event Data Set Definition: CHID list
            n = subpayload.u8()
            chids = subpayload.unpack(f"<{n}B")
            state.chid_list = tuple(int(x) for x in chids)

            unknown = [c for c in state.chid_list if c not in CHID_to_str]
            if unknown:
                _log_discard(
                    "unknown_chid_list",
                    f"Event Data Set Definition contains unknown CHIDs: {unknown}",
                    every=1,
                    first=50,
                )

        elif subid == 23:
            handled = True
            # Set Gain
            cid = subpayload.u8()
            gain_db = subpayload.u8()
            state.gain[cid] = gain_db

        elif subid == 173:
            # Nested hardware record
            subid2 = subpayload.u8()
            if subid2 == 42:
                handled = True
                # Parse only the fields needed for waveform scaling/time axis.
                _ = subpayload.read_bytes(2)  # MVERN, b2
                _ = subpayload.read_bytes(1)  # ADT
                _ = subpayload.read_bytes(2)  # SETS, b2
                _ = subpayload.read_bytes(2)  # SLEN
                ch = subpayload.u8()          # channel id
                _ = subpayload.read_bytes(2)  # HLK
                _ = subpayload.read_bytes(2)  # HITS
                srate = subpayload.u16()
                _ = subpayload.read_bytes(2)  # TMODE
                _ = subpayload.read_bytes(2)  # TSRC
                tdly = subpayload.i16()
                _ = subpayload.read_bytes(2)  # MXIN
                _ = subpayload.read_bytes(2)  # THRD

                state.hardware_rows.append([ch, 1000 * srate, tdly])

        if not handled:
            _log_discard(
                "hw_subrecord",
                f"msg_id=42 skipping unhandled subrecord subid={subid} len={sub_len} bytes",
                every=200,
                first=20,
            )

        # Log leftover bytes even for handled records (partial decode)
        left = subpayload.remaining()
        if left > 0:
            _log_discard(
                "hw_subrecord_tail",
                f"msg_id=42 subid={subid} leaving {left} unconsumed bytes (fields not decoded)",
                every=200,
                first=20,
            )
            _ = subpayload.read_bytes(left)

    # Refresh hardware view
    state.hardware = None
    _ensure_hardware_recarray(state)


def _decode_test_start_payload(p: PayloadReader, state: _State) -> TestStartEvent:
    raw = p.read_bytes(p.remaining())
    s = raw.decode("ascii").strip("\x00")
    dt = datetime.strptime(s, "%a %b %d %H:%M:%S %Y\n")
    state.test_start_time = dt
    return TestStartEvent(test_start_time=dt)


def _decode_waveform_payload(p: PayloadReader, state: _State, skip_wfm: bool) -> Optional[WaveformEvent]:
    # msg 173: SUBID(u8), TOT(6), CID(u8), ALB(u8), then remaining is int16 samples
    _ = p.u8()  # subid
    tot_s = p.rtot_seconds()
    cid = p.u8()
    _ = p.u8()  # ALB

    # Remaining payload is samples (note: spec may also allow AEF tail; we currently ignore it)
    sample_bytes = p.read_bytes(p.remaining())
    n = len(sample_bytes) // 2
    samples = struct.unpack(f"<{n}h", sample_bytes)

    if skip_wfm:
        return None

    # Scale factor matches passing version
    max_input = 10.0
    gain_db = state.gain[cid]
    gain_lin = 10 ** (gain_db / 20.0)
    max_counts = 32768.0
    amp_scale = max_input / (gain_lin * max_counts)

    _ensure_hardware_recarray(state)
    channel = state.hardware[state.hardware["CH"] == cid]
    sr = int(channel["SRATE"][0])
    td = int(channel["TDLY"][0])

    v_bytes = (amp_scale * np.array(samples)).tobytes()
    return WaveformEvent(tot_s=tot_s, cid=cid, srate=sr, tdly=td, waveform_bytes=v_bytes)


# --- Public API -----------------------------------------------------------

PathOrPaths = Union[Path, Sequence[Path]]


def iter_bin(files: PathOrPaths, skip_wfm: bool = False, *, validate: bool = True) -> Iterable[BaseModel]:
    """
    Stream parsed events from one or more .DTA files.
    - If 'files' is a path-like, it is treated as a single path.
    - If 'files' is a sequence, paths are processed in that order.
    validate=True keeps pydantic validation on; set False for maximum throughput.
    """
    files = [files] if isinstance(files, (str, Path)) else list(files)
    files = [Path(f) for f in files]

    state = _State()

    for path in files:
        with path.open("rb") as fp:
            sr = StreamReader(fp)

            while True:
                len_bytes = fp.read(2)
                if not len_bytes:
                    break

                if len(len_bytes) != 2:
                    raise EOFError("Truncated LEN field")

                msg_len = struct.unpack(FMT_U16, len_bytes)[0]
                msg_id = sr.read_u8()

                payload_len = msg_len - 1
                payload = sr.read_exact(payload_len)

                # IDs 40–49 have an extra byte (ignored) inside payload:
                if 40 <= msg_id <= 49:
                    p0 = PayloadReader(payload)
                    _ = p0.u8()  # ext id
                    payload = p0.read_bytes(p0.remaining())

                p = PayloadReader(payload)

                if msg_id == 1:
                    ev = _decode_hit_payload(p, state)
                    yield ev if validate else ev.model_construct(**ev.model_dump())

                elif msg_id == 7:
                    # User comment (ignored in current outputs)
                    rem = p.remaining()
                    if rem:
                        _log_discard("msg7_comment", f"msg_id=7 dropping {rem} bytes (user comment not decoded)", every=500, first=20)
                        _ = p.read_bytes(rem)

                elif msg_id == 8:
                    # Continued file marker: we currently ignore the embedded setup record if present.
                    rem = p.remaining()
                    _ = p.read_bytes(min(8, rem))
                    rem2 = p.remaining()
                    if rem2 > 0:
                        _log_discard("msg8_tail", f"msg_id=8 dropping {rem2} bytes after DOS time/date (embedded setup not decoded)", every=50, first=50)
                        _ = p.read_bytes(rem2)

                elif msg_id == 41:
                    # ASCII Product Definition: PVERN(u16) then ignore remainder
                    _ = p.u16()
                    rem = p.remaining()
                    if rem:
                        _log_discard("msg41_ascii", f"msg_id=41 dropping {rem} bytes (ASCII product definition not decoded)", every=50, first=50)
                        _ = p.read_bytes(rem)

                elif msg_id == 42:
                    _decode_hw_setup_payload(p, state)

                elif msg_id == 99:
                    ev = _decode_test_start_payload(p, state)
                    yield ev if validate else ev.model_construct(**ev.model_dump())

                elif msg_id == 109:
                    # Partial Power setup: segment_type(u8) then num_segments(u16)
                    _ = p.u8()  # segment_type
                    state.partial_power_segments = p.u16()
                    rem = p.remaining()
                    if rem:
                        _log_discard("msg109_tail", f"msg_id=109 dropping {rem} bytes (per-segment metadata not decoded)", every=50, first=50)
                        _ = p.read_bytes(rem)

                elif msg_id in (128, 129, 130):
                    # Control events: RTOT(6) (ignore rest)
                    _ = p.rtot_seconds()
                    rem = p.remaining()
                    if rem:
                        _log_discard("control_tail", f"msg_id={msg_id} dropping {rem} bytes after RTOT (extra payload not decoded)", every=200, first=20)
                        _ = p.read_bytes(rem)

                elif msg_id == 173:
                    # Note: we treat remaining payload as samples; spec may allow trailing AEF copy.
                    # If present, we'd be mis-decoding. Log if sample_bytes length is odd.
                    if p.remaining() % 2 != 0:
                        _log_discard("wfm_odd", f"msg_id=173 payload remaining {p.remaining()} is odd; may indicate trailing non-sample data", every=20, first=20)
                    ev = _decode_waveform_payload(p, state, skip_wfm=skip_wfm)
                    if ev is not None:
                        yield ev if validate else ev.model_construct(**ev.model_dump())

                else:
                    # Unknown message: ignore payload (INFO, rate-limited)
                    rem = p.remaining()
                    _log_discard(
                        "unknown_msg",
                        f"unknown msg_id={msg_id} skipping payload={rem} bytes",
                        every=500,
                        first=50,
                    )
                    _ = p.read_bytes(rem)


def read_bin(files: PathOrPaths, skip_wfm: bool = False):
    """Read AE hit summary and (optionally) waveform data from one or more Mistras .DTA files."""
    logger.info(f"Starting to read DTA file(s): {files}")
    logger.info(f"skip_wfm={skip_wfm}")

    rec_rows: List[List[Any]] = []
    wfm_rows: List[List[Any]] = []
    state_test_start: Optional[datetime] = None

    hit_count = 0
    wfm_count = 0

    for ev in iter_bin(files, skip_wfm=skip_wfm, validate=True):
        if isinstance(ev, TestStartEvent):
            state_test_start = ev.test_start_time

        elif isinstance(ev, HitEvent):
            hit_count += 1
            row = [ev.rtot_s, ev.cid] + [ev.features[k] for k in ev.features.keys()]
            rec_rows.append(row)

        elif isinstance(ev, WaveformEvent):
            wfm_count += 1
            wfm_rows.append([ev.tot_s, ev.cid, ev.srate, ev.tdly, ev.waveform_bytes])

        if hit_count and hit_count % 1000 == 0:
            logger.debug(f"Processed {hit_count} hit records")
        if wfm_count and wfm_count % 100 == 0:
            logger.debug(f"Processed {wfm_count} waveform records")

    logger.info(f"Finished reading file(s). Found {len(rec_rows)} hit records and {len(wfm_rows)} waveform records")

    rec = rec_rows
    wfm = wfm_rows

    if rec_rows:
        # Capture schema CHID order from the first hit across all files (cheap second pass).
        schema_chids: Tuple[int, ...] = ()
        name_to_chid = {v: k for k, v in CHID_to_str.items()}

        for ev2 in iter_bin(files, skip_wfm=True, validate=False):
            if isinstance(ev2, HitEvent):
                schema_chids = tuple(name_to_chid[n] for n in ev2.features.keys())
                break

        names = ["SSSSSSSS.mmmuuun", "CH"] + [CHID_to_str[i] for i in schema_chids]
        rec = np.rec.fromrecords(rec_rows, names=names)

        if state_test_start is None:
            raise ValueError("Missing test start time (msg 99); cannot compute TIMESTAMP")

        ts = [(state_test_start + timedelta(seconds=t)).timestamp() for t in rec["SSSSSSSS.mmmuuun"]]
        rec = append_fields(rec, "TIMESTAMP", ts, usemask=False, asrecarray=True)

    if wfm_rows:
        wfm = np.rec.fromrecords(
            wfm_rows, names=["SSSSSSSS.mmmuuun", "CH", "SRATE", "TDLY", "WAVEFORM"]
        )

    return rec, wfm


def get_waveform_data(wfm_row):
    """Return time (microseconds) and voltage arrays from a waveform record."""
    v = np.frombuffer(wfm_row["WAVEFORM"])
    t = 1e6 * (np.arange(0, len(v)) + wfm_row["TDLY"]) / wfm_row["SRATE"]
    return t, v
