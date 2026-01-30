# AEwin Hit Message Feature IDs and Scaling (AEwin 32-bit Notes)

This document reformats the supplied PDF into Markdown for quick reference. It focuses on:
- Hit message feature IDs (CHIDs), their stored types, and scaling
- The ordering rule between the “hit definition” message and the hit data message
- The partial power feature (CHID 22) and the partial power setup message (ID 109)
- Worked examples showing byte layouts

Source: :contentReference[oaicite:0]{index=0}

---

## 1) Hit Message Feature IDs (decimal), types, units, scaling

> In the original notes, `p` is a pointer to the first byte of the parameter within the AE hit data message (ID 1).

### RISETIME (ID 1)
- Stored: 2 bytes, **unsigned short**
- Units: **microseconds**
- Decode: `u16`

### COUNTS TO PEAK (ID 2)
- Stored: 2 bytes, **unsigned short**
- Units: none
- Decode: `u16`

### COUNTS (ID 3)
- Stored: 2 bytes, **unsigned short**
- Units: none
- Decode: `u16`

### ENERGY (ID 4)
- Stored: 2 bytes, **unsigned short**
- Units: none
- Decode: `u16`

### DURATION (ID 5)
- Stored: 4 bytes, **unsigned long**
- Units: **microseconds**
- Decode: `u32` / `i32` (reader-specific; many implementations treat as unsigned)

### AMPLITUDE (ID 6)
- Stored: 1 byte, **unsigned char**
- Units: **dB**
- Decode: `u8`

### RMS (ID 7) — 8-bit, obsolete
- Stored: 1 byte, **unsigned char**
- Units: **volts**
- Scaling: `float(*p) / 20.0` (scaled to ≤ 7.07V)
- Decode: `u8`, then scale

### ASL (ID 8)
- Stored: 1 byte, **unsigned char**
- Units: **dB**
- Decode: `u8`

### THRESHOLD (ID 10)
- Stored: 1 byte, **unsigned char**
- Units: **dB**
- Decode: `u8`

### AVERAGE FREQUENCY (ID 13)
- Stored: 2 bytes, **unsigned short**
- Units: **kHz**
- Decode: `u16`

### RMS16 (ID 17)
- Replaces 8-bit RMS in older systems
- Stored: 2 bytes, **unsigned short**
- Scaling: `float(u16) / 5000.0`
- Decode: `u16`, then scale

### REVERBERATION FREQUENCY (ID 18)
- Stored: 2 bytes, **unsigned short**
- Units: **kHz**
- Decode: `u16`

### INITIATION FREQUENCY (ID 19)
- Stored: 2 bytes, **unsigned short**
- Units: **kHz**
- Decode: `u16`

### SIGNAL STRENGTH (ID 20)
- Stored: 4 bytes, **unsigned long**
- Scaling: multiply by **3.05** to convert A/D units → pVs
- Decode: `u32`, then scale

### ABSOLUTE ENERGY (ID 21)
- Stored: 4 bytes, **float**
- Scaling: multiply by **931e-6** (0.000931) to convert to milli-attoJoules
- Decode: `f32`, then scale

### PARTIAL POWER (ID 22)
- Meaning: indicates presence of **one or more partial powers** in the data set
- Dependency: must process **message 109** to know which ones / how many segments are present
- More details in §3

### FREQUENCY CENTROID (ID 23)
- Stored: 2 bytes, **unsigned short**
- Units: **kHz**
- Decode: `u16`

### FREQUENCY PEAK (ID 24)
- Stored: 2 bytes, **unsigned short**
- Units: **kHz**
- Decode: `u16`

---

## 2) Ordering rule: Message 5 vs Hit Data Message (ID 1)

- The order in which features are defined in **message 5** is the same order that they occur in the **hit data message** (ID 1).
- Therefore, message 5 provides the **schema/order** for decoding the variable portion of message 1.

---

## 3) Partial Power behavior (CHID 22) and offset warning

When partial powers are enabled:
- **message 109** determines **which and how many segments** are present in the data set.
- If you use message 5 to compute **byte offsets** into the hit message:
  - You must re-adjust offsets for all features that occur **after CHID 22**
  - Adjustment amount is **one byte per partial power segment defined**

> Practical implication: decoding “in order while consuming bytes” is simpler than offset arithmetic.

---

## 4) Partial Power Setup Message (ID 109)

Layout (as described):

- `u16` message length
- `u8` message id (= 109 = 0x6D)
- `u8` segment type (always 0)
- `u16` number of segments defined (if 0, then no partial powers defined)
- `u16` reserved (always 1)
- `u16` reserved (always 1)
- Repeated per segment (up to 4 segments):
  - `u16` segment number (0 to 3)
  - `u16` segment start (not required for hit processing)
  - `u16` segment end (not required for hit processing)

For hit decoding, only the **number of segments defined** is required.

---

## 5) Example: Message 5 (Hit definition) bytes and interpretation

Example bytes from message 5:

```

0B 00 05 08 01 03 04 05 06 15 16 17 01

```

Interpretation:
- First 2 bytes: message length (little-endian) = `0x000B` = **11**
- Next byte: message id = **5** (hit definition)
- Next `Length-1` bytes are body
- Body indicates **8 AE feature ids**:
  - 1  = risetime
  - 3  = counts
  - 4  = energy
  - 5  = duration
  - 6  = amplitude
  - 21 = absolute energy
  - 22 = partial powers
  - 23 = frequency centroid
- Last byte: number of **parametric values** in the hit message = **1**
  - (note: this is count, not the parametric ID)

Because partial powers are defined (CHID 22 present), **message 109** also needs to be processed.

---

## 6) Example: Message 109 bytes and interpretation

Example bytes:

```

24 00 6D 00 04 00 00 00 00 00 03 00 01 00 04 00 09 00 02 00
0A 00 0D 00 03 00 0E 00 13 00 01 00 01 00 00 00 13 00

```

Interpretation:
- message length = `0x0024` = **36**
- message id = `0x6D` = **109**
- segment type = **0**
- number of segments defined = `0x0004` = **4**

The remaining per-segment info is not required to decode hit message contents (only the segment count is needed).

Also shown:
- total power number of segments = 1
- total power segment number = 1
- total power segment start = 0
- total power segment end = 19

---

## 7) Example: Hit message (ID 1) line display + bytes

Line display:

```

1 0 00:00:04.8508117 0.0095 98 16 31 108 70 46.878E+03
0 0 1 98 192

```

Bytes for a hit message:

```

20 00 01 EF 11 28 01 00 00 01 62 00 10 00 1F 00 6C 00 00 00
46 1C 14 40 4C 00 00 01 62 C0 00 01 1F 00

```

Interpretation:
- First 2 bytes: message length = `0x0020` = **32**
- Next byte: message id = **1** (AE hit)
- Next `Length-1` bytes are body

Body fields:
1) **Time of test** (RTOT), 6 bytes:
   - `EF 11 28 01 00 00`
2) **Channel ID** (CID), 1 byte:
   - `01` (channel 1)

3) Features, in the order defined by message 5:
- Risetime = `98` from `62 00`
- Counts = `16` from `10 00`
- Energy = `31` from `1F 00`
- Duration = `108` from `6C 00 00 00`
- Amplitude = `70` from `46`
- Absolute energy = `46.878E+03` from `1C 14 40 4C`
  - computed as `5.035224E7 * 0.000931`
- Partial powers = `0, 0, 1, 98` from `00 00 01 62` (4 segments)
- Frequency centroid = `192` from `C0 00`

4) Parametric:
- Parametric 1 = `0.0095` from `1F 00`
  - computed as `31 * 10 / 32768`

---
