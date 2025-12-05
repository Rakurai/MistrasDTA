[![release](https://img.shields.io/github/v/release/d-cogswell/MistrasDTA)](https://github.com/d-cogswell/MistrasDTA/releases)
[![NewareNDA regression tests](https://github.com/d-cogswell/MistrasDTA/actions/workflows/tests.yml/badge.svg)](https://github.com/d-cogswell/MistrasDTA/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/d-cogswell/MistrasDTA/badge.svg?branch=development)](https://coveralls.io/github/d-cogswell/MistrasDTA?branch=development)

# MistrasDTA
Python module to read acoustic emissions hit data and waveforms from Mistras DTA files. The structure of these binary files is detailed in Appendix II of the Mistras user manual.

# Installation
MistrasDTA can be installed from PyPI with the following command:
```
python -m pip install MistrasDTA
```

# Usage
Read the hit summary table from a DTA file:
```
import MistrasDTA
rec, _ = MistrasDTA.read_bin('cluster.DTA', skip_wfm=True)

```

Read hit summary and waveform data from a DTA:
```
import MistrasDTA
from numpy.lib.recfunctions import join_by

# Read the binary file and join summary and waveform tables
rec, wfm = MistrasDTA.read_bin('cluster.DTA')
merged = join_by(['SSSSSSSS.mmmuuun', 'CH'], rec, wfm)

# Extract the first waveform in units of microseconds and volts
t, V = MistrasDTA.get_waveform_data(merged[0])
```
