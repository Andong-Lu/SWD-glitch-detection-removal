# SWD-based Glitch Detection and Removal

This repository provides an implementation of a Shock Waveform Decomposition (SWD)
based method for detecting and removing glitches and onset spikes in planetary seismic data.

The method is described in:

> Andong Lu, Qingming Li (2026), *A New Framework for the Detection and Removal of Glitches in InSight Seismic Data*, submitted to *Seismological Research Letters*.

## Method overview

The recorded signal \( x(t) \) is decomposed into:
- glitch and spike components \( g(t) \)
- remaining seismic signal \( y(t) \)

The deglitched signal is obtained as:
\[
y(t) = x(t) - g(t)
\]

## Repository structure

- `run_deglitch.py` — main entry script (user-editable parameters)
- `swd_deglitch.py` — SWD-based detection and removal algorithm
- `SWTools.py` — SWD implementation
- `data/` — placeholder (raw data not included)
- `output/` — generated outputs

## Data

This code was developed for InSight SEIS data.
Raw seismic data are available from:

https://datacenter.ipgp.fr/networks/detail/XB_2016/

## Requirements

Python ≥ 3.9

Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

Place the input MiniSEED file in the data/ directory, then run:

```bash
python run_deglitch.py
```

The deglitched seismic records and detected glitch time intervals will be written to the output/ directory.



## License

This project is released under the MIT License.
