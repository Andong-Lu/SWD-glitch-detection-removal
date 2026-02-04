# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 21:48:08 2026

@author: Andong Lu
"""

# run_deglitch.py
"""
Run this script to detect and remove glitches from InSight SEIS data using SWD.

Signal relationship:
  raw x(t) = deglitched y(t) + removed model g(t)
  y(t) = x(t) - g(t)

Outputs (in ./output):
  - deglitched_data.mseed          : deglitched traces y(t)
  - output/<CH>/glitches_time.txt  : merged glitch intervals (UTC)
"""

import os
from swd_deglitch import DeglitchConfig, deglitch_mseed_file

if __name__ == "__main__":
    # -----------------------
    # Paths (relative to this file)
    # -----------------------
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")

    # Put your input file in ./data/
    MSEED_FILE = os.path.join(
        DATA_DIR,
        "XB.ELYSE.03.BH__20190522T2318_20190523T2359_raw.mseed"
    )

    CHANNELS = ["BHU", "BHV", "BHW"]

    # -----------------------
    # Parameters (edit here only)
    # -----------------------
    cfg = DeglitchConfig(
        window_len=600,
        overlap=100,
        edge_guard=50,
        dur_criterion=50,
        amp_criterion=5,
        max_iter_glitch=10,
        max_iter_spike=1,
        err_tol=0.1,
        energy_lo=0.0005,
        energy_hi=0.9995,
        spike_pad_before=1.0,
        spike_pad_after=1.0,
        spike_mask_halfwidth=1.0,
    )

    # -----------------------
    # Run
    # -----------------------
    results = deglitch_mseed_file(
        mseed_path=MSEED_FILE,
        out_dir=OUTPUT_DIR,
        channels=CHANNELS,
        cfg=cfg,
        out_clean_name="deglitched_data.mseed",
    )

    print("Done. Processed channels:", list(results.keys()))
    print("Outputs written to:", OUTPUT_DIR)
