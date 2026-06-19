# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 22:28:37 2026

@author: Andong Lu

Process all MiniSEED files in data/.
"""
#%%
import os
from multiprocessing import freeze_support
from pathlib import Path

# Limit numerical threads in each worker.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from swd_deglitch import DeglitchConfig, deglitch_mseed_file

#%%
if __name__ == "__main__":
    freeze_support()

    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = ["BHU", "BHV", "BHW"]

    # Parameters
    cfg = DeglitchConfig(
        window_len=600,
        overlap=100,
        edge_guard=50,
        dur_criterion=50,
        amp_criterion=5,
        max_iter_glitch=10,
        max_iter_spike=1,
        err_tol=0.1,
        parallel_windows=True,
        max_window_workers=5,
    )

    mseed_files = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mseed"
    )
    if not mseed_files:
        raise FileNotFoundError(f"No .mseed files found in {data_dir}")

    failures = []
    for file_number, mseed_file in enumerate(mseed_files, start=1):
        file_out_dir = output_dir / mseed_file.stem
        clean_name = f"{mseed_file.stem}_deglitched.mseed"

        print(f"\n[{file_number}/{len(mseed_files)}] Processing {mseed_file.name}")
        try:
            deglitch_mseed_file(
                mseed_path=str(mseed_file),
                out_dir=str(file_out_dir),
                channels=channels,
                cfg=cfg,
                out_clean_name=clean_name,
            )
        except Exception as exc:
            failures.append((mseed_file.name, str(exc)))
            print(f"Failed: {exc}")

    print(f"\nCompleted {len(mseed_files) - len(failures)} of {len(mseed_files)} files.")
    if failures:
        print("Failed files:")
        for filename, error in failures:
            print(f"  {filename}: {error}")
