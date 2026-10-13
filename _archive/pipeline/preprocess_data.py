
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.paths import bootstrap; bootstrap()
from lib import paths
import argparse
import os
import pandas as pd
import numpy as np

from lib.corpus_layout import output_path

# Each session in the raw dataset holds the same 10 recordings twice, under
# two folders with IDENTICAL filenames:
#   ica_eeg/  ICA-cleaned, in volts   <- the only source this pipeline uses
#   pp_eeg/   pre-ICA (filtered only), in microvolts
# This is deliberately not configurable. An earlier version walked the whole
# dataset and wrote to a flat folder, so pp_eeg (visited second) silently
# overwrote every ICA-cleaned trace and the entire pipeline ran on pre-ICA
# data without any error. Hard-wiring the source makes that unrepeatable.
SOURCE_SUBDIR = "ica_eeg"

# Scalp EEG sits in the tens of microvolts. Anything whose 99th percentile is
# below this is being reported in volts and needs scaling up by 1e6.
_VOLT_DETECT_CEILING = 1e-3


def volts_to_microvolts_scale(values):
    """1.0 if the array is already in microvolts, 1e6 if it is in volts."""
    p99 = np.nanpercentile(np.abs(values), 99)
    if p99 == 0 or np.isnan(p99):
        return 1.0
    return 1e6 if p99 < _VOLT_DETECT_CEILING else 1.0


def process_eeg_for_telex(input_root, output_folder, sampling_rate=256):

    os.makedirs(output_folder, exist_ok=True)

    csv_files = []

    # Only ever descend into ica_eeg/. Never collect from any sibling folder.
    for root, dirs, files in os.walk(input_root):
        if os.path.basename(root) != SOURCE_SUBDIR:
            continue
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    # Belt and braces: nothing outside ica_eeg/ may reach the output.
    stray = [p for p in csv_files
             if os.path.basename(os.path.dirname(p)) != SOURCE_SUBDIR]
    assert not stray, f"non-{SOURCE_SUBDIR} files collected: {stray[:3]}"

    if not csv_files:
        print(f"No CSV files found under '{input_root}/*/*/{SOURCE_SUBDIR}'.")
        return

    print(f"Found {len(csv_files)} CSV files in {SOURCE_SUBDIR}/.")
    print("--- Starting EEG Preprocessing ---")

    for file_path in csv_files:

        file_name = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)

            # --------------------------------------------------
            # Extract trial between markers
            # --------------------------------------------------
            start_matches = df[df["marker"] == 1000.0]
            end_matches = df[df["marker"] == 1001.0]

            if start_matches.empty or end_matches.empty:
                print(
                    f"  -> Skipped {file_name}: "
                    f"Missing 1000.0 or 1001.0 marker."
                )
                continue

            start_idx = start_matches.index[0]
            end_idx = end_matches.index[0]

            window_df = df.iloc[start_idx:end_idx + 1].copy()

            # --------------------------------------------------
            # Time column
            # --------------------------------------------------
            time_array = np.arange(len(window_df)) / float(sampling_rate)

            telex_df = pd.DataFrame({
                "time": time_array
            })

            eeg_channels = [
                "Fp1",
                "Fp2",
                "F3",
                "Fz",
                "F4",
                "T3",
                "Cz",
                "T4",
            ]

            # --------------------------------------------------
            # Scale to microvolts
            # The two source folders ship different units: ica_eeg is in
            # volts, pp_eeg in microvolts. Everything downstream (the uV
            # ceilings in slidingwindow.py, the thresholds in
            # SELECTED_RULES) assumes microvolts, so normalise here.
            # --------------------------------------------------
            present = [c for c in eeg_channels if c in window_df.columns]
            scale = volts_to_microvolts_scale(window_df[present].to_numpy())
            for col in present:
                telex_df[col] = window_df[col].values * scale

            # --------------------------------------------------
            # Add T3-T4 channel
            # --------------------------------------------------
            if "T3" in telex_df.columns and "T4" in telex_df.columns:
                telex_df["T3T4"] = (
                    telex_df["T3"] - telex_df["T4"]
                )

            telex_df["marker"] = window_df["marker"].values

            # Mirror the dataset's sub_XX/ses-YY hierarchy
            out_path = output_path(output_folder, file_name)

            telex_df.to_csv(out_path, index=False)

            print(f"  -> Success! {file_name} | Rows={len(telex_df)}")

        except Exception as e:
            print(f"  -> Error processing {file_name}: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Cut raw recordings down to their marker-bounded trial segment.")
    parser.add_argument("--input", default=paths.DATASET)
    parser.add_argument("--output", default=paths.PROCESSED_TRACES)
    args = parser.parse_args()

    process_eeg_for_telex(
        input_root=args.input,
        output_folder=args.output,
        sampling_rate=256,
    )