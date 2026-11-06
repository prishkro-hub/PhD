import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "Data" / "5umGapRes_6789MHz_data"
DATAFILE = DATA_DIR / "tau_compute.csv"
OUTPUT_FILE = DATA_DIR / "processed_compute.csv"

def process_excess(filepath):
    df = pd.read_csv(filepath)

    if 'y_exp' in df.columns and 'y_tau' in df.columns and 'dE_dt' in df.columns:

        # 1. Get Raw S-TLS Loss
        df['raw_y_S'] = df['y_exp'] - df['y_tau']

        # 2. Initialize New Excess Loss
        df['new_y_S'] = 0.0

        unique_powers = df['power'].unique()

        for P in unique_powers:
            # 3. Isolate specific power and low sweep rate
            pwr_mask = df['power'] == P
            Eb_mask = df['dE_dt'] <= 4.0e9

            # 4. Locate this power's baseline
            baseline_mask = (pwr_mask) & (Eb_mask)

            # 5. Calculate and apply the offset 
            if baseline_mask.any():
                offset = df.loc[baseline_mask, 'raw_y_S'].mean()
            else:
                print(f"Warning: No Eb <= 3.4e8 found in {filepath}. Offset set to 0.")
                offset = 0.0

            # 6. Center baseline at zero
            df.loc[pwr_mask, 'new_y_S'] = df.loc[pwr_mask, 'raw_y_S'] - offset

            print(f"{P} dBM | Offset Applied: {offset:.3e}")

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Processed data saved to: {OUTPUT_FILE}")

    else:
        print(f"Warning: Missing required columns in {filepath}. Found: {df.columns}")    

if __name__ == "__main__":
    process_excess(DATAFILE)
