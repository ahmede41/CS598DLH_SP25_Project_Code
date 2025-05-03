import pandas as pd
import numpy as np
import torch
import os
import shutil

# ----------------------------------------
# Auto-Cleanup of Old Outputs
# ----------------------------------------
# Remove existing output directories if they exist
for path in ["synthetic_hirid_dataset", "DONTCOMMITdata/hirid_numpy", "DONTCOMMITplots/HiRID"]:
    if os.path.exists(path):
        print(f"🧹 Removing existing directory: {path}")
        shutil.rmtree(path)

# ----------------------------------------
# Settings
# ----------------------------------------
# Define the vital signs of interest and their corresponding item IDs
VITAL_SIGNS = {
    'Heart Rate': [220045],
    'SysBP': [220179],
    'DiasBP': [220180],
    'RespRate': [220210],
    'SpO2': [220277],
    'TempC': [223761],
}

SEQ_LEN = 60
MAX_SAMPLES = 1000
CHUNK_SIZE = 100_000
BASE_DIR = "mimic-iv-3.1"
OUTPUT_DIR = "synthetic_hirid_dataset"
NPY_DIR = "DONTCOMMITdata/hirid_numpy"

# ----------------------------------------
# Data Extraction
# ----------------------------------------
def load_chart_events_filtered(chart_path, icu_path, valid_itemids):
    """
    Loads and filters the chart events and ICU stay data to include only relevant vital signs.
    
    Args:
        chart_path (str): Path to the chart events CSV.
        icu_path (str): Path to the ICU stays CSV.
        valid_itemids (List[int]): List of item IDs to filter on.
    
    Returns:
        pd.DataFrame: Merged and filtered dataframe with chart events within ICU stay times.
    """
    print("Loading ICU stays...")
    icu = pd.read_csv(icu_path)
    icu['intime'] = pd.to_datetime(icu['intime'])
    icu['outtime'] = pd.to_datetime(icu['outtime'])

    print("Reading chartevents in chunks...")
    chunks = []
    for i, chunk in enumerate(pd.read_csv(chart_path, usecols=['subject_id', 'stay_id', 'charttime', 'itemid', 'valuenum'], chunksize=CHUNK_SIZE)):
        chunk = chunk[chunk['itemid'].isin(valid_itemids)]
        chunk = chunk.dropna(subset=['valuenum'])
        chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
        chunk = chunk.dropna(subset=['charttime'])
        chunks.append(chunk)
        print(f"Chunk {i+1} → {len(chunk)} rows after filtering")

    chart = pd.concat(chunks, ignore_index=True)
    print("Merging with ICU stays...")
    merged = chart.merge(icu[['stay_id', 'intime', 'outtime']], on='stay_id')
    merged = merged[(merged['charttime'] >= merged['intime']) & (merged['charttime'] <= merged['outtime'])]

    # Map itemids to readable labels
    label_map = {iid: label for label, iids in VITAL_SIGNS.items() for iid in iids}
    merged['label'] = merged['itemid'].map(label_map)

    return merged

def generate_sequences(df, max_samples=MAX_SAMPLES, seq_len=SEQ_LEN):
    """
    Converts time-series data into fixed-length sequences grouped by stay ID.
    
    Args:
        df (pd.DataFrame): Filtered time-series data with labeled vital signs.
        max_samples (int): Maximum number of sequences to generate.
        seq_len (int): Desired sequence length.
    
    Returns:
        List[np.ndarray]: List of generated sequences.
    """
    sequences = []
    for stay_id, group in df.groupby('stay_id'):
        pivoted = group.pivot_table(index='charttime', columns='label', values='valuenum')
        pivoted = pivoted.resample('1H').mean()

        # Ensure all vital sign columns are present
        for feat in VITAL_SIGNS:
            if feat not in pivoted.columns:
                pivoted[feat] = 0.0
        pivoted = pivoted[VITAL_SIGNS.keys()].fillna(0)

        MIN_LEN = 49  # TRACE minimum requirement
        if len(pivoted) >= max(seq_len, MIN_LEN):
            clipped = pivoted.iloc[:seq_len].values.astype(np.float32)
            if clipped.shape[1] == 6:
                sequences.append(clipped)
            else:
                print(f"⚠️ Skipped stay_id {stay_id} due to bad shape {clipped.shape}")

        if len(sequences) >= max_samples:
            break
    print(f"Generated {len(sequences)} sequences.")
    return sequences

def normalize(sequences):
    """
    Normalizes each vital sign across all sequences using mean and std.
    
    Args:
        sequences (List[np.ndarray]): List of raw sequences.
    
    Returns:
        Tuple: Normalized sequences, global mean, and global std.
    """
    all_data = np.concatenate(sequences, axis=0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    normed = [(seq - mean) / (std + 1e-6) for seq in sequences]
    return normed, mean, std

# Saving for TRACE-compatible model inputs
def save_dataset(normed, output_dir, npy_dir):
    """
    Saves the normalized sequences and labels to disk in the format expected by TRACE.
    
    Args:
        normed (List[np.ndarray]): List of normalized sequences.
        output_dir (str): Directory to save .pt files.
        npy_dir (str): Directory to save .npy metadata files.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs("DONTCOMMITplots/HiRID", exist_ok=True)

    clean_seqs = []
    for i, seq in enumerate(normed):
        try:
            tensor = torch.tensor(seq, dtype=torch.float32)
            if tensor.ndim == 2 and tensor.shape[1] == 6:
                # Input shape: (T, 6) → transpose → (6, T)
                tensor = tensor.T
                tensor = tensor.repeat(6, 1)  # Make it (36, T) for 36-channel encoder
                assert tensor.shape[0] == 36, f"Unexpected shape after repeat: {tensor.shape}"
                clean_seqs.append(tensor)
            else:
                print(f"⚠️ Skipped sample {i}: shape {tensor.shape}")
        except Exception as e:
            print(f"⚠️ Error converting sample {i}: {e}")

    if len(clean_seqs) < 100:
        raise ValueError(f"❌ Only {len(clean_seqs)} valid sequences found — too few for training.")

    print(f"✅ Final valid sample count: {len(clean_seqs)}")

    # Label = 1 if mean Heart Rate > 0 else 0
    labels = torch.tensor([
        int(torch.mean(seq[0]) > 0) for seq in clean_seqs
    ])
    indices = np.random.permutation(len(clean_seqs))
    split = int(0.8 * len(clean_seqs))
    train_idx, val_idx = indices[:split], indices[split:]

    train = [clean_seqs[i] for i in train_idx]
    val = [clean_seqs[i] for i in val_idx]
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]

    train_tensor = torch.stack(train).permute(0, 2, 1)  # (N, T, 36)
    val_tensor = torch.stack(val).permute(0, 2, 1)

    torch.save(train_tensor, os.path.join(output_dir, "train.pt"))
    torch.save(val_tensor, os.path.join(output_dir, "val.pt"))
    torch.save(train_labels, os.path.join(output_dir, "train_labels.pt"))
    torch.save(val_labels, os.path.join(output_dir, "val_labels.pt"))

    # Save metadata for TRACE
    os.makedirs(npy_dir,exist_ok=True)
    np.save(f"{npy_dir}/train_mortality_labels.npy", train_labels.numpy())
    np.save(f"{npy_dir}/test_mortality_labels.npy", val_labels.numpy())
    np.save(f"{npy_dir}/train_mortality_data_maps.npy", np.arange(len(train)))
    np.save(f"{npy_dir}/test_mortality_data_maps.npy", np.arange(len(val)))
    np.save(f"{npy_dir}/train_first_24_hrs_PIDs.npy", np.arange(len(train)))
    np.save(f"{npy_dir}/test_first_24_hrs_PIDs.npy", np.arange(len(val)))
    np.save(f"{npy_dir}/train_Apache_Groups.npy", np.zeros(len(train), dtype=np.int32))
    np.save(f"{npy_dir}/test_Apache_Groups.npy", np.zeros(len(val), dtype=np.int32))
    np.save(f"{npy_dir}/train_encoder_data_maps.npy", np.stack([np.arange(len(train)), np.zeros(len(train), dtype=int)], axis=1))
    np.save(f"{npy_dir}/test_encoder_data_maps.npy", np.stack([np.arange(len(val)), np.zeros(len(val), dtype=int)], axis=1))

    # TRACE expects some upper-case filenames
    for f in os.listdir(npy_dir):
        src = os.path.join(npy_dir, f)
        dst = os.path.join(npy_dir, f.upper())
        if not os.path.exists(dst):
            os.system(f"cp {src} {dst}")
        dst2 = os.path.join(npy_dir, f.replace(f.split("_")[0], f.split("_")[0].upper()))
        if not os.path.exists(dst2):
            os.system(f"cp {src} {dst2}")

    print(f"✅ Saved {len(train)} training and {len(val)} validation samples")

# Entry point for dataset generation
def main():
    """
    Main script execution to generate synthetic HiRID-style dataset from MIMIC-IV.
    """
    chart_path = os.path.join(BASE_DIR, "icu/chartevents.csv.gz")
    icu_path = os.path.join(BASE_DIR, "icu/icustays.csv.gz")
    itemids = [iid for iids in VITAL_SIGNS.values() for iid in iids]

    df = load_chart_events_filtered(chart_path, icu_path, itemids)
    sequences = generate_sequences(df)
    normed, mean, std = normalize(sequences)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "mean.npy"), mean)
    np.save(os.path.join(OUTPUT_DIR, "std.npy"), std)

    save_dataset(normed, OUTPUT_DIR, NPY_DIR)
    print(f"📦 Dataset ready for TRACE in '{OUTPUT_DIR}' and '{NPY_DIR}'")

if __name__ == "__main__":
    main()
