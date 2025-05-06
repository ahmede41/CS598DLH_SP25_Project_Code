# Reproducing TRACE with Synthetic HiRID from MIMIC-IV

This repository contains code to reproduce results from the paper:

**"Learning Unsupervised Representations for ICU Time Series"**  
by Addison Weatherhead et al., from The Hospital for Sick Children and University of Toronto.

This work is part of my final project for the **Deep Learning for Healthcare** course. It adapts the TRACE model to work with a synthetic HiRID-style dataset generated from the MIMIC-IV demo dataset.

---
## ⚠️ Data Access Disclaimer

<p style="color:red"><strong>Important Notice:</strong> My official Data Access Request for <em>HiRID, a high time-resolution ICU dataset v1.1.1</em> has been in <strong>Pending</strong> status since <strong>April 20, 2025</strong>. As a result, I have used a <strong>synthetic HiRID-style dataset</strong> generated from MIMIC-IV v3.1 data for development and testing purposes.</p>

<p style="color:red">This substitution is the main reason I was <strong>not able to reproduce the exact results</strong> presented in the original TRACE paper. While care was taken to match the data format and dimensionality, the absence of authentic HiRID data limits full replication fidelity.</p>

### 🧪 Testing Environment Setup

To reproduce the experiments or run the scripts, the following environment setup was used:

#### System Information

```bash
# Machine
uname -a
Linux HPFolio9470m 5.14.0-580.el9.x86_64 #1 SMP PREEMPT_DYNAMIC Tue Apr 22 20:29:48 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux

# Python and Pips:
python --version
Python 3.9.21


python3 -m pip install --upgrade pip

pip3 install torch
pip3 install seaborn
pip3 install hdbscan
pip3 install umap
pip3 install statsmodels


wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-rhel9-12-8-local-12.8.1_570.124.06-1.x86_64.rpm
sudo dnf -y install ./cuda-repo-rhel9-12-8-local-12.8.1_570.124.06-1.x86_64.rpm
sudo dnf -y install cuda-toolkit-12-8

sudo dnf install cuda-nvcc-12-*

export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

---

## 📁 Project Structure

```
.
├── generate_and_prep_synthetic_hirid.py   # Generates synthetic HiRID from MIMIC-IV
├── clean_synthetic_hirid.py               # Cleans all generated data
├── check_train.py                         # Validates .pt tensors
├── check_train_data_maps.py               # Validates encoder data maps
├── tnc_for_hyper_param_optimization.py    # Training pipeline (TRACE)
├── DONTCOMMITdata/                        # Stores .npy tensors
├── DONTCOMMITplots/HiRID/                 # Stores generated plots
├── synthetic_hirid_dataset/               # Stores .pt train/val files
```

---

## 🛠️ Setup Instructions

### 1. Prepare MIMIC-IV Data

Download and unzip the MIMIC-IV demo dataset:
```bash
unzip ~/Downloads/mimic-iv-3.1.zip -d ./
```

### 2. Create Required Folders

```bash
mkdir -p DONTCOMMITdata
mkdir -p DONTCOMMITplots/HiRID/
mkdir -p synthetic_hirid_dataset
```

If re-running the process, clean old outputs:
```bash
python clean_synthetic_hirid.py
```

Sample output:
```bash
python3 clean_synthetic_hirid.py 
🧹 Removing directory: synthetic_hirid_dataset
🧹 Removing directory: DONTCOMMITdata/hirid_numpy
🧹 Removing directory: DONTCOMMITplots/HiRID
🧹 Removing directory: ckpt/HiRID/
✅ Cleanup complete. Removed: ['synthetic_hirid_dataset', 'DONTCOMMITdata/hirid_numpy', 'DONTCOMMITplots/HiRID', 'ckpt/HiRID/']
```

---

## 🧪 Generate Synthetic HiRID Dataset

```bash
python generate_and_prep_synthetic_hirid.py
```

Sample output:
```bash
Chunk 4322 → 8841 rows after filtering
Chunk 4323 → 10907 rows after filtering
Chunk 4324 → 8729 rows after filtering
Chunk 4325 → 9241 rows after filtering
Chunk 4326 → 9385 rows after filtering
Chunk 4327 → 9327 rows after filtering
Chunk 4328 → 8763 rows after filtering
Chunk 4329 → 9704 rows after filtering
Chunk 4330 → 10507 rows after filtering
Merging with ICU stays...
Generated 1000 sequences.
✅ Final valid sample count: 1000
✅ Saved 800 training and 200 validation samples
📦 Dataset ready for TRACE in 'synthetic_hirid_dataset' and 'DONTCOMMITdata/hirid_numpy'
```

---

## ✅ Validate Prepared Data

Check training tensors:
```bash
python check_train.py
# Output: torch.Size([800, 60, 36])
```

Check encoder data maps:
```bash
python check_train_data_maps.py
# Output: Shape (800, 2)
```

---

## 🧠 Start Training with TRACE

Run the TRACE training script with the desired hyperparameters:

```bash
cd TRACE

python -u -m tnc_for_hyper_param_optimization \
  --train --cont --ID 0109 --plot_embeddings \
  --encoder_type CausalCNNEncoder --window_size 12 --w 0.05 \
  --batch_size 30 --lr 0.00005 --decay 0.0005 --mc_sample_size 6 \
  --n_epochs 150 --data_type HiRID --n_cross_val_encoder 1 --ETA 4 \
  --ACF_PLUS --ACF_nghd_Threshold 0.6 --ACF_out_nghd_Threshold 0.1 \
  --CausalCNNEncoder_in_channels 36 --CausalCNNEncoder_channels 4 \
  --CausalCNNEncoder_depth 1 --CausalCNNEncoder_reduced_size 2 \
  --CausalCNNEncoder_encoding_size 10 --CausalCNNEncoder_kernel_size 2 \
  --CausalCNNEncoder_window_size 12 --n_cross_val_classification 3
```

---

## 📌 Notes

- This repo assumes you are working with the demo version of MIMIC-IV v3.1. (wget -r -N -c -np https://physionet.org/files/mimic-iv-demo/2.2/)
- The synthetic dataset mimics the structure and resolution of HiRID to allow reuse of TRACE with minimal code modification.
- Training will produce embeddings and plots in `DONTCOMMITplots/HiRID/`.

---

## 📚 Citation

If you use this project or TRACE in your research, please cite:

```
@misc{weatherhead2023trace,
  title={Learning Unsupervised Representations for ICU Time Series},
  author={Addison Weatherhead and others},
  year={2023},
  url={https://github.com/Addison-Weatherhead/TRACE}
}
```
