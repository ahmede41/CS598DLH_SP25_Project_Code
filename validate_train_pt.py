import torch

path = "synthetic_hirid_dataset/train.pt"

try:
    data = torch.load(path)
except Exception as e:
    print(f"❌ Failed to load {path}: {e}")
    exit(1)

print(f"🔍 Loaded {len(data)} samples from {path}")

bad_samples = 0

for i, sample in enumerate(data):
    if not isinstance(sample, torch.Tensor):
        print(f"❌ Sample {i} is not a tensor → {type(sample)}")
        bad_samples += 1
        continue

    if sample.ndim != 2:
        print(f"❌ Sample {i} has {sample.ndim} dims, expected 2 → shape={sample.shape}")
        bad_samples += 1
        continue

    if sample.shape != (48, 6):
        print(f"❌ Sample {i} has wrong shape → {sample.shape} (expected [48, 6])")
        bad_samples += 1

if bad_samples == 0:
    print("✅ All samples are valid: torch.Tensor with shape [48, 6]")
else:
    print(f"⚠️ Found {bad_samples} invalid samples — consider regenerating your dataset.")

