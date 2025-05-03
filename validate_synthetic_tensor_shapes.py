import torch

train = torch.load("synthetic_hirid_dataset/train.pt")
val = torch.load("synthetic_hirid_dataset/val.pt")

def check_train(train_data):
    for i, sample in enumerate(train_data):
        print(f"Sample {i}: shape={sample.shape}")
        assert sample.ndim == 2 and sample.shape[0] == 6 and sample.shape[1] == 12

def check(tensors, name):
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            print(f"[{name}] Sample {i} is not a tensor")
        elif t.ndim != 2:
            print(f"[{name}] Sample {i} shape = {t.shape} ❌ (expected 2D)")
        elif t.shape[1] != 6:
            print(f"[{name}] Sample {i} has wrong feature dim: {t.shape} ❌")
        elif t.shape[0] != 48:
            print(f"[{name}] Sample {i} has wrong time dim: {t.shape} ❌")

print("🔍 Validating training data:")
check(train, "train")

print("\n🔍 Validating validation data:")
check(val, "val")

check_train(train)
