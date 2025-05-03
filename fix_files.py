import os

npy_dir = "DONTCOMMITdata/hirid_numpy"

def save_dataset():
    for f in os.listdir(npy_dir):
        src = os.path.join(npy_dir, f)
        dst = os.path.join(npy_dir, f.upper())
        if not os.path.exists(dst):
            os.system(f"cp {src} {dst}")

        dst = os.path.join(npy_dir, f.replace(f.split("_")[0], f.split("_")[0].upper()))
        if not os.path.exists(dst):
            os.system(f"cp {src} {dst}")

save_dataset()
