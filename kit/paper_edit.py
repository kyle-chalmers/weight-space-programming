"""Rebuild the paper's published Qwen3.5-0.8B edit (50-pass run) as an overrides folder this kit can load.
Uses the authors' own apply_edit.py, which checks every tensor against their hashes.

python kit/paper_edit.py --out edits/qwen35-paper
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import save_file
from common import KIT, MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="p50c")
ap.add_argument("--out", required=True)
a = ap.parse_args()

logs = KIT / "vendor" / "lm_understand-logs" / "exp8_math"
model_dir = KIT / "models" / MODELS["qwen3.5-0.8b"][0]
edit_dir = logs / "runs" / a.run / "edit"
names = [t["name"] for t in json.load(open(edit_dir / "manifest.json"))["tensors"]]
idx = model_dir / "model.safetensors.index.json"
if idx.exists():
    files = {json.load(open(idx))["weight_map"][n] for n in names}
    assert len(files) == 1, f"edited tensors span shards {files}; merge needed"
    base = model_dir / files.pop()
else:
    base = model_dir / "model.safetensors"
tmp = KIT / "edits" / f"_npy_{a.run}"
subprocess.run([sys.executable, str(logs / "code" / "apply_edit.py"), str(base), str(edit_dir), str(tmp)], check=True)
tensors = {n: torch.tensor(np.load(tmp / f"{n}.npy")).to(torch.bfloat16).contiguous() for n in names}
Path(a.out).mkdir(parents=True, exist_ok=True)
save_file(tensors, f"{a.out}/edit.safetensors")
print(f"{len(tensors)} tensors -> {a.out}")
