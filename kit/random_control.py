"""Random-direction control for the paper's Qwen3.5-0.8B edit, built the way the paper built its own:
a random unit direction orthogonal to the repetition directions, put through the same formula on the
same 64 columns per layer, then scaled so each layer's change has the paper edit's Frobenius norm.
Reads only weights (the paper edit and the base model), no problems of any split.

python kit/random_control.py --paper edits/qwen35-paper --out edits/qwen35-random-control

--all-columns --match-strength 4 --seed 492698 rebuilds the paper's own control instead: every column changed,
scaled so each layer's change equals the repetition edit at strength 4 across every column (its norm is
4 * |u @ W0| because u is a unit vector), exactly as in exp8_math/runs/p50c/tool_calls.jsonl.
"""
import argparse, json
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from common import KIT, MODELS

ap = argparse.ArgumentParser()
ap.add_argument("--paper", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--all-columns", action="store_true", help="change every column, not just the paper's 64")
ap.add_argument("--match-strength", type=float, help="match strength * |u @ W0| instead of the paper edit's norm")
a = ap.parse_args()

model_dir = KIT / "models" / MODELS["qwen3.5-0.8b"][0]
idx = model_dir / "model.safetensors.index.json"
paper = load_file(f"{a.paper}/edit.safetensors")
base = {}
for name in paper:
    shard = json.load(open(idx))["weight_map"][name] if idx.exists() else "model.safetensors"
    with safe_open(model_dir / shard, "pt") as f:
        base[name] = f.get_tensor(name).float()

# The paper's change is rank one, -6 * outer(u, mask * (u @ W0)), so u is its top left singular vector.
deltas, us, cols = {}, [], {}
for name, W0 in base.items():
    D = paper[name].float() - W0
    U, S, _ = torch.linalg.svd(D, full_matrices=False)
    deltas[name], cols[name] = D, torch.nonzero(D.abs().sum(0)).flatten()
    us.append(U[:, 0])
    print(f"{name}: rank-1 check s2/s1 = {float(S[1] / S[0]):.4f}, columns = {len(cols[name])}")

if a.all_columns:
    cols = {name: torch.arange(W0.shape[1]) for name, W0 in base.items()}
Q, _ = torch.linalg.qr(torch.stack(us, 1))         # orthonormal basis of both repetition directions
g = torch.Generator().manual_seed(a.seed)
out, notes, total = {}, [], 0
for name, W0 in base.items():
    r = torch.randn(W0.shape[0], generator=g)
    r -= Q @ (Q.T @ r)                              # orthogonal to both layers' repetition directions
    r /= r.norm()
    Dr = torch.zeros_like(W0)
    Dr[:, cols[name]] = -torch.outer(r, (r @ W0)[cols[name]])
    u = us[len(notes)]                              # this layer's repetition direction (unit, sign irrelevant)
    target = a.match_strength * (u @ W0).norm() if a.match_strength else deltas[name].norm()
    scale = target / Dr.norm()
    new = (W0 + scale * Dr).to(torch.bfloat16)
    changed = int((new != W0.to(torch.bfloat16)).sum())
    total += changed
    out[name] = new.contiguous()
    notes.append({"tensor": name, "columns": int(len(cols[name])), "weights_changed": changed,
                  "frobenius_paper": round(float(deltas[name].norm()), 4),
                  "frobenius_control": round(float((new.float() - W0).norm()), 4),
                  "alpha": round(float(scale), 5),
                  "relative_change": round(float((new.float() - W0).norm() / W0.norm()), 5),
                  "max_cos_to_repetition_dirs": round(max(abs(float(r @ u / u.norm())) for u in us), 6)})
Path(a.out).mkdir(parents=True, exist_ok=True)
save_file(out, f"{a.out}/edit.safetensors")
method = ("random unit direction orthogonal to the paper's repetition directions, paper formula on every column, "
          f"norm matched per layer to the repetition edit at strength {a.match_strength:g} on every column"
          if a.all_columns and a.match_strength else
          "random unit direction orthogonal to the paper's repetition directions, paper formula on the "
          "same 64 columns per layer, Frobenius norm matched per layer")
manifest = {"model": "qwen3.5-0.8b", "control_for": a.paper, "seed": a.seed, "method": method,
            "tensors": notes, "weights_changed": total}
json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
