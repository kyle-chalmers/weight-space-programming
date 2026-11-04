"""Write a weight edit that removes a direction from what chosen MLP layers write:
W' = W - alpha * u (u^T W), on mlp.down_proj, optionally only on the k columns that write u most
(the paper's final Qwen edit used 64 columns in each of two layers).

python kit/edit_direction.py --model llama-3.2-1b --directions probes/llama-directions.npz \
    --layers 9,12 --alpha 1.0 --columns 64 --out edits/llama-claude
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from safetensors.torch import save_file
from common import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--directions", required=True)
ap.add_argument("--layers", required=True, help="comma-separated layer numbers")
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--columns", type=int, default=0, help="0 = all columns")
ap.add_argument("--out", required=True)
a = ap.parse_args()

model, _, _ = load_model(a.model, device="cpu")
D = np.load(a.directions)
out, total, notes = {}, 0, []
for l in [int(x) for x in a.layers.split(",")]:
    name = next(n for n, _ in model.named_parameters() if n.endswith(f"layers.{l}.mlp.down_proj.weight"))
    W = dict(model.named_parameters())[name].detach().float()          # [hidden, intermediate]
    u = torch.tensor(D[f"layer{l}"]).float()
    writes = u @ W                                                     # how much each column writes u
    cols = torch.arange(W.shape[1]) if a.columns <= 0 else torch.topk(writes.abs(), a.columns).indices
    W2 = W.clone()
    W2[:, cols] = W[:, cols] - a.alpha * torch.outer(u, writes[cols])
    new = W2.to(torch.bfloat16)
    changed = int((new != W.to(torch.bfloat16)).sum())
    total += changed
    out[name] = new.contiguous()
    notes.append({"tensor": name, "columns": int(len(cols)), "weights_changed": changed})
Path(a.out).mkdir(parents=True, exist_ok=True)
save_file(out, f"{a.out}/edit.safetensors")
n_all = sum(p.numel() for p in model.parameters())
manifest = {"model": a.model, "layers": a.layers, "alpha": a.alpha, "columns": a.columns, "tensors": notes,
            "weights_changed": total, "share_of_model": f"{100 * total / n_all:.4f}%"}
json.dump(manifest, open(f"{a.out}/manifest.json", "w"), indent=1)
print(manifest)
