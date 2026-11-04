"""Activation probe for the editing session. PRACTICE PROBLEMS ONLY.

1. Generates answers to practice problems (or reuses a saved eval file on the practice split).
2. Splits them into looping vs finished.
3. Re-reads each answer with one forward pass and records, per layer, what the MLP writes into the
   residual stream over the last 64 generated tokens.
4. Per layer, saves the difference-of-means direction (looping minus finished) and how well it
   separates the two groups.

This is tooling: it turns millions of activations into a per-layer table Claude can reason about.
Claude does not read raw weights at this size.

python kit/probe.py --model llama-3.2-1b --from results/llama-base-practice.json --out probes/llama
"""
import argparse, json
import numpy as np, torch
from common import load_model, gsm8k, build_prompt, generate, tail_repetition, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--overrides")
ap.add_argument("--from", dest="src", help="eval JSON on the practice split to reuse")
ap.add_argument("--n", type=int, default=48)
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--thinking", choices=["on", "off", "default"], default="default")
ap.add_argument("--loop-threshold", type=float, default=0.3)
ap.add_argument("--tail", type=int, default=64)
ap.add_argument("--out", required=True)
a = ap.parse_args()

model, tok, _ = load_model(a.model, a.overrides)
think = None if a.thinking == "default" else a.thinking == "on"
if a.src:
    src = json.load(open(a.src))
    assert src["summary"]["split"] == "practice", "probe only reads practice-split results"
    byid = {p["id"]: p for p in gsm8k("practice", 100000)}
    items = [(build_prompt(tok, byid[r["id"]]["question"], think), r.get("tokens") or r["text"], r["hit_cap"],
              r["tail_repetition"]) for r in src["rows"] if r["id"] in byid]
else:
    probs = gsm8k("practice", a.n, seed=1)
    prompts = [build_prompt(tok, p["question"], think) for p in probs]
    gens = generate(model, tok, prompts, a.max_new_tokens)
    items = [(pr, g["tokens"], g["hit_cap"], tail_repetition(g["tokens"])) for pr, g in zip(prompts, gens)]

layers = model.model.layers
store = {}
def hook(i):
    def f(_m, _inp, out):
        store[i] = out.detach().float()
    return f
hs = [layers[i].mlp.register_forward_hook(hook(i)) for i in range(len(layers))]

feats, labels = [], []
with torch.no_grad():
    for pr, gen, hit, rep in items:
        p_ids = tok(pr, add_special_tokens=False)["input_ids"]
        g_ids = gen if isinstance(gen, list) else tok(gen, add_special_tokens=False)["input_ids"]
        if len(g_ids) < a.tail:
            continue
        ids = torch.tensor([p_ids + g_ids], device=model.device)
        model(ids)
        feats.append(np.stack([store[i][0, -a.tail:].mean(0).cpu().numpy() for i in range(len(layers))]))
        labels.append(1 if (hit and rep >= a.loop_threshold) else 0)
for h in hs:
    h.remove()

if not feats:
    raise SystemExit(f"No answer was at least {a.tail} tokens long; lower --tail or raise --max-new-tokens.")
F, y = np.stack(feats), np.array(labels)
n_loop, n_ok = int(y.sum()), int((1 - y).sum())
report = {"model": a.model, "answers_read": len(y), "looping": n_loop, "finished": n_ok, "layers": []}
dirs = {}
if n_loop >= 3 and n_ok >= 3:
    for i in range(F.shape[1]):
        d = F[y == 1, i].mean(0) - F[y == 0, i].mean(0)
        u = d / (np.linalg.norm(d) + 1e-8)
        proj = F[:, i] @ u
        gap = proj[y == 1].mean() - proj[y == 0].mean()
        pooled = np.sqrt((proj[y == 1].var() + proj[y == 0].var()) / 2) + 1e-8
        dirs[f"layer{i}"] = u.astype(np.float32)
        report["layers"].append({"layer": i, "separation": round(float(gap / pooled), 2),
                                 "gap_norm": round(float(np.linalg.norm(d)), 3)})
    np.savez(f"{a.out}-directions.npz", **dirs)
else:
    report["note"] = "Too few looping or finished answers to compare (need 3 of each). Nothing to fix here, or raise --n / --max-new-tokens."
save_json(report, f"{a.out}-report.json")
print(json.dumps({k: v for k, v in report.items() if k != "layers"}))
for r in sorted(report["layers"], key=lambda r: -r["separation"])[:8]:
    print(r)
