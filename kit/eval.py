"""Score a model (optionally with weight overrides) on GSM8K: accuracy, token-cap hits, looping.

python kit/eval.py --model llama-3.2-1b --split test --n 20 --max-new-tokens 1024 --out results/llama-base-test.json
python kit/eval.py --model llama-3.2-1b --overrides edits/llama-claude --split test ... --out results/llama-edit-test.json
"""
import argparse, time
from common import load_model, gsm8k, build_prompt, generate, tail_repetition, correct, summarize, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--overrides")
ap.add_argument("--split", choices=["practice", "test"], required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0, help="picks the problems")
ap.add_argument("--sample-seed", type=int, help="seeds --sampling paper; defaults to --seed")
ap.add_argument("--max-new-tokens", type=int, default=1024)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--thinking", choices=["on", "off", "default"], default="default")
ap.add_argument("--sampling", choices=["greedy", "paper"], default="greedy",
                help="paper = the authors' vLLM settings: temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5")
ap.add_argument("--out", required=True)
a = ap.parse_args()

t0 = time.time()
model, tok, changed = load_model(a.model, a.overrides)
think = None if a.thinking == "default" else a.thinking == "on"
probs = gsm8k(a.split, a.n, a.seed)
gens = generate(model, tok, [build_prompt(tok, p["question"], think) for p in probs], a.max_new_tokens, a.batch_size,
                sampling=None if a.sampling == "greedy" else a.sampling,
                seed=a.seed if a.sample_seed is None else a.sample_seed)
rows = []
for p, g in zip(probs, gens):
    rows.append({"id": p["id"], "answer": p["answer"], "correct": correct(g["text"], p["answer"]),
                 "hit_cap": g["hit_cap"], "tail_repetition": round(tail_repetition(g["tokens"]), 3),
                 "n_tokens": len(g["tokens"]), "text": g["text"], "tokens": g["tokens"]})
summary = summarize(rows)
summary.update({"model": a.model, "overrides": a.overrides, "weights_changed": changed, "split": a.split,
                "max_new_tokens": a.max_new_tokens, "thinking": a.thinking, "sampling": a.sampling,
                "seed": a.seed, "sample_seed": a.seed if a.sample_seed is None else a.sample_seed, "minutes": round((time.time() - t0) / 60, 1)})
save_json({"summary": summary, "rows": rows}, a.out)
print(summary)
