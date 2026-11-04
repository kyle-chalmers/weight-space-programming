"""Shared helpers for the open-model kit: device, model loading (with weight overrides),
GSM8K problems, batched generation and loop metrics.

Everything runs locally. Model weights come from the Hugging Face cache (models/ folder).
"""
import json
import os
import random
import re
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers.generation import (LogitsProcessor, LogitsProcessorList, TemperatureLogitsWarper,
                                     TopKLogitsWarper, TopPLogitsWarper)

KIT = Path(__file__).resolve().parent.parent
MODELS = {
    # name: (local folder under models/, loader kind)
    "llama-3.2-1b": ("Llama-3.2-1B-Instruct", "auto"),
    "qwen3.5-0.8b": ("Qwen3.5-0.8B-text", "qwen35_text"),
    "qwen2.5-0.5b": ("Qwen2.5-0.5B-Instruct", "auto"),
}
PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _remap(name):
    # The Qwen3.5 release stores the language model under model.language_model.*;
    # the text-only class names it model.*.
    return name.replace("model.language_model.", "model.", 1)


def load_model(name, overrides=None, device=None, dtype=torch.bfloat16):
    """Load a model by kit name. `overrides` is a folder of .safetensors files whose tensors
    replace same-named weights (names may use either the release or the text-only prefix)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    folder, kind = MODELS[name]
    path = KIT / "models" / folder
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run setup.sh first.")
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if kind == "qwen35_text":
        # Text-only class built from the authors' config, then the release weights remapped in.
        from transformers import AutoModelForCausalLM as A
        cfg = AutoConfig.from_pretrained(path)
        model = A.from_config(cfg, dtype=dtype)
        state = {}
        for f in sorted(path.glob("*.safetensors")):
            for k, v in load_file(str(f)).items():
                k2 = _remap(k)
                if k2.startswith("model.") or k2.startswith("lm_head."):
                    state[k2] = v
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [m for m in missing if not (m == "lm_head.weight" and cfg.tie_word_embeddings)]
        if missing:
            raise SystemExit(f"Weights missing after load ({len(missing)}), e.g. {missing[:3]}")
        if cfg.tie_word_embeddings:
            model.tie_weights()
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    changed = apply_overrides(model, overrides) if overrides else 0
    model.to(device).eval()
    return model, tok, changed


def apply_overrides(model, folder):
    params = dict(model.named_parameters())
    n_changed = 0
    for f in sorted(Path(folder).glob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            k2 = _remap(k)
            if k2 not in params:
                raise SystemExit(f"Override {k} matches no weight in the model")
            p = params[k2]
            if tuple(p.shape) != tuple(v.shape):
                raise SystemExit(f"Override {k} has shape {tuple(v.shape)}, weight is {tuple(p.shape)}")
            with torch.no_grad():
                new = v.to(p.dtype)
                n_changed += int((new != p.data).sum())
                p.data.copy_(new)
    return n_changed


def gsm8k(split, n, seed=0):
    """split: 'practice' (GSM8K train) or 'test' (GSM8K test). Fixed random sample of n."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train" if split == "practice" else "test")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:n]:
        q, a = ds[i]["question"], ds[i]["answer"]
        out.append({"id": f"{split}-{i}", "question": q, "answer": a.split("####")[-1].strip().replace(",", "")})
    return out


def build_prompt(tok, question, thinking):
    msgs = [{"role": "user", "content": f"{question}\n{PROMPT_SUFFIX}"}]
    kw = {"enable_thinking": thinking} if thinking is not None else {}
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# The paper's evaluation sampling (vLLM): exp8_math/README.md in the authors' logs.
PAPER_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


class PaperSampler(LogitsProcessor):
    """vLLM-style sampling in vLLM's order: presence penalty on tokens already generated, then
    temperature, top-k, top-p. One processor, so transformers' own warpers stay off."""

    def __init__(self, prompt_len, temperature, top_p, top_k, presence_penalty):
        self.prompt_len, self.penalty = prompt_len, presence_penalty
        self.warpers = [TemperatureLogitsWarper(temperature), TopKLogitsWarper(top_k), TopPLogitsWarper(top_p)]

    def __call__(self, input_ids, scores):
        out = input_ids[:, self.prompt_len:]
        if out.shape[1]:
            seen = torch.zeros_like(scores).scatter_(1, out, 1.0)
            scores = scores - self.penalty * seen
        for w in self.warpers:
            scores = w(input_ids, scores)
        return scores


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size=8, sampling=None, seed=0):
    """Greedy decoding by default (deterministic, and the setting where small models loop most).
    sampling="paper" uses PAPER_SAMPLING, seeded per batch."""
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        if sampling == "paper":
            torch.manual_seed(seed + s)
            proc = PaperSampler(enc["input_ids"].shape[1], **PAPER_SAMPLING)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=1.0,
                                 top_k=0, top_p=1.0, logits_processor=LogitsProcessorList([proc]),
                                 pad_token_id=tok.pad_token_id)
        else:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            ids = row.tolist()
            gc_eos = getattr(model.generation_config, "eos_token_id", None)
            gc_eos = gc_eos if isinstance(gc_eos, (list, tuple)) else [gc_eos]
            eos = {t for t in [tok.eos_token_id, *gc_eos] if t is not None}
            ended = any(t in eos for t in ids)
            if ended:
                ids = ids[:next(i for i, t in enumerate(ids) if t in eos)]
            ids = [t for t in ids if t != tok.pad_token_id]
            outs.append({"tokens": ids, "text": tok.decode(ids, skip_special_tokens=True), "hit_cap": not ended})
        print(f"  generated {min(s + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def tail_repetition(tokens, window=256, n=8):
    """Share of repeated 8-grams in the last `window` tokens: 0 = no repeats, near 1 = stuck in a loop."""
    t = tokens[-window:]
    grams = [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    if not grams:
        return 0.0
    return 1 - len(set(grams)) / len(grams)


def extract_answer(text):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    cand = boxed[-1] if boxed else (re.findall(r"-?\d[\d,]*\.?\d*", text) or [""])[-1]
    cand = cand.replace(",", "").replace("$", "").strip().rstrip(".")
    return cand


def correct(text, answer):
    a = extract_answer(text)
    try:
        return abs(float(a) - float(answer)) < 1e-6
    except ValueError:
        return False


def summarize(rows, loop_threshold=0.3):
    n = len(rows)
    loops = [r for r in rows if r["hit_cap"] and r["tail_repetition"] >= loop_threshold]
    return {
        "problems": n,
        "accuracy": round(sum(r["correct"] for r in rows) / n, 3),
        "hit_token_cap": round(sum(r["hit_cap"] for r in rows) / n, 3),
        "looping": round(len(loops) / n, 3),
        "mean_tail_repetition": round(sum(r["tail_repetition"] for r in rows) / n, 3),
    }


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
