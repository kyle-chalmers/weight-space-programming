# Weight-space programming on a laptop: replicating a loop-fixing weight edit

In *[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)* (Takehi, Teoh,
Zhang and Khattab, MIT CSAIL, Nov 2026, a blog post that is not peer reviewed), a model working in a Python sandbox
fixed Qwen3.5-0.8B's habit of getting stuck in reasoning loops on math problems. It did that without any training: it
removed one direction from what two MLP layers write, a change to about 131,000 of the model's 750 million weights.

This repo checks that result on one Mac (M4 Pro, 24 GB), using the authors' published edit and their scoring
settings, and asks two follow-up questions:

1. **Does the paper's edit reproduce?** Yes, at about the paper's size.
2. **Is it the direction, or just disturbing those neurons?** Random-direction controls of the same size, built two
   ways, separate the two.

It also gave Claude Code the same kind of task blind (practice problems only, a per-layer probe table, an edit tool)
to see whether it could find its own fix. It didn't, and `results.md` explains why.

The full write-up, with every table and caveat, is **[results.md](results.md)**.

## Headline results

Qwen3.5-0.8B, GSM8K test problems, thinking on, the paper's sampling (temperature 0.6, top_p 0.95, top_k 20,
presence penalty 1.5), 16,384-token cap.

| Version | Weights changed | Correct | Looping |
|---|---|---|---|
| Base (50 problems x 3 sampling seeds) | 0 | 17.3% | 87% |
| Paper's edit (same 150 answers) | 130,188 | 63.3% | 8% |
| Random direction, same 64 neurons per layer, same size (seed 0) | 129,318 | 34% | 60% |

The paper reported 10.6% to 60.4%. Against base on the same seeds, the edit fixed 70 answers and broke 1.

## What Claude reads vs what code does

A model with nearly a billion weights can't be read as text. `probe.py` runs the model on practice problems, records
what each layer writes, and reduces it to one table: for each layer, how strongly a "looping" direction separates
stuck answers from finished ones. Claude reads that table, picks layers, a strength and how many columns to touch, and
`edit_direction.py` writes the edit. That is the same split the paper used: the model reasoned over statistics a
Python kernel computed.

## Run it yourself

```bash
./setup.sh
```

`setup.sh` creates `.venv`, installs PyTorch and transformers, clones the paper's logs into `vendor/` (for their
published edit), and downloads the models into `models/`. It accepts no license and creates no token. Llama 3.2 1B is
gated by Meta; accept its license on Hugging Face and run `hf auth login` yourself if you want it.

Score the base model and the paper's edit (the paper's sampling, 50 test problems, sampling seed 0):

```bash
source .venv/bin/activate
python kit/eval.py --model qwen3.5-0.8b --split test --n 50 --seed 0 --max-new-tokens 16384 \
  --thinking on --sampling paper --sample-seed 0 --out results/my-base.json
python kit/paper_edit.py --out edits/qwen35-paper
python kit/eval.py --model qwen3.5-0.8b --overrides edits/qwen35-paper --split test --n 50 --seed 0 \
  --max-new-tokens 16384 --thinking on --sampling paper --sample-seed 0 --out results/my-paper.json
```

Build the two random controls:

```bash
python kit/random_control.py --paper edits/qwen35-paper --out edits/qwen35-random-control
python kit/random_control.py --paper edits/qwen35-paper --out edits/qwen35-paper-style-control \
  --all-columns --match-strength 4 --seed 492698
```

On an M4 Pro each 50-problem run at 16,384 tokens took about 1.75 to 3.5 hours on an otherwise idle machine, because Qwen3.5's fast linear-attention
kernels don't run on Apple Silicon. Try `--n 2 --max-new-tokens 512` first to time your machine.

## Files

| Path | What it is |
|---|---|
| `kit/eval.py` | GSM8K accuracy, token-cap hits and looping share; `--split practice` or `--split test`; greedy or `--sampling paper` |
| `kit/probe.py` | per-layer looping direction from practice answers (practice only) |
| `kit/edit_direction.py` | W' = W - alpha * u(u^T W) on chosen `mlp.down_proj` layers, optionally top-k columns |
| `kit/paper_edit.py` | rebuilds the paper's published Qwen edit with the authors' `apply_edit.py` (hash-checked) |
| `kit/random_control.py` | random-direction controls, on the paper's 64 columns or the paper's own all-column version |
| `results/*.json` | every scored run, with per-problem answers, token counts and the run's settings |
| `edits/*/manifest.json` | what each edit changed; the edited weights themselves aren't committed |
| `results.md` | the full write-up |
| `LAUNCH-PROMPT.md` | the prompt that started the run, including the blind brief |

"Looping" means the answer hit the token cap and its last 256 tokens are at least 30% repeated 8-grams. GSM8K test
problems were used only for before/after scoring and never shown to the session that built an edit.

## Credits

The edit, the scoring settings and the control design come from the paper's authors and their public logs at
[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (CC BY 4.0 for logs and data; see its
LICENSE). This repo doesn't copy those logs; `setup.sh` clones them.
