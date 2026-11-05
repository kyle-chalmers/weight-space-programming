# Open-model kit: results (Nov 2 to 4 2026)

Run on an M4 Pro Mac (24 GB, Apple GPU) by Claude Code, with Kyle Chalmers reviewing.

The paper: [Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights](https://rikiyat.github.io/blog/weight-space-programming/), a blog post by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL), dated Nov 2 2026 and not peer reviewed. Full run logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

## TLDR

- **On 50 problems and 3 sampling seeds, the paper's edit reproduced at about the paper's size.** Scored with the paper's sampling at a 16,384-token cap, the edited Qwen3.5-0.8B got 63.3% of 150 answers right against 17.3% for the unedited model, and looped on 8% against 87%. The paper reported 10.6% to 60.4%. Every seed agreed within 4 points. Against base on the same seed, the edit fixed 70 answers and broke 1, and it did better on 35 of the 50 problems and worse on 1 (sign test p = 1e-9).
- **Most of the gain comes from the paper's direction, and only a small part from the neurons it edits.** A random direction of the same size on the same 64 neurons per layer, scored on the same 3 seeds, got 26.7% of 150 answers right and looped on 73%, against 17.3% and 87% for base and 63.3% and 8% for the paper's edit. That is about a fifth of the accuracy gain, and it swung a lot by seed (34%, 18% and 28%). The paper's edit beat it by 57 answers fixed to 2 broken. Seed 0 alone had suggested a third, and seeds 1 and 2 brought that down.
- **Spreading a random change across every neuron did about nothing, which matches the paper's own control.** A random direction across all columns, matched to the paper's control setup, got 20% right against 14% for base on seed 0 (6 fixed, 3 broken, not significant), and looped on 78% against 90%. The paper reported +1.6 points for its version.
- **The first 20-problem check, scored the way the paper scored, already showed it, accuracy included.** At a 16,384-token cap with the paper's sampling (temperature 0.6, presence penalty 1.5), the edited Qwen3.5-0.8B got 11 of 20 test problems right (55%), against 1 of 20 (5%) for the unedited model. It looped on 2 of 20 answers against 18 of 20. Twelve of its answers finished, and 10 of those were correct. That is the same shape as the paper's 7.9% to 48.1%. A gap of 11 vs 1 on 20 problems is very unlikely to be chance (Fisher exact p = 0.001), but 20 problems can't pin down its size.
- **Greedy decoding hid the gain.** Under greedy decoding the paper's edit looped least at every cap (2048, 4096 and 16,384), but at most 2 of its 20 answers finished, and raising the cap from 4,096 to 16,384 finished nothing new. The five wrap-up loops seen under greedy ("The reasoning is clear. I will write the response." repeated without closing) don't happen with the paper's sampling.
- **The paper's sampling alone doesn't fix the unedited model.** Base still looped on 18 of 20 and got 1 right. It takes the edit and the sampling together.
- **Claude did not find a working edit on either model.** On Qwen2.5-0.5B there was nothing to fix: the model almost never loops. On Qwen3.5-0.8B, Claude found plenty of looping and a clear per-layer signal, tried five edits on practice problems, and none of them cut looping without costing accuracy. The edit it kept scored the same as the unedited model in every test run, including with the paper's sampling (2 of 20 correct vs 1 of 20, 18 of 20 looping for both).
- **Llama 3.2 1B did not run.** It is gated by Meta, and nobody accepted the license or logged in for Kyle, so Qwen2.5-0.5B stood in as the second model.
- Confidence: high that the paper's edit helps this model under the paper's settings, and moderate on the size (about 46 points, on 150 scored answers). Moderate on the controls: the 64-neuron control's small gain is real on these 150 answers (p = 0.03), but it is one random direction, and the all-columns control ran on one seed. Still untested: the paper's 32,768 cap, and our reimplementation of the paper's vLLM sampling against vLLM itself.

## Settings (identical for every before/after pair)

| Model | Test problems | Seed | Token cap | Thinking | Minutes per eval |
|---|---|---|---|---|---|
| qwen3.5-0.8b | 20 (GSM8K test) | 0 | 2048 | on | 5.1 |
| qwen3.5-0.8b (rerun) | the same 20 | 0 | 4096 | on | 8.5 to 11.6 |
| qwen3.5-0.8b (rerun) | the same 20 | 0 | 16384 | on | 38 to 39 |
| qwen3.5-0.8b (paper's sampling) | the same 20 | 0 | 16384 | on | 42 |
| qwen3.5-0.8b (paper's sampling, sampling seeds 0 to 2) | 50 (GSM8K test) | 0 | 16384 | on | 105 to 323 |
| qwen3.5-0.8b (64-column random control, sampling seeds 0 to 2) | the same 50 | 0 | 16384 | on | 115 to 209 |
| qwen3.5-0.8b (all-columns random control, sampling seed 0) | the same 50 | 0 | 16384 | on | 160 |
| qwen2.5-0.5b | 20 (GSM8K test) | 0 | 1024 | default (none) | 0.8 |

Greedy decoding except the "paper's sampling" row, batch 8. That row uses `eval.py --sampling paper`: temperature 0.6, top_p 0.95, top_k 20 and a presence penalty of 1.5, taken from the paper's scoring setup (`exp8_math/README.md` in the logs), with each batch seeded from the run seed. "Looping" means the answer hit the token cap and its last 256 tokens are at least 30% repeated 8-grams. Timing check (2 problems, 512 tokens): Qwen3.5 took 1.0 minute and Qwen2.5 0.7 minutes. Qwen3.5 ran on transformers' slow reference kernels, because `flash-linear-attention` and `causal_conv1d` don't run on Apple Silicon. Even so, a full eval took about 5 minutes, well under the 30-minute budget.

## Results

### Qwen3.5-0.8B (the paper's model), 20 test problems

| Version | Weights changed | Share of model | Accuracy | Hit token cap | Looping | Mean tail repetition |
|---|---|---|---|---|---|---|
| Base | 0 | 0 | 15% (3) | 95% (19) | 65% (13) | 0.47 |
| Paper's edit (layers 15 and 19, 64 columns each) | 130,188 | 0.0173% | 25% (5) | 100% (20) | 25% (5) | 0.24 |
| Claude's edit (layers 10 and 14, 64 columns each, alpha 1.0) | 128,505 | 0.0171% | 10% (2) | 100% (20) | 75% (15) | 0.59 |

How to read accuracy here: at a 2048-token cap the model almost never finishes its thinking. Only 1 of 60 Qwen3.5 answers across these three runs finished. Of the 10 answers scored correct, 9 were cut off at the cap and scored from the last number in the unfinished reasoning. So for this model, **looping is the meaningful measure and accuracy is close to noise.** The paper's 7.9% to 48.1% came from runs allowed up to 32,768 tokens.

Share of model uses 752,393,024 weights in the text model, counted the way `edit_direction.py` counts. The paper's edit has 131,072 entries in its files. 884 of them round back to their original bf16 value, which leaves 130,188 actually changed (the authors' `apply_edit.py` matched every tensor hash). The paper's own 0.016% figure is computed against a larger total.

### Qwen3.5-0.8B at a 4096-token cap (rerun, same 20 problems)

Run after the blind work had finished, to see whether answers would finish once they had more room. Each version used the same problems, seed and thinking setting as above, with only the cap changed.

| Version | Accuracy | Finished (and correct) | Hit token cap | Looping | Mean tail repetition |
|---|---|---|---|---|---|
| Base | 10% (2) | 1 (1) | 95% (19) | 90% (18) | 0.65 |
| Paper's edit | 25% (5) | 2 (2) | 90% (18) | 70% (14) | 0.57 |
| Claude's edit | 10% (2) | 0 (0) | 100% (20) | 95% (19) | 0.71 |

What this shows (observed):
- **Doubling the cap barely let answers finish.** Only 3 of 60 answers finished, up from 1 of 60 at 2048. The answers this model gets stuck on stay stuck past 4096 tokens.
- **Looping went up for every version.** An answer that is going in circles at 2048 tokens has settled into tighter repetition by 4096, so more answers cross the 30% line. Looping rates can be compared within a cap but not across caps.
- **The order held.** The paper's edit loops least at both caps (5 of 20 at 2048, 14 of 20 at 4096), and Claude's edit loops most. At 20 problems, a gap of 4 answers like 14 vs 18 could still be chance.
- **Accuracy still can't be read.** Of the paper edit's 5 correct answers, 2 finished and 3 were scored from the last number in cut-off text. Its 2 vs 1 finished-and-correct answers over base are too few to call a gain.

A difference in kind (observed in the text, interpretation inferred): base and Claude's-edit loops are all second-guessing ("Wait, is it possible the question means...", "Wait, looking at the prompt again"). Five of the paper edit's 14 loops are different. The model has reached its answer and repeats a closing line without ever closing its thinking block, for example "The answer is 72. The reasoning is clear." over and over. One reading is that the edit weakens the doubt loop, and a different failure, not stopping, shows up behind it. That would also explain why the paper needed a much larger token budget to show its accuracy gain. This rests on 5 answers, so treat it as something to check, not a finding. (Update: with the paper's sampling at 16,384 tokens, none of the 20 paper-edit answers did this, so it looks like a side effect of greedy decoding. See the last Qwen3.5 section below.)

### Qwen3.5-0.8B at a 16,384-token cap, greedy (same 20 problems)

Run on Nov 3 2026, with the same problems, seed, thinking setting and greedy decoding, and only the cap raised. About 38 to 39 minutes per eval.

| Version | Accuracy | Finished (and correct) | Correct but cut off | Hit token cap | Looping | Mean tail repetition |
|---|---|---|---|---|---|---|
| Base | 5% (1) | 1 (1) | 0 | 95% (19) | 95% (19) | 0.66 |
| Paper's edit | 25% (5) | 2 (2) | 3 | 90% (18) | 75% (15) | 0.63 |
| Claude's edit | 10% (2) | 0 (0) | 2 | 100% (20) | 95% (19) | 0.71 |

What this shows (observed):
- **Four times the room finished nothing new.** The answers that finished are the same ones that finished at 4096, at the same lengths (1,046 tokens for base; 3,261 and 3,271 for the paper's edit). Greedy decoding is deterministic, so each answer's first 4,096 tokens are identical to the 4096 run, and every answer still running at 4,096 tokens ran on to 16,384 without stopping. With greedy decoding, the cap is not what holds accuracy down. A model in a loop under greedy decoding stays in it.
- **The order held a third time.** The paper's edit loops least (15 of 20) and Claude's edit and base loop most (19 of 20). The gap is 4 answers on 20 problems, the same size as at 4096.
- **The same five wrap-up loops.** The paper edit's five "The reasoning is clear. I will write the response." loops are the same five problems as at 4096, still repeating the line 12,000 tokens later. Base and Claude's edit have none. One more paper-edit answer closed its thinking block and then looped inside the written answer.
- **Accuracy is the same as at 4096 for the edits, and base lost its one cut-off "correct" answer** (the last number in the longer cut-off text changed). That confirms cut-off scoring is noise.

### Qwen3.5-0.8B at a 16,384-token cap with the paper's sampling (same 20 problems)

Why this was run: the paper scored every answer by sampling, not greedy decoding (see Settings). Its presence penalty pushes the model away from tokens it has already written, which works directly against loops. The kit gained `--sampling paper` for this run. Everything else matches the greedy 16,384 run above. About 42 minutes per eval.

| Version | Accuracy | Finished (and correct) | Correct but cut off | Hit token cap | Looping | Mean tail repetition |
|---|---|---|---|---|---|---|
| Base | 5% (1) | 1 (1) | 0 | 95% (19) | 90% (18) | 0.67 |
| Paper's edit | 55% (11) | 12 (10) | 1 | 40% (8) | 10% (2) | 0.17 |
| Claude's edit | 10% (2) | 2 (2) | 0 | 90% (18) | 90% (18) | 0.62 |

What this shows (observed):
- **The accuracy gain showed up.** The paper's edit got 11 right against 1 for base, and 10 of the 11 finished with a boxed answer, so this isn't cut-off noise. Two finished answers were wrong (test-1070 and test-590).
- **Its answers stopped on their own.** 12 of 20 finished, from 750 to 12,378 tokens long (median about 4,250). Every finished answer closed its thinking block first, and none of the 20 got stuck repeating a wrap-up line.
- **What is left in its 8 capped answers.** Two collapsed into long runs of spaces, which count as looping. Five were still deliberating ("Wait, ...", "So I will stick with 32.5.") without repeating closely enough to count as looping. One was cut off with the right number last.
- **Base and Claude's edit barely changed.** Base finished 1 answer (test-1101 at 7,698 tokens; under greedy its one finished answer was test-480 at 1,046) and looped on 18. Claude's edit finished 2, both correct, and looped on 18. So the paper's sampling, presence penalty included, doesn't break this model's loops by itself. A likely reason (inferred): the penalty takes a flat 1.5 off every token already used once, and after a few thousand tokens nearly every common token has been used, so it stops telling a loop apart from progress.
- **How likely chance is (computed, two-sided Fisher exact test on 20 vs 20).** Paper's edit vs base: accuracy 11 vs 1, p = 0.001; finished 12 vs 1, p = 0.0004; looping 2 vs 18, p < 0.0001. Claude's edit vs base: 2 vs 1, p = 1.0. Paper's edit, greedy vs sampled: accuracy 5 vs 11, p = 0.11, though finished went from 2 to 12.
- **Compared with the paper.** Their final check went from 10.6% to 60.4% on 1,152 held-out practice problems at 32,768 tokens. Ours went from 5% to 55% on 20 test problems at 16,384. Their edited model's correct answers had a median length of 9,172 tokens; ours finished shorter (median about 4,150 for the 10 correct ones), probably because a 16,384 cap cuts off the long ones (inferred). That would also mean the 32,768 cap could add a few more correct answers.

### Qwen3.5-0.8B, 50 test problems and 3 sampling seeds (paper's sampling, 16,384 cap)

Run on Nov 3 to 4 2026 to pin down the size of the 20-problem result. The same 50 test problems (`--n 50 --seed 0`) were scored six times: base and the paper's edit, each with sampling seeds 0, 1 and 2 (`--sample-seed`, added to `eval.py` for this run). Everything else matches the 20-problem sampled run above.

| Version | Sampling seed | Accuracy | Finished (and correct) | Hit token cap | Looping | Minutes |
|---|---|---|---|---|---|---|
| Base | 0 | 14% (7) | 4 (4) | 92% (46) | 90% (45) | 125.5 |
| Base | 1 | 18% (9) | 6 (6) | 88% (44) | 84% (42) | 122.1 |
| Base | 2 | 20% (10) | 6 (6) | 88% (44) | 86% (43) | 181.5 |
| **Base, pooled** | | **17.3% (26 of 150)** | 16 (16) | 89.3% | 86.7% | |
| Paper's edit | 0 | 66% (33) | 37 (32) | 26% (13) | 8% (4) | 105.2 |
| Paper's edit | 1 | 62% (31) | 36 (30) | 28% (14) | 10% (5) | 120.3 |
| Paper's edit | 2 | 62% (31) | 37 (30) | 26% (13) | 6% (3) | 322.5 |
| **Paper's edit, pooled** | | **63.3% (95 of 150)** | 110 (92) | 26.7% | 8.0% | |

What this shows (observed unless marked):
- **The gain held on every seed.** The paper's edit beat base by 42 to 52 points on each seed, and the seeds agree with each other within 4 points for both versions. The 20-problem result (5% to 55%) sits inside this range once its small size is allowed for.
- **It fixes far more than it breaks.** Problem by problem, against base on the same seed, the edit turned a wrong answer right 26, 23 and 21 times, and turned a right answer wrong 0, 1 and 0 times: 70 fixed and 1 broken across 150 pairs.
- **Per problem (computed).** Counting how many of the 3 seeds each problem got right, the edit did better on 35 problems, worse on 1 and tied on 14 (two-sided sign test over problems, p = 1e-9). Base got 35 of the 50 wrong on every seed. The edit got 24 right on all three seeds and 13 wrong on all three.
- **Finished answers are mostly right, and base only finishes easy ones.** The edit finished 110 of 150 answers and 92 of those were correct (84%). Base finished 16 and all 16 were correct, which suggests (inferred) it only escapes the loop on problems short enough to solve before it starts second-guessing.
- **Compared with the paper.** Their final check went from 10.6% to 60.4% at 32,768 tokens, ours from 17.3% to 63.3% at 16,384 on 150 scored answers. The paper's held-out check fixed 579 and broke 5; ours fixed 70 and broke 1. The same shape, at about the same size.
- **The slow seed 2 runs are the Mac, not the model.** A separate job ran alongside them and pushed the machine to its swap limit; the paper edit's seed 2 run sat at 24 of 50 for about three hours. Its scores are in line with the other seeds.

### Random-direction controls (same 50 problems)

Why: to check whether the gain comes from the paper's direction or from any edit of that size. Two controls, both built by `kit/random_control.py` from a random direction orthogonal to both layers' repetition directions (cosine 0.0). They read only weights, never problems.

- **64-column control.** The random direction goes through the paper's formula on the same 64 columns per layer, scaled so each layer's change has the paper edit's Frobenius norm (1.749 for layer 15, 1.399 for layer 19). It changes 129,318 weights against the paper edit's 130,188. Scored on sampling seeds 0, 1 and 2. Seed 0 ran on Nov 3 to 4 and seeds 1 and 2 on Nov 4. It is one random direction (seed fixed), scored three times.
- **All-columns control (the paper's version).** The random direction goes through the formula on every column (3,584 per layer, `--all-columns`), with each layer's strength set to 4 × ‖u·W‖ / ‖v·W‖ (`--match-strength 4`), where u is the repetition direction and v the random one. That is how the paper matched its control to the strength-4, every-column version of its edit. The resulting strengths, 4.456 for layer 15 and 4.284 for layer 19, match the ones in the paper's log. It changes 6,975,377 weights, 14.1% and 13.5% of each matrix's norm (`edits/qwen35-paper-style-control/manifest.json`). Scored on sampling seed 0 only.

| Version | Sampling seed | Accuracy | Finished (and correct) | Hit token cap | Looping | Minutes |
|---|---|---|---|---|---|---|
| Base | 0 | 14% (7) | 4 (4) | 92% (46) | 90% (45) | 125.5 |
| Base | 1 | 18% (9) | 6 (6) | 88% (44) | 84% (42) | 122.1 |
| Base | 2 | 20% (10) | 6 (6) | 88% (44) | 86% (43) | 181.5 |
| **Base, pooled** | | **17.3% (26 of 150)** | 16 (16) | 89.3% | 86.7% | |
| 64-column control | 0 | 34% (17) | 15 (15) | 70% (35) | 60% (30) | 209.3 |
| 64-column control | 1 | 18% (9) | 6 (6) | 88% (44) | 86% (43) | 123.7 |
| 64-column control | 2 | 28% (14) | 9 (9) | 82% (41) | 74% (37) | 115.0 |
| **64-column control, pooled** | | **26.7% (40 of 150)** | 30 (30) | 80.0% | 73.3% | |
| All-columns control | 0 | 20% (10) | 7 (7) | 86% (43) | 78% (39) | 160.2 |
| **Paper's edit, pooled** | | **63.3% (95 of 150)** | 110 (92) | 26.7% | 8.0% | |

What this shows (observed unless marked):
- **The 64-column control helps a little, and much less than seed 0 suggested.** Against base on the same seed it fixed 12 and broke 2 on seed 0, 4 and 4 on seed 1, and 9 and 5 on seed 2: 25 fixed and 11 broken in all (sign test p = 0.03). Looping fell from 130 to 110 of 150 (Fisher p = 0.006). Of the 46-point accuracy gain from base to the paper's edit, it recovers about 9, roughly a fifth. On seed 0 alone it looked like a third.
- **The paper's direction beats it on every seed.** Against the 64-column control, the paper's edit fixed 16 and broke 0 on seed 0, 22 and 0 on seed 1, and 19 and 2 on seed 2: 57 fixed and 2 broken (p = 6e-15). It looped on 12 answers against 110.
- **Spreading the change over every column removes almost all of the effect.** The all-columns control was not significantly different from base on seed 0 (6 fixed, 3 broken, p = 0.51; looping 45 to 39, p = 0.17), and did worse than the 64-column control on that seed (3 fixed, 10 broken, p = 0.09). The paper's edit fixed 24 problems against it and broke 1. This matches the paper's own control (+1.6 points), with one seed behind it.
- **Both 64-column edits hit those 64 neurons hard.** Each one changes those neurons' write weights by 85% of their norm in layer 15 and 62% in layer 19 (computed from the edit files). The paper's direction carries 10 to 14% of what those neurons write, and a random direction carries about 3%, so matching the size took a strength of 21 to 26 for the control against the paper's 6. A change that large to the neurons that write the loop direction most plausibly damages their loop-writing on its own, whatever the direction (inferred, not tested).
- **What the two controls say together (inferred).** A random change concentrated on the paper's 64 neurons per layer does a little on its own, most plausibly by damaging those neurons' loop-writing. The same kind of random change spread over every neuron does about nothing, as the paper found (+1.6 points, 14 to 16 of 128 practice answers at 32,768 tokens; `exp8_math/runs/p50c/tool_calls.jsonl`). Either way, most of the gain needs the paper's direction.

### Qwen2.5-0.5B (stand-in for Llama), 20 test problems

| Version | Weights changed | Accuracy | Hit token cap | Looping |
|---|---|---|---|---|
| Base | 0 | 45% (9) | 5% (1) | 5% (1) |
| Claude's edit | none made | n/a | n/a | n/a |

The subagent stopped, as the brief says to, because there was too little looping to fix. Its probe found 1 looping answer out of 200 practice answers, and its practice baseline had 0 of 24. There is no Claude-edit test file for this model because no edit exists.

## What Claude did, in its own reasoning

Each blind run was a fresh subagent. It got the verbatim brief from LAUNCH-PROMPT.md step 4 and could use only practice problems, `probe.py`, `edit_direction.py` and `eval.py --split practice`.

**Qwen3.5, attempt 1 (verbatim brief): stopped, nothing to fix.** It ran practice with the kit's default settings, thinking off and a 1024-token cap. There, 0 of 48 probe answers looped and only 1 of 24 practice answers did. It said the likely gap was the token cap and stopped rather than fit a direction to noise. It also flagged that `edits/_npy_p50c`, the paper's rebuilt tensors, sat in `edits/` with file names that give away the paper's layers. Before the next run I moved that folder and `edits/qwen35-paper` out of the kit.

**Qwen3.5, attempt 2 (verbatim brief plus one sentence).** Attempt 1 was decided by a settings mismatch, not by the model: the test scoring uses thinking on with a 2048 cap, while the brief never mentions `--thinking`. So I ran a second fresh subagent with one added sentence: "your edit will be scored with --thinking on --max-new-tokens 2048, so use those same two settings for every practice eval and probe run." It contained no test data. This is my addition to the protocol, so attempt 1 stays on the record as the verbatim result. In its own words, condensed:
- Practice baseline: 4 of 24 correct, 24 of 24 hit the cap, 20 of 24 looping.
- A second probe on 48 fresh practice problems (29 looping, 19 not) peaked in a middle band of layers: 14 (4.82), 10 (4.56), 13 (4.55), 11 and 12 (4.22). It picked layers 10 to 14 as the band to edit.
- Five candidates, each checked on 24 practice problems:

  | Candidate | Weights changed | Practice looping | Practice accuracy |
  |---|---|---|---|
  | None (baseline) | 0 | 0.833 | 0.167 |
  | 1: layers 10 and 14, top 64 columns, alpha 1 | 128,505 | 0.833 | 0.167 |
  | 2: layers 10 to 14, all columns, alpha 1 | 15,756,736 | 0.708 | 0.125 |
  | 3: layers 10 to 14, alpha 3 | 17,212,317 | 0.708 | 0.125 |
  | 4: layers 10 to 14, alpha -1 | 15,756,721 | 0.750 | 0.083 |
  | 5: layers 14, 19 and 22 | 9,382,736 | 0.792 | 0.083 |

- It kept candidate 1, "the only candidate that left accuracy alone", and said plainly that it does not fix looping. Pushing harder (alpha 3) tightened the loops into repeated "Let's check." lines. It concluded the direction "looks like a symptom of being stuck rather than something that causes it."
- Its own caveat: at 2048 tokens almost every answer is capped. So the probe compared repeating answers against other unfinished answers, never against answers that finished, and may never have targeted what keeps the model thinking.

**Qwen2.5 (verbatim brief): stopped, nothing to fix**, for the reason in the table above. It spent 1 of its 6 practice evals.

## What Claude read vs what code computed

Claude never read the model's weights. Code did all the large-scale work:
- `eval.py` generated the answers and scored them.
- `probe.py` re-ran each practice answer through the model, recorded what every layer's MLP writes over the last 64 tokens, and computed a looping-minus-finished direction per layer.
- `edit_direction.py` applied W' = W - alpha * u(u^T W) to the chosen columns.

What Claude actually read was the small output: the probe's per-layer table (separation and gap size for each of 24 layers) and the eval summaries. From those it chose layers, alpha and column count.

This is the table it read in Qwen3.5 attempt 2 (probe on 48 practice answers, 29 looping vs 19 not). Separation is the gap between looping and non-looping answers along that layer's direction, in pooled standard deviations:

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Separation | 2.24 | 2.58 | 2.91 | 3.07 | 3.22 | 3.04 | 3.63 | 3.70 | 3.75 | 3.92 | **4.56** | 4.22 |

| Layer | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Separation | 4.22 | 4.55 | **4.82** | 4.28 | 4.08 | 4.04 | 3.42 | 4.45 | 3.86 | 4.09 | 4.19 | 3.29 |

The paper's edit sits on layers 15 and 19 (4.28 and 4.45 here). Claude chose 10 and 14, the two highest by separation. In this table every layer from 6 up separates strongly, which is what you would expect when looping and not looping is mostly "how long has this answer been stuck". So the table didn't single out an obvious place to edit.

## How the paper found its edit (for comparison)

The blog post leaves this out, but the run's own notes (`exp8_math/runs/p50c/work_notes/final_report.md` in the logs) cover it. Several steps differ from what the blind Claude runs here did:
- **Matched pairs, not a group average.** It compared four matched pairs of contexts, each with one where the model kept making progress and one where it looped, and took the difference as the "repetition direction". Claude's probe instead averaged looping answers against other unfinished ones.
- **Dose tests.** It tried strengths 1, 2 and 4, checking that stronger edits made the looping text less likely while leaving the productive text almost unchanged. It also tried layer 15 alone and layer 19 alone, and dropped an attention-head alternative.
- **A random control.** An edit of the same size in a random direction gained only 1.6 points, about the same as no edit. It was matched to their strength-4 edit across every column, not to the final 64-neuron edit. Our control on the final edit's 64 neurons gained more (see the control section above).
- **Locked before the final check.** The final edit (strength 6, the 64 neurons per layer that write the direction most) was fixed before it was confirmed on 1,152 held-out practice problems: 579 fixed, 5 broken.
- **Scored with sampling and a 32,768 cap**, as in Settings. Every practice eval Claude ran here was greedy at 2048 tokens.

## What happened vs what we expected

| Expected (before the run) | What happened |
|---|---|
| The paper's edit reproduces on the Mac | Yes, once scored with the paper's sampling. On 50 problems and 3 seeds at 16,384 tokens, accuracy went from 17.3% to 63.3% and looping from 87% to 8% (observed). The first 20-problem check went from 5% to 55% and from 90% to 10%. Under greedy decoding looping fell at every cap (65% to 25% at 2048, 90% to 70% at 4096, 95% to 75% at 16,384), but almost no answer finished, so the accuracy gain never showed (observed). The paper's 32,768 cap is untested. |
| Claude makes its own loop-fixing edit on two families | No. On one model there was nothing to fix, and on the other none of the five edits worked on practice. The kept edit scored the same as base in every test run, greedy or sampled. |
| A random edit of the same size does about nothing (the paper's control) | Mostly. Spread over every column it did about nothing (14% to 20% on one seed, not significant). Concentrated on the paper's 64 neurons per layer it recovered about a fifth of the gain over 3 seeds (17.3% to 26.7%, p = 0.03). Seed 0 alone had suggested a third. The paper's direction beat both by a wide margin (observed). |
| Llama might rarely loop | Untested: Llama is gated. The stand-in, Qwen2.5-0.5B, rarely loops (5% test, 0.5% practice), the same pattern the paper saw with Gemma. |
| Qwen3.5 may be painfully slow on Apple Silicon | It wasn't. About 15 seconds per problem at 2048 tokens on the reference kernels. |

## What failed or needs fixing

- **setup.sh did not download any weights on the first run.** With `huggingface_hub` 2.4, `hf download --include` takes one pattern per flag, and the extra patterns were read as file names, so only config and tokenizer files arrived. Its skip checks also tested for JSON files that now existed. Fixed in setup.sh: one `--include` per pattern, and each check looks for a `.safetensors` file.
- **The step 4 brief is under-specified for Qwen3.5.** It doesn't say which decoding settings the test uses, and Qwen3.5's chat template turns thinking off unless asked. Attempt 1 was decided by that alone. If the brief is reused, add the settings sentence from attempt 2.
- **The kit leaks the paper's layer choice.** `paper_edit.py` writes `edits/_npy_p50c/` and `edits/qwen35-paper/` beside where the blind session works. Those file names give away the layers. They should be written outside `edits/`, or deleted after the paper eval.
- **LAUNCH-PROMPT.md is readable by the blind session.** The Qwen2.5 subagent read it. It holds the protocol but no results, so no test data leaked, but a cleaner blind run would hide it.
- **Blindness check (observed).** No subagent ran `eval.py --split test` or opened a `results/*-test.json` file or anything under `vendor/`. I checked this by searching each subagent transcript's commands and file paths. Their file listings did show test file names.
- **The kit only decoded greedily, which doesn't match how the paper scored.** Fixed in the kit: `kit/common.py` adds `PaperSampler`, which applies the presence penalty to tokens already generated and then temperature, top-k and top-p, in vLLM's order (transformers has no presence penalty of this kind). `kit/eval.py` adds `--sampling {greedy,paper}` and records the choice in each summary. Greedy stays the default so the earlier result files stay comparable. This is our reimplementation, checked against vLLM's documented order, not against vLLM itself.
- **Added for the 50-problem runs.** `kit/eval.py` gains `--sample-seed`, which seeds the sampling separately from `--seed` (which picks the problems), so one problem set can be scored under several sampling seeds. Summaries record both. `kit/random_control.py` builds the random-direction controls from the paper edit and the base weights; `--all-columns` and `--match-strength` give the paper's version.
- **Two runs slowed to a crawl on memory.** Another job on the Mac filled swap during base seed 2 and the paper edit's seed 2. One run sat at 24 of 50 for about three hours and then finished normally, so those minutes measure the machine, not the model. Long evals here need the Mac to themselves.
- **Not run:** Llama 3.2 1B (gated), the paper's 32,768 cap, the all-columns control on more than one seed, more than one random direction for the 64-column control, Claude's edit on 50 problems, and Claude's blind edit work redone with the paper's sampling.

## Worth trying next (not done here)

1. **Firm up the controls.** Done here: the 64-column control on 3 seeds and the all-columns control on one. Still open: a few more random directions for the 64-column control (each is one direction, about 2 hours per seed), the all-columns control on seeds 1 and 2, and the paper edit's 13 to 14 still-capped answers per seed rerun at 32,768 to see how many more finish.
2. **Redo Claude's blind attempt on the paper's terms.** Give the blind session the paper's sampling settings and the matched-pair method (progress vs loop from the same problem) instead of a group average. It's slow: each practice eval at 16,384 tokens takes about 40 minutes, so a session with 6 evals is about 4 hours of GPU time.
3. Accept Meta's license and run `hf auth login` (Kyle only), then rerun Llama. Or pick another small open model known to loop. The curated list at github.com/underlines/awesome-ml (llm-tools.md, "Open LLM models") is a place to start.

## Files

- `results/*-test.json`: test-split scores (base, paper edit, Claude's edit). The `*-test-4096.json` and `*-test-16384.json` files are the greedy reruns at those caps, and `*-test-16384-sampled.json` is the run with the paper's sampling (its summaries say `"sampling": "paper"`). The `*-test50-16384-sampled-s{0,1,2}.json` files are the 50-problem runs by sampling seed, `qwen3.5-0.8b-control-test50-16384-sampled-s{0,1,2}.json` are the 64-column random control, and `qwen3.5-0.8b-paperstyle-control-test50-16384-sampled-s0.json` is the all-columns control. The base greedy 16,384 file has no `sampling` key because it ran before the flag existed; it is greedy. The paper's edit was moved back into `edits/qwen35-paper` for these reruns, after the blind runs had finished, so every paper-edit file records the same overrides path.
- `results/*-practice.json`: the subagents' practice runs. `qwen3.5-0.8b-attempt1-*` is attempt 1. `qwen3.5-0.8b-claude-*` is attempt 2, whose `overrides` fields point at `edits/claude-cand1` to `5`.
- `edits/*/manifest.json`: Claude's kept edit, the five attempt 2 candidates, `qwen35-random-control` (the 64-column control's per-layer norms and its cosine to the repetition directions), and `qwen35-paper-style-control` (the all-columns control's strengths and relative change). The `.safetensors` files stay local. The paper's edit has no manifest, because `paper_edit.py` doesn't write one; its numbers come from `results/qwen3.5-0.8b-paper-test.json`.
