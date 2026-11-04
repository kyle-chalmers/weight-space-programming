# Launch prompt for Claude Code

This is the prompt that started the run, lightly edited: private folder paths became paths in this repo, and a
pointer to private research notes was removed. Step 4 holds the blind brief each editing subagent received word for
word (`{MODEL}` filled in). `results.md` describes what happened at each step, including the follow-up runs added
later.

```
Run the open-model weight-editing test in this repo end to end and record the results. Read README.md first.

Rules:
- Activate .venv for every Python command.
- Do not accept any license, create or paste any token, or sign in anywhere for me. If the Llama download fails because it is gated, tell me in one line and continue with qwen2.5-0.5b as the second model.
- No paid API calls. Never commit models/, vendor/ or any .safetensors file.
- GSM8K test problems are only for before/after scoring. The editing work in step 4 must never see results/*-test.json or test-split problems.

Steps:
1. Run ./setup.sh. Report which models downloaded.
2. Timing check: for each model run kit/eval.py --split test --n 2 --max-new-tokens 512 and note minutes. From that, choose --n (aim for 20, at least 10) and --max-new-tokens (1024 for Llama and Qwen2.5; up to 2048 with --thinking on for Qwen3.5) so each full eval takes under about 30 minutes. If Qwen3.5 is far too slow, use it only for step 3 and use qwen2.5-0.5b as the second model in step 4. Tell me the settings you picked before the long runs.
3. Baselines and the paper's edit: run the test-split eval on each model you'll use, saving to results/{model}-base-test.json. Then run kit/paper_edit.py --out edits/qwen35-paper and evaluate qwen3.5-0.8b with --overrides edits/qwen35-paper on the same test settings (results/qwen3.5-0.8b-paper-test.json). Use the same --n, --seed and --max-new-tokens for every before/after pair.
4. Blind edit by a subagent, once per model (Llama, and the second model). Start a fresh subagent with only this brief: "You are editing a language model's weights to stop it getting stuck in repetitive loops, with no training. Work in this repo with .venv active. You may use only kit/eval.py with --split practice, kit/probe.py and kit/edit_direction.py, on model {MODEL}. Do not open results/*-test.json or anything under vendor/. Budget: at most 6 practice evals of at most 24 problems each. Use probe.py to find which layers separate looping from finished answers, choose layers, alpha and columns with edit_direction.py, check each candidate on practice problems, and keep the one that cuts looping without hurting accuracy. If the probe says there is too little looping to fix, stop and say so. Save the chosen edit to edits/{model}-claude and report: what you read in the probe table, what you chose and why, practice before and after, and how many weights changed." Wait for it, then evaluate its edit on the test split with the step 3 settings (results/{model}-claude-test.json).
5. Write results.md: a table per model with accuracy, token-cap hits and looping for base, Claude's edit, and (Qwen3.5) the paper's edit; weights changed and share of the model; the subagent's reasoning in a few lines; and a plain statement of what Claude read (the probe's per-layer table) versus what code computed. Separate what happened from what we expected, and call out anything that failed or had nothing to fix.
6. Commit results.md, the results/*.json files and each edits/*/manifest.json on a new branch, open a PR, and give me the link. Don't merge it.
```
