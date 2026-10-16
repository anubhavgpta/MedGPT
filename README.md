# MedGPT audit and reproducible continuation

All commands run in PowerShell from `D:\MedGPT`, using Python 3.13 / PyTorch 2.9.1+cu126. No pretrained weights or external APIs are used. Original source and checkpoint backups are in `Backups/Original`. SHA-256 comparison confirmed `MedGPT.pt` is unchanged. The datasets and PDFs were not changed.

## Findings and changes

The original Q/K/V split, head reshaping, attention scaling, triangular mask, learned position embeddings, pre-LayerNorm residual blocks, shifted targets, flattened cross-entropy, gradient clipping, AdamW updates and final-token sampling logits were correct. Training and inference vocabulary ordering was also consistent. There is no evidence of a fundamental Transformer correctness bug causing the gibberish.

The original training script always initialized a new model and overwrote the checkpoint, and saved neither optimizer state nor completed steps. Train.py now loads architecture and vocabulary directly from the checkpoint, strictly loads weights, restores optimizer and torch CPU/CUDA RNG states for new-format checkpoints, and writes separate outputs atomically. Legacy loading preserves weights but explicitly initializes a fresh AdamW optimizer. The legacy step of 1000 is an assumption from your training history, configurable with --legacy-step; it cannot be recovered from the original file. Historical optimizer moments and historical RNG state cannot be recovered.

Model.py retains every parameter and state-dict key; it adds rejection of empty sequences. Tokenizer.py can reconstruct an explicitly ordered checkpoint vocabulary and reports unsupported characters clearly. Generate.py and Train.py no longer execute expensive operations on import. Invalid top-k / temperature / top-p / repetition penalty inputs are rejected. Nucleus sampling retains the threshold-crossing token; repetition penalties are sign-aware and apply to unique tokens in the active context. For character models, leave the penalty at 1.0 initially because common repeated letters are necessary.

Runtime.py provides vectorized contiguous-window gathering, loss evaluation with an independent fixed RNG, checkpoint helpers and the absolute-update LR schedule. Evaluation therefore does not consume training batch RNG. Each optimizer update averages AccumulationSteps microbatches before clipping once. FP32 and the original explicit attention remain the safe baseline: no FlashAttention, BF16, Triton, AMP or architecture changes are required. GPU-resident corpus caching and alternative attention kernels are deferred until profiling shows a benefit.

## Training plan

TrainingConfig.json targets 10,000 TOTAL optimizer updates, batch 16, context 128, accumulation 1, LR 3e-4 decaying to 3e-5, and 200 warmup updates. Continuing at step 1000 uses the absolute schedule and does not restart warmup. The 10,000-step run was NOT launched. Evaluation occurs every 250 updates on the same 20 batches per split, checkpoints every 500, with best checkpoints at validation improvements of at least 0.002. Early stopping triggers after 12 evaluations without such improvement. Sampling noise and corpus leakage limit how strongly to interpret this rule.

Change learning rate and batch settings in the JSON before starting a new legacy continuation. New-format resumes require identical config so the schedule cannot silently reset. To reduce memory, choose batch 8 / accumulation 2 before a new experiment. The checkpoint's architecture is authoritative; incompatible weight shapes fail strict loading. No architecture override is exposed on resume.

Resume your original checkpoint into a new output:

```powershell
Set-Location D:\MedGPT
.\.venv\Scripts\python.exe Train.py --resume MedGPT.pt --legacy-step 1000 --output Runs/continued.pt --device cuda
```

Resume the tested 20-update smoke checkpoint, retaining its optimizer and RNG state:

```powershell
.\.venv\Scripts\python.exe Train.py --resume Runs/smoke.pt --output Runs/continued.pt --device cuda
```

For later resumes, select a new destination, for example:

```powershell
.\.venv\Scripts\python.exe Train.py --resume Runs/continued.pt --output Runs/continued-next.pt --device cuda
```

The output has an adjacent `.best.pt` and append-only `.jsonl` log. Ctrl+C saves the current completed-step state; partial accumulation gradients are discarded, so an interrupted partial update does not promise bit-identical replay. Atomic writes protect against a partially written destination, but process termination/power failure can still lose updates since the last periodic checkpoint. Fixed seeds support reproducibility on the same environment; bitwise results across hardware/PyTorch versions are not guaranteed. Keep dataset bytes unchanged for resume comparisons.

## Generate, evaluate and test

```powershell
.\.venv\Scripts\python.exe Generate.py --checkpoint Runs/smoke.pt --prompt "The human heart is" --max-new-tokens 300 --temperature 0.8 --top-k 20 --top-p 0.95 --repetition-penalty 1.0 --seed 42
.\.venv\Scripts\python.exe Generate.py --checkpoint Runs/smoke.pt --prompt "The nervous system" --compare
.\.venv\Scripts\python.exe Generate.py --checkpoint Runs/smoke.pt --prompt "Blood flows through" --greedy
.\.venv\Scripts\python.exe Generate.py --checkpoint Runs/smoke.pt
.\.venv\Scripts\python.exe Evaluate.py --checkpoints MedGPT.pt Runs/smoke.pt --output Runs/comparison.json --device cuda
.\.venv\Scripts\python.exe -m unittest TestMedGPT -v
.\.venv\Scripts\python.exe CleanMedicalData.py --report Runs/data-audit.json
```

Omit --prompt for interactive input. Top-k 0 disables top-k; top-p 1 disables nucleus filtering. All inference scripts support --device cpu. Tests cover future-token invariance, finite gradients and actual updates, shifted targets, loss semantics, tiny-batch overfitting, exact checkpoint logits, optimizer continuation, legacy loading, vocabulary order, schedule endpoints, sampling masks, seed repeatability and prompt validation.

## Actual results, 2026-10-16

CUDA was available on the NVIDIA GeForce GTX 1650. A 20-update continuation and a separate one-update optimizer-resume smoke test completed successfully. Same evaluation seed (123), batch 16, 20 windows/batches per split:

| Checkpoint | Completed updates | Train CE | Validation CE | Validation character perplexity |
|---|---:|---:|---:|---:|
| Original | 1000 (reported) | 2.358262 | 2.373192 | 10.731589 |
| Smoke continuation | 1020 | 2.354475 | 2.367081 | 10.666211 |
| Resume check | 1021 | 2.354385 | 2.367006 | — |

Both original and smoke outputs remain incoherent. The small loss decrease validates the continuation path; it does not establish a meaningful language-quality improvement. The first prompt's smoke output begins: `The human heart is cenctaned. colen Inatios pem.` Full samples for all five requested prompts are in Runs/comparison.json.

Mean measured generation speeds across five 200-character samples were approximately 222.5 and 274.2 characters/sec. These are short sequential measurements with warmup/order effects, not evidence of a speed improvement between identical architectures. Peak allocated PyTorch tensor memory during training was 93.87 MiB (excludes CUDA context, reserved allocator space and other processes). The final smoke update reported about 144,889 training tokens/sec; a cold resume update reported about 7,490, illustrating why a single-step figure is not a stable throughput benchmark. Training logging excludes evaluation/checkpoint time. Monitor whole-device usage separately with `nvidia-smi -l 1`.

Evaluation measures sampled character cross-entropy/perplexity, not full-corpus loss or word perplexity. The exact 50-character-window memorization screen found no matches in the generated continuations; it is not an exhaustive or semantic near-duplicate check and does not prove absence of memorization. Language-model loss does not establish medical factual accuracy or clinical safety.

## Data audit and next priorities

Runs/data-audit.json records Unicode distributions, broken-hyphen candidates, repeated paragraphs and overlap. Training contains 6,573 duplicate paragraph occurrences, including 2,308 `Access for free at openstax.org` footers. There are 37 normalized validation paragraphs of at least 100 characters present in training. Samples include headings fused to sentences and `semi- solid`; table-like learning-objective fragments recur. The original cleaning indiscriminately joined single line breaks and dehyphenated words, losing table/heading boundaries and sometimes meaningful hyphens. Conservative cleaning now preserves line boundaries and symbols, removes no content categories, and reports character counts when writing an explicitly requested new directory. It cannot reconstruct already-lost layout.

1. Continue the baseline with the supplied plan. At the original 1000 updates only 2,048,000 sampled character positions were optimized against a 12.67M-character train corpus (~0.16 corpus equivalents, with repeated sampling). At 10,000 updates this is ~1.62 equivalents. Review fixed samples and validation curves at each checkpoint; more training is necessary but coherent text is not guaranteed.
2. Use ExtractPDF.py to create separate document text and page-offset manifests in a new directory. Manually review headers, tables and OCR artifacts; preserve source attribution and medically meaningful punctuation/units. Hold out complete documents where practical, or whole chapters with explicit source/page manifests. Deduplicate across splits before training and keep the current dataset as the baseline. A paragraph-count tail split is neither a controlled chapter split nor leakage-free. Do not silently discard the corpus.
3. Once the baseline trajectory is measured, start a separate random-initialization experiment: embedding 192, 4 layers, 4 heads, context 256, microbatch 8, accumulation 2. Profile peak memory before increasing context to 512 or embedding to 256. Longer character context is useful but quadratic attention increases memory. The provided resume CLI deliberately does not migrate weights to this architecture; implement a distinct fresh-run configuration/checkpoint for that experiment.
4. Consider a separately versioned BPE tokenizer experiment only after the character baseline is evaluated. It needs fresh embeddings, output head, vocabulary metadata and training; never reinterpret existing character IDs as BPE IDs.

## Next phase: continue the 10,000-step checkpoint

NextTrainingConfig.json extends the unchanged character model to 20,000 total optimizer updates. It starts a new cosine segment at step 10,000, from the previous terminal LR 3e-5 down to 1e-5. There is no learning-rate jump and no warmup restart. This conservative experiment tests whether additional baseline training helps; it does not promise coherent sentences. AdamW moments, weights, vocabulary, architecture and torch RNG states are retained. Early-stop patience resets only on an explicitly requested new phase; the best validation loss is retained.

The new --new-phase flag permits only schedule changes after the previous plan has completed. Ordinary resumes still require the exact saved config. ScheduleStartStep is stored in that config so interrupted phase-two runs continue at the correct LR. Existing destination checkpoint, best-checkpoint or log files now cause an error: choose a fresh output name. Logging reports the LR actually used for the completed update.

Start the full second phase yourself (not automatically launched):

```powershell
.\.venv\Scripts\python.exe Train.py --config NextTrainingConfig.json --resume Runs/continued.pt --output Runs/phase2.pt --new-phase --device cuda
```

Alternatively, continue from the newly tested phase-two smoke checkpoint without repeating those updates:

```powershell
.\.venv\Scripts\python.exe Train.py --config NextTrainingConfig.json --resume Runs/phase2-smoke.pt --output Runs/phase2.pt --device cuda
```

Choose one of these commands. For an interrupted phase-two run, resume Runs/phase2.pt into a new output name with NextTrainingConfig.json and WITHOUT --new-phase. Use the original TrainingConfig.json for original-phase checkpoints.

Generate from the actual next checkpoint:

```powershell
.\.venv\Scripts\python.exe Generate.py --checkpoint Runs/phase2.pt --prompt "The human heart is" --compare
.\.venv\Scripts\python.exe Evaluate.py --checkpoints Runs/continued.pt Runs/phase2.pt --output Runs/evaluation-phase2.json --device cuda
```

The evaluation of the completed first phase is saved in Runs/evaluation-10000.json, including all five fixed prompts: train CE 1.478722, validation CE 1.498917, validation character perplexity 4.476837. Original source versions before this change are saved in Backups/BeforeNextPhase. No model architecture or corpus changes were made.

Phase-two validation: all six unittest cases passed. A 20-update CUDA smoke run produced Runs/phase2-smoke.pt at step 10020, with train CE 1.478843 and validation CE 1.499576. This slight validation increase over step 10000 is not evidence of improvement; use the full phase's validation trend and early stopping to decide whether continuation is worthwhile. The original Runs/continued.pt remains available as the better measured baseline.
