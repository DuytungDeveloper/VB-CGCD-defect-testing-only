# Reproducibility notes — VB-CGCD

Baseline: `upstream-baseline` = 7fe9501 (daihao42/VB-CGCD, MIT).
Machine: WSL2 Ubuntu 22.04, RTX 4060 Laptop 8GB, Python 3.12.14.

- **Part A** — problems in the upstream repo.
- **Part B** — changes we made. Each says whether it can change the numbers.
- **Part C** — run log (frozen; new runs are in `runs/` and `notes/LOG.md`).
- **Part D** — environment.

Paper-vs-code method differences are in `paper-code-discrepancies.md`.

---

# Part A — Upstream issues

## A1. Unpinned deps silently break JAX GPU

`requirements.txt` pins `jax==0.4.34` but leaves `optax` and `chex` unpinned.
A fresh install resolves `optax 0.2.8` -> `chex 0.1.92` -> upgrades `jax` to
0.11.1, while `jax-cuda12-pjrt` and `jax-cuda12-plugin` stay at 0.4.34.

JAX then falls back to CPU with no error. `pip check` reports nothing wrong —
it does not validate PJRT plugins.

Working pins: `jax/jaxlib/jax-cuda12-* == 0.4.34`, `chex == 0.1.87`,
`optax == 0.2.3`.

Verify with `python -c "import jax; print(jax.devices())"` -> must show
`[CudaDevice(id=0)]`.

## A2. requirements.txt is incomplete and inconsistent

Imported in code but not listed: `peft`, `transformers`, `optax`, `tqdm`,
`tensorboard`, `scipy`.

Installing them today pulls `transformers 5.x`, which needs PyTorch >= 2.5 and
conflicts with the pinned `torch==2.4.0`. Transformers disables its PyTorch
backend, peft then imports into the disabled path, and it fails with
`NameError: name 'nn' is not defined`.

Working versions: `transformers==4.46.3`, `peft==0.13.2`, torch stays 2.4.0.

Also: `torchaudio==2.3.1` conflicts with `torch==2.4.0` and is never imported;
16 `nvidia-*` packages are pinned by hand although they ship with the torch
cu121 wheel.

**The repo cannot be installed from `requirements.txt` as published.**

## A3. Python version is undocumented

Code uses `typing.override` (PEP 698), so Python >= 3.12 is required. Not
stated in the README or the paper.

## A4. JAX GPU is Linux-only

`jax-cuda12-plugin` has no Windows wheels. Not stated in the paper.

## A5. BOFT butterfly factor silently drops from 2 to 1

`sl_finetuned_model.py` (search `boft_n_butterfly_factor`) sets
`boft_n_butterfly_factor=2`. When peft cannot
build the CUDA extension `fbd_cuda`, `peft/tuners/boft/layer.py:95-96` catches
the exception and sets the factor to 1. The only signal is a `UserWarning`.

Measured: `trainable params: 221,184`, which is exactly factor=1:

```
  12 * (768/4)  * 16   attention.output.dense  =  36,864
+ 12 * (3072/4) * 16   layer.output.dense      = 147,456
+ 12 * 2 * 768         boft_s                  =  18,432
+ 12 * 2 * 768         bias="boft_only"        =  18,432
                                        total  = 221,184
```

Factor=2 would double the `boft_R` terms -> 405,504. The number tells you
which configuration actually ran.

The same decomposition shows which modules are adapted: only the 24
`output.dense` modules. `intermediate.dense` (1st MLP layer) is not adapted,
because `mlp.fc1` / `mlp.fc2` in `target_modules` are timm names that do not
exist in the Hugging Face ViT.

**Attempts to fix, in order.**
1. `pip install ninja` -> error moved to `CUDA_HOME is not set`.
2. `conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1` +
   `export CUDA_HOME=$CONDA_PREFIX` -> error moved to
   `fatal error: nv/target: No such file or directory`.
3. `conda install -c nvidia cuda-cccl=12.1` (down from 13.3) -> compilation
   now succeeds, linking fails: `/usr/bin/ld: cannot find -lcudart`.

Likely remaining cause: conda ships `libcudart.so.12` without the unversioned
`libcudart.so` symlink that `-lcudart` needs. Untested fix:
`ln -sf $CONDA_PREFIX/lib/libcudart.so.12 $CONDA_PREFIX/lib/libcudart.so`,
plus `TORCH_CUDA_ARCH_LIST="8.9"` and `LDFLAGS`/`LD_LIBRARY_PATH` pointing at
`$CONDA_PREFIX/lib`.

Root cause: the pip torch wheel ships the CUDA runtime but no `nvcc`, and
hand-assembling CUDA packages in conda gives an inconsistent toolchain.

**Decision: run with factor=1.** The paper never mentions this parameter, so
there is no evidence the authors ran factor=2. Installing torch via pip is a
common setup, so they plausibly hit the same fallback. Forcing factor=2 would
move away from the repo's default without evidence that it moves closer to the
paper — and it changes the training configuration, not just speed.

**Suspect list.** If S0 differs from 91.68 by more than 2pp, come back here.

## A6. Phase 2 on the GPU is not bit-reproducible

Measured on 2026-10-15 with `gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s0` (the steps of `lab/replay.py`,
which call the functions of `main.py`; same input in every process):

- Every printed result equals the run of 2026-10-05: S0 and S1 tables, kept count, early-stop steps,
  cluster sizes, `pi_test`.
- The class means and covariances do not: at most 1.7e-7 of the largest value (float32 epsilon is
  1.2e-7). Two runs on the same day also differ at this level (max |diff| 6e-8).
- With `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, two runs are identical to the bit.

Likely cause (not verified): in the SVI model the gradient of `class_means[y_batch]` is a scatter-add,
which XLA runs on the GPU with atomic adds, so the order of the float additions changes from run to run.
The GPU driver also changed in between: 616.92 on 2026-10-05, 617.42 on 2026-10-15.

`main.py` does not set the flag, so its runs repeat only up to this noise. `lab/replay.py` v2 sets it
(PL-QD6); its check against `main.py` runs (C1) compares the parameters within 1e-6 of their largest value.

---

# Part B — Patches

Rule: adding files is free; editing an upstream file needs its own commit with
a stated reason.

Scope: Part B lists only the patches needed to run the upstream pipeline on this machine
(PATCH 1-5). Experimental variants added later (FT_SEED, FT_AUG, NT-Xent, FT_LABELED_IDX/FT_FOLD,
k-means, --gmm_n_init) are listed in `code-changes.vi.md` and discussed in D18/D19.

## PATCH 1 — `@torch.no_grad()` on feature extraction

**File.** `dino-cifar100.py`, one new line above line 17.

```python
@torch.no_grad()
def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):
```

**Problem.** Line 21 calls `dino.eval()`, but `eval()` does not disable
autograd. The model is a `PeftModel` whose 221,184 BOFT parameters still have
`requires_grad=True`, so PyTorch builds the autograd graph across the whole
network even though `backward()` is never called. Line 36 uses
`features.cpu().data`, which detaches *after* the graph was already built.

**Can this change results? No.** Same kernels, same order, same values. The
`.npy` output is bit-identical. Only memory and speed change.

**Why needed** (measured, `scripts/vram_probe.py`):

| setting | peak alloc |
|---|---|
| batch 32, with backward | 4.64 GB |
| batch 128, no `no_grad` (projected) | ~15 GB |

Free VRAM is ~6.9 GB, so batch 128 would OOM without this.

## PATCH 2 — configurable batch size + gradient accumulation

**File.** `sl_finetuned_model.py`, line 76 and the epoch loop.

Line 76 becomes `int(os.environ.get("FT_BATCH_SIZE", 128))`. The loop divides
the loss by `accum`, steps every `accum` batches, and flushes the remainder at
the end of each epoch.

**Why needed.** `batch_size = 128` is hardcoded and `finetune_dino()` never
receives `args`, so there is no CLI path. 8GB VRAM cannot fit batch 128 with
backward.

**Can this change results? YES, unless accumulation is used.** The optimizer is
`AdamW(lr=1e-3)` with no scheduler. A smaller batch means more
`optimizer.step()` calls at the same learning rate — a different trajectory.
Accumulation keeps the effective batch and the step count equal to upstream code (the paper gives no batch size).

**Step count check** (CIFAR-100, 50 base classes, 25,000 images):

```
upstream batch 128       -> ceil(25000/128) = 196 batches      = 196 steps
ours   batch 32, accum 4 -> ceil(25000/32)  = 782 batches
                         -> 782//4 = 195 steps + 1 flush       = 196 steps
```

Without the flush it would be 195, dropping the last 2 batches per epoch.

**Defaults unchanged.** With no env vars: batch 128, accum 1, flush is a no-op
-> identical to upstream.

**Log change.** Per-batch print (7,820 lines/epoch at batch 32) replaced by a
per-epoch mean. `cls_loss` is logged before the `/accum` division, so the value
stays comparable to upstream.

**Setting used.** `FT_BATCH_SIZE=32`, `FT_ACCUM=4` -> effective batch 128.

**Correction (2026-09-17). 32x4 is NOT fully equivalent to batch 128.**
BOFT dropout (`peft/tuners/boft/layer.py:141+`) draws a new mask on every forward call.
The mask acts on `boft_R` (shape factor+1, n_block, H, H), not on the batch. So 32x4 averages
4 masks per optimizer step; batch 128 uses 1. The extra draws also use the global CPU RNG,
so sample order likely differs from epoch 1 on (RandomSampler seeds from it). The last batch
of each epoch is also weighted differently (0.25/0.25 vs 0.8/0.2). Measured effect looks small
(S0 91.90 vs paper 91.68). The exception in "YES, unless accumulation is used" is wrong:
effective batch and step count match upstream, but the training trajectory does not.

## PATCH 3 — raise `num_workers` to 6 — REVERTED

**Files.** `sl_finetuned_model.py:89` (2 -> 6), `dino-cifar100.py:113`
(unset -> 6). Both restored to upstream values.

**Measured effect: small.**

```
real run (batch 32, num_workers=2)      1.27  s/it
synthetic input, no dataloader at all   1.178 s/it
=> I/O is ~7% of fine-tune step time
```

**Why reverted.** Run 2 hung at epoch 3 while run 1 reached epoch 7, and
`num_workers` was the only variable that differed. Reverting isolates it, and
the 7% measurement means there was little to lose.

**This is isolation, not a proven fix.** The py-spy native stack pointed at
`cuStreamSynchronize` inside `libcuda.so`. DataLoader workers are separate
processes with no CUDA context — a stuck worker would block the main thread in
`dataloader.py` on a queue, not inside libcuda. So `num_workers` is unlikely to
be the real cause.

**Note.** The two upstream loaders disagree (2 vs unset), so neither looks
tuned. On the paper's RTX 8000 with batch 128 the difference would be
invisible.

## PATCH 4 — save trainable params each epoch

**File.** `sl_finetuned_model.py`, end of the epoch loop.

Saves only `requires_grad=True` params (298,084 ~ 1.2 MB) to
`/tmp/ft_ckpt.pt`, not the full state_dict (86M params, ~346 MB). Frozen
weights are recoverable from the HuggingFace checkpoint.

298,084 = 221,184 (BOFT, see A5) + 76,900 (classification head, 768 x 100 + 100).

**Exception: the pooler.** The HF ViT pooler is NOT in the HF checkpoint
(random init, frozen) and is NOT in `ft_ckpt.pt` (`requires_grad=False`).
Checked 2026-09-16: two fresh processes gave the same pooler checksum
(-23.905103495538352), so the seeded init repeats in this environment.
Not guaranteed after a torch/transformers version change.

**Other caveats.** The fixed path `/tmp/ft_ckpt.pt` is overwritten by every run
and is not linked to a run id. `/tmp` may be wiped at boot if systemd tmpfiles
is active (not checked yet).

**Can this change results? No.** Write-only; nothing is ever read back. There
is no resume logic.

**Why.** Two runs died mid-training with nothing on disk. `/tmp` is ext4, not
the drvfs mount (`/mnt/e`) that lost directory metadata in run 1.

**Fixed by PATCH 5** (path and pooler caveats).

## PATCH 5 — checkpoint path from env + keep the pooler

**File.** `sl_finetuned_model.py`, end of the epoch loop (search `PATCH 5`).

`FT_CKPT_PATH` sets the checkpoint path; default stays `/tmp/ft_ckpt.pt`.
The parent folder is created if needed. The saved dict now also holds
`pooler.dense.weight` (768x768) and `pooler.dense.bias` (768): 590,592 params, ~2.4 MB.
Total saved: 298,084 + 590,592 = 888,676 params.

**Can this change results? No.** Still write-only, after `optimizer.step()`;
`.detach().cpu()` copies; no random numbers are drawn.

**Why.** PATCH 4 wrote every run to the same `/tmp` file. The pooler init is
reproducible in this environment (see PATCH 4), but saving it removes the
dependency on the RNG sequence and library versions.

**Verify during the next extraction.** After `Epoch 0`, the checkpoint must hold
888,676 params including 2 pooler keys.

---

# Part C — Run log

Frozen on 2026-09-16. New runs are recorded in `runs/` and `notes/LOG.md`.

## Run 1 — 2026-09-15 02:17 — killed by Windows Update reboot

`FT_BATCH_SIZE=32 FT_ACCUM=4`, num_workers 6/6. Reached epoch 7, mean CE
0.0382.

A forced reboot killed it. `console.log` ended padded with NUL bytes and
`datasets/cifar100` came back as `datasets/cifar10` — drvfs lost metadata on
the hard reset. Cause unrelated to the code.

## Run 2 — 2026-09-15 08:30 — GPU hang at epoch 3

`FT_BATCH_SIZE=32 FT_ACCUM=4`, num_workers 6/6. Frozen at
`382it [07:51, 1.24s/it]` for 2 hours.

```
py-spy --native:
  THPVariable_item -> _local_scalar_dense_cuda
    -> c10::cuda::memcpy_and_sync -> cudaStreamSynchronize
    -> cuStreamSynchronize (libcuda.so.1.1)     <-- stuck here
nvidia-smi: 6320 MiB held, utilization 6%
/proc/PID/io: write_bytes not advancing
```

GPU-side hang inside the NVIDIA driver, not a Python deadlock. `.item()`
copies one scalar but must wait for the whole stream, so it spins at 99.9% CPU
on one core. Any sync point would block identically, including upstream's
per-batch print.

Suspect: WDDM under sustained fp32 load with only ~1.6 GB headroom (Windows
held ~2.2 GB of the 8 GB).

**Changes before run 3.** Reverted `num_workers` (PATCH 3). Closed GPU apps on
Windows — baseline VRAM dropped from ~2.2 GB to 805 MiB, giving ~2.7 GB
headroom instead of ~1.6 GB.

## Run 3 — 2026-09-15 13:30 — completed

Features used by baseline seeds 0-2. Moved to `archive/pre-rerun-20260915/`
on 2026-09-16; packed into a tar on drive E: on 2026-10-15 (path and restore
command at the top of `notes/LOG.md`).

Run 1 is one reason features now live inside the project (ext4), not on
`/mnt/e` (drvfs).

---

# Part D — Environment

```
python        3.12.14
torch         2.4.0+cu121
jax/jaxlib    0.4.34          (CudaDevice(id=0) confirmed)
numpyro       0.15.3
transformers  4.46.3
peft          0.13.2
scikit-learn  1.5.2
```

`sklearn` and `numpyro` versions matter more than they look. sklearn decides
`GaussianMixture` initialization, which produces the pseudo-labels. numpyro
decides how `Trace_ELBO` scales the subsampled loss — the only loss function
in the whole system.

`requirements-lock.txt` was regenerated on 2026-09-16 from this working env
(`pip list --format=freeze`, pip line removed). The first lock (commit fd8316b,
2026-09-14) captured the broken state from A2 (transformers 5.17.0, peft 0.20.0,
tokenizers 0.23.2, huggingface_hub 1.31.0), had a conda `@ file://` line, and had no
`--extra-index-url` for torch cu121, so it could not be installed.

Line 1 of the 2026-09-16 lock also held terminal escape codes (VS Code shell
integration, OSC 633), captured with the `echo` that wrote it, so `pip install -r`
stopped at line 1 ("Invalid requirement"). Fixed on 2026-10-15. The pins match the
env of the runs of 2026-10-04/05 (`runs/envs/f70b1aeb7805.txt`). The env of the runs
of 2026-10-07 (`893900df6375.txt`) adds 21 Jupyter packages (ipykernel, ipython,
jupyter_client and dependencies) and changes no pinned version. No fresh install
has been tested yet.

`environment.yml` (commit fd8316b) had the same broken pins and a machine-specific `prefix`.
Removed on 2026-09-17; `requirements-lock.txt` is the only environment file.

The lock only covers pip packages. Conda packages from A5 (cuda-nvcc, cuda-cccl) are not in it.
`ninja` is in the lock: on a machine with a complete CUDA toolchain, `fbd_cuda` may build and
the butterfly factor becomes 2 (a different training setup). `check_phase1.py` catches this:
the checkpoint would hold 1,072,996 params instead of 888,676.