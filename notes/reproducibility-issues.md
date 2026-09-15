# Reproducibility notes — VB-CGCD

Baseline: `upstream-baseline` = 7fe9501 (daihao42/VB-CGCD, MIT).
Machine: WSL2 Ubuntu 22.04, RTX 4060 Laptop 8GB, Python 3.12.14.

- **Part A** — problems in the upstream repo.
- **Part B** — changes we made. Each says whether it can change the numbers.
- **Part C** — run log.
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

`sl_finetuned_model.py:126` sets `boft_n_butterfly_factor=2`. When peft cannot
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

---

# Part B — Patches

Rule: adding files is free; editing an upstream file needs its own commit with
a stated reason.

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
Accumulation keeps the effective batch and the step count equal to the paper.

**Step count check** (CIFAR-100, 50 base classes, 25,000 images):

```
paper  batch 128         -> ceil(25000/128) = 196 batches      = 196 steps
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

**Can this change results? No.** Write-only; nothing is ever read back. There
is no resume logic.

**Why.** Two runs died mid-training with nothing on disk. `/tmp` is ext4, not
the drvfs mount (`/mnt/e`) that lost directory metadata in run 1.

---

# Part C — Run log

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

## Run 3 — pending

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