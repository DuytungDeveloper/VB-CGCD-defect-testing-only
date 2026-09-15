# Reproducibility notes — upstream VB-CGCD

Baseline: `upstream-baseline` = 7fe9501 (daihao42/VB-CGCD, MIT).
Machine: WSL2 Ubuntu 22.04, RTX 4060 Laptop 8GB (~6.9GB free), Python 3.12.14.

Two sections:
- **Part A — Issues**: problems in the upstream repo. Not caused by us.
- **Part B — Patches**: changes we made. Each says whether it can change results.

Paper-vs-code method differences live in `paper-code-discrepancies.md`, not here.

---

# Part A — Issues in the upstream repo

## A1. Unpinned transitive deps silently break JAX GPU

`requirements.txt` pins `jax==0.4.34` but leaves `optax` and `chex` unpinned.

What happens on a fresh install: pip resolves `optax 0.2.8` -> `chex 0.1.92` ->
upgrades `jax` to 0.11.1, while `jax-cuda12-pjrt` and `jax-cuda12-plugin` stay
at 0.4.34.

**Impact.** Version mismatch. JAX falls back to CPU with no error message.
`pip check` says "No broken requirements found" — it does not validate PJRT
plugins, so this passes every standard check.

**Working pin set.** jax / jaxlib / jax-cuda12-* == 0.4.34, chex == 0.1.87,
optax == 0.2.3.

**How to verify.** `python -c "import jax; print(jax.devices())"` must print
`[CudaDevice(id=0)]`, not `[CpuDevice(id=0)]`.

## A2. Missing and inconsistent entries in requirements.txt

Imported in code but **not listed**: `peft`, `transformers`, `optax`, `tqdm`,
`tensorboard`, `scipy`.

Installing the missing ones today pulls `transformers 5.x`, which requires
PyTorch >= 2.5 and conflicts with the pinned `torch==2.4.0`. Transformers
disables its PyTorch backend, then peft imports into the disabled path and
fails with `NameError: name 'nn' is not defined` in
`transformers/integrations/accelerate.py`.

**Working versions.** `transformers==4.46.3`, `peft==0.13.2` (keep torch 2.4.0).

Also inconsistent:
- `torchaudio==2.3.1` does not match `torch==2.4.0` and would downgrade torch.
  `torchaudio` is never imported.
- 16 `nvidia-*` packages are pinned by hand; they are transitive deps of the
  torch cu121 wheel and do not need pinning.

**Conclusion.** The repo cannot be installed from `requirements.txt` as
published.

## A3. Python version is undocumented

Code uses `typing.override` (PEP 698), so Python >= 3.12 is required. Stated
nowhere in the README or the paper.

## A4. JAX GPU is Linux-only

`jax-cuda12-plugin` has no Windows wheels. Paper does not state this.

## A5. BOFT butterfly factor is silently downgraded from 2 to 1

`sl_finetuned_model.py:126` sets `boft_n_butterfly_factor=2`. When peft cannot
build the CUDA extension `fbd_cuda`, `peft/tuners/boft/layer.py:95-96` catches
the exception and **sets the factor to 1**. The only signal is a `UserWarning`.

**Measured evidence.** `trainable params: 221,184`, which decomposes exactly as
factor=1:

```
  12 * (768/4)  * 16   attention.output.dense  =  36,864
+ 12 * (3072/4) * 16   layer.output.dense      = 147,456
+ 12 * 2 * 768         boft_s                  =  18,432
+ 12 * 2 * 768         bias="boft_only"        =  18,432
                                       total   = 221,184
```

With factor=2 the `boft_R` terms double, giving 405,504. So the number itself
tells you which configuration actually ran.

**Attempts to fix (all failed).**
1. `pip install ninja` -> error moved to `CUDA_HOME is not set`.
2. `conda install -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1` +
   `export CUDA_HOME=$CONDA_PREFIX` -> error moved to
   `fatal error: nv/target: No such file or directory`.
3. `conda install -c nvidia cuda-cccl=12.1` (downgrade from 13.3) -> still
   `Error building extension 'fbd_cuda'`.

Root cause: the pip torch wheel ships the CUDA runtime but no `nvcc`, and
hand-assembling CUDA packages in conda gives an inconsistent toolchain.

**Decision: run with factor=1.**
- The paper never mentions `boft_n_butterfly_factor`, so there is no evidence
  the authors ran factor=2.
- Installing torch via pip (no nvcc) is a very common setup, so the authors
  plausibly hit the same fallback without noticing.
- Forcing factor=2 would move away from the repo's default behaviour without
  any evidence that it moves closer to the paper.

**Suspect list.** If S0 differs from 91.68 by more than 2pp, come back here.

---

# Part B — Patches applied locally

Rule: adding files is free; editing upstream files needs its own commit with a
stated reason. Every patch below says whether it can change the numbers.

## PATCH 1 — add `@torch.no_grad()` to feature extraction

**File.** `feature_extractor/dino-cifar100.py`, one new line above line 17.

```python
@torch.no_grad()
def infer_features_labels(dino, data_loader, features_dir, labels_dir, device, args):
```

**Problem.** Line 21 calls `dino.eval()`, but `eval()` does not disable
autograd — it only switches dropout and normalization modes. The model passed
in is `lora_model.backbone`, a `PeftModel` whose 221,184 BOFT parameters still
have `requires_grad=True`. One such parameter in the path is enough for PyTorch
to build the autograd graph across almost the whole network, even though
`backward()` is never called. The graph is discarded each loop iteration.

Line 36 uses `features.cpu().data`. `.data` detaches the tensor *after* the
graph was already built, so it cleans up the symptom rather than preventing it.

**Can this change results? No.**
- Identical forward pass: same kernels, same order, same values.
- dropout / normalization are governed by `.eval()`, unrelated to grad mode.
- The resulting `.npy` files are bit-identical.
- Only peak memory and speed change.

**Why it is needed here** (measured with `scripts/vram_probe.py`):

| setting | peak alloc | activation |
|---|---|---|
| batch 32, with backward | 4.64 GB | 3.86 GB |
| batch 128, no `no_grad` (projected) | ~15 GB | ~15 GB |

Free VRAM is 6.93 GB, so batch 128 would OOM. With `no_grad` it runs fine.

**Classification.** Technical fix, not a numerical deviation. Do not add to the
suspect list if S0 differs.

## PATCH 2 — configurable fine-tune batch size + gradient accumulation

**Applied.** Line 76 -> `int(os.environ.get("FT_BATCH_SIZE", 128))`.
Lines 91-108 -> accumulation loop with `loss = cls_loss / accum`, a step every
`accum` batches, and an end-of-epoch flush for the remainder.

**Step count check (CIFAR-100, 50 base classes, 25,000 images):**
  paper  batch 128          -> ceil(25000/128) = 196 batches = 196 steps/epoch
  ours   batch 32, accum 4  -> ceil(25000/32)  = 782 batches
                            -> 782//4 = 195 steps + 1 flush   = 196 steps/epoch
  Match. Without the flush it would be 195 and the last 2 batches dropped.

**Defaults unchanged.** With no env vars: batch_size=128, accum=1, flush is a
no-op -> identical to upstream.

**Log change.** Per-batch print (7,820 lines/epoch at batch 32) replaced by a
per-epoch mean. `cls_loss` is logged before the `/accum` division so the value
stays comparable to upstream.

**File.** `feature_extractor/sl_finetuned_model.py`, lines 76 and 92-106.

**Why needed.** `batch_size = 128` is hardcoded and `finetune_dino()` never
receives `args`, so there is no CLI path. 8GB VRAM cannot fit batch 128 with
backward.

**Can this change results? YES, unless accumulation is used.** The optimizer is
`AdamW(lr=1e-3)` with no scheduler (`sl_finetuned_model.py:83`). Lowering the
batch size raises the number of `optimizer.step()` calls at the same learning
rate, which is a different optimization trajectory. Gradient accumulation keeps
the effective batch at 128 and the step count equal to the paper.

**Planned setting.** `FT_BATCH_SIZE=32`, `FT_ACCUM=4` -> effective batch 128.

---

# Environment versions actually used

```
python        3.12.14
torch         2.4.0+cu121
jax/jaxlib    0.4.34          (CudaDevice(id=0) confirmed)
numpyro       0.15.3
transformers  4.46.3
peft          0.13.2
scikit-learn  1.5.2
```

`sklearn` and `numpyro` versions matter more than they look: sklearn decides
`GaussianMixture` initialization (which produces the pseudo-labels), and
numpyro decides how `Trace_ELBO` scales the subsampled loss — the only loss
function in the whole system.

## PATCH 3 — raise DataLoader num_workers to 6

**Files.** `sl_finetuned_model.py:89` (2 -> 6),
`dino-cifar100.py:113` (unset, i.e. 0 -> 6).

**Can this change results? No.** `train_transforms` (dino-cifar100.py:92-99) is
Resize -> CenterCrop -> ToTensor -> Normalize. All deterministic, no random
augmentation, so worker RNG streams never enter the pipeline. `shuffle=True`
runs in the main process and does not depend on worker count.

**Measured effect on fine-tuning: small.**
  real run (batch 32, num_workers=2)          1.27  s/it
  synthetic input, no dataloader at all       1.178 s/it
  => I/O accounts for roughly 7% of fine-tune step time.

**Why keep it anyway.** The measurement above covers the fine-tune phase only,
where each batch is expensive (forward + backward). The extraction phase is
forward-only under no_grad, so per-batch compute is much cheaper while the
decode/resize cost is unchanged — the I/O share should be larger there. That
loader is exactly the one the authors left at the default of 0. Not measured
separately.

**Note on the upstream values.** The two loaders disagree (2 vs unset), which
suggests neither was tuned. On the paper's RTX 8000 with batch 128 the
difference would be invisible.

**Classification.** Technical fix, not a numerical deviation.

## PATCH 4 — save trainable params after each epoch

**File.** `sl_finetuned_model.py`, end of the epoch loop.

Saves only `requires_grad=True` params (298,084 ~ 1.2 MB) to `/tmp/ft_ckpt.pt`,
not the full state_dict (86M params, ~346 MB). Frozen weights are recoverable
from the HuggingFace checkpoint.

**Can this change results? No.** Write-only. Nothing is ever read back; no
resume logic exists.

**Why.** A forced Windows Update reboot killed a 2.2-hour run at epoch 7 with
nothing on disk. `/tmp` is ext4, not the drvfs mount (`/mnt/e`) that lost
directory metadata in that crash — `datasets/cifar100` came back as
`datasets/cifar10`, and `console.log` was padded with NUL bytes.

**Classification.** Technical fix, not a numerical deviation.

### A5 addendum — final build error identified
After `export CUDA_HOME=$CONDA_PREFIX`, compilation of both `.cpp` and `.cu`
succeeds; the build now fails only at the link step:

    gcc fbd_cuda.o fbd_cuda_kernel.cuda.o -shared ... -lcudart -o fbd_cuda.so
    /usr/bin/ld: cannot find -lcudart: No such file or directory

Likely cause: conda ships `libcudart.so.12` without the unversioned
`libcudart.so` symlink that `-lcudart` needs. Untested fix:
`ln -sf $CONDA_PREFIX/lib/libcudart.so.12 $CONDA_PREFIX/lib/libcudart.so`
plus `export TORCH_CUDA_ARCH_LIST="8.9"` and `LDFLAGS`/`LD_LIBRARY_PATH`
pointing at `$CONDA_PREFIX/lib`.

Not pursued — see the A5 decision. Note that building it would switch the
factor from 1 to 2, which changes the training configuration, not just speed.
