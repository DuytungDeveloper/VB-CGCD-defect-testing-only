# Reproducibility issues found in upstream VB-CGCD

## 1. Unpinned transitive dependencies break JAX GPU silently
`requirements.txt` pins `jax==0.4.34` but leaves `optax` and `chex` unpinned.
Installing today: pip resolves optax 0.2.8 -> chex 0.1.92 -> upgrades jax to 0.11.1,
while `jax-cuda12-pjrt` / `jax-cuda12-plugin` stay at 0.4.34.
Result: version mismatch, JAX falls back to CPU with NO error message.
`pip check` reports "No broken requirements found" — it does not validate PJRT plugins.
Working pin set: jax/jaxlib/jax-cuda12-* == 0.4.34, chex == 0.1.87, optax == 0.2.3.

## 2. requirements.txt inconsistencies
- `torchaudio==2.3.1` mismatched with `torch==2.4.0` (would downgrade torch); torchaudio never imported.
- Missing from requirements but imported in code: peft, transformers, tqdm, tensorboard, scipy.
- 16 `nvidia-*` packages pinned manually; these are transitive deps of torch's cu121 wheel.

## 3. Python version undocumented
Code uses `typing.override` (PEP 698) -> requires Python >= 3.12. Not stated anywhere.

## 4. JAX GPU is Linux-only
`jax-cuda12-plugin` has no Windows wheels. Paper does not state this constraint.
