# LOG

One entry per run, appended by exp run. Fill in "why" and "note".

## Open questions
- Are old-class params identical for ALL old classes, not only class 0?
- Is learned Sigma close to I, or close to the real class covariance?
- Is S1 New (-15.2 vs paper) just seed noise?
- Does one GMM cluster absorb the 1250 known samples in S1?
- Novelty count ~4000: how many false positives / false negatives?
- Why does S-pass early stop come earlier each stage (700 -> 572)?
- How does S0_All grow with SVI steps (5 -> 1000)? Smoke run gave 84.86 at 5 steps.

---

## 20260915_1330__extract__cifar100_dino-vitb16-sl
why:    features for CIFAR-100 reproduction (FT 32x4, extract batch 128)
result: no metrics (feature extraction)
note:   adopted run; commit inferred 42e4eae, dirty unknown

## 20260915_1646__cifar100_b50t5_seed0__baseline
why:    reproduce Table 2, seed 0
result: S0_All 91.84 | S1_New 67.90 | final_All 80.37 | M_d 76.92 | M_f 6.54
note:   S0 matches paper (+0.16); S1 New far below (83.10); M_d -2.22. One seed only.

## 20260915_195303__smoke
why:    pipeline test (exp + run_vbcgcd.sh + extractor), n_epochs 5
result:
note:   pipeline OK. S0_All 84.86 after only 5 SVI steps (vs 91.84 at 1000) -> direction of means may carry most accuracy (inference, untested)
