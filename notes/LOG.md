# LOG

One entry per run, appended by exp run. Fill in "why" and "note".

## Open questions
- Are old-class params identical for ALL old classes, not only class 0?
- Is learned Sigma close to I, or close to the real class covariance?
- Why does S-pass early stop fall across stages (seen in all 3 seeds)? Guess: scaling factor depends on label_offset (mngmm.py:256-263)
- Does one GMM cluster absorb the 1250 known samples in S1?
- Novelty count ~4000: how many false positives / false negatives?
- Why does S-pass early stop come earlier each stage (700 -> 572)?
- How does S0_All grow with SVI steps (5 -> 1000)? Smoke run gave 84.86 at 5 steps.
- Is New acc lower when cluster_max_min is high? (15 stage-seed points available now)
- Why is S4_New so stable across seeds (std 0.32) while other stages are not?

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

## 20260915_200057__baseline_s1
why:    seed spread for Table 2 (seed 1)
result: S0_All 92.00 | S1_New 80.70 | final_All 81.75 | M_d 79.84 | M_f 6.80
note:   see baseline_s2 entry for the 3-seed summary

## 20260915_201338__baseline_s2
why:    seed spread for Table 2 (seed 2)
result: S0_All 91.96 | S1_New 76.40 | final_All 78.47 | M_d 74.84 | M_f 7.88
note:   3 seeds (s0-s2) vs paper: final_All 80.20 +-1.65 (81.23), M_d 77.20 +-2.51 (79.14),
        M_f 7.07 +-0.71 (6.94), S0_All 91.93 +-0.08 (91.68). Paper values inside seed spread.
        New acc depends strongly on seed (S1_New 67.9-80.7). n=3 is weak evidence, not equivalence.
        S-pass early stop falls across stages in all 3 seeds.
