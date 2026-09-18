# LOG

One entry per run, appended by exp run. Fill in "why" and "note".
Before the clean rerun: archive/pre-rerun-20260915/ (not in git) and git tag pre-rerun-20260915

## Open questions
- Are paper numbers inside the 5-seed spread? -> YES for final metrics (S0, final_All, M_d, M_f).
  Paper above all 5 seeds at S1_All, S1_New, S2_Old; not resolvable, paper reports 1 run (see agg_c100_base.txt)
- Are re-extracted features identical to the archived ones? -> NO: max|diff| 2.8e-5; phase 2 numbers
  still identical for seeds 0-2 (see c100_extract, c100_base_s0-2)
- Does PATCH 5 work? -> YES: 888,676 params, 2 pooler keys (see c100_extract)
- Are old-class params identical for ALL old classes, not only class 0?
- Is learned Sigma close to I, or close to the real class covariance?
- Why does S-pass early stop fall across stages (all 5 seeds: 752 -> 597)? Guess: scaling factor
  sqrt(10/(offset+10)) shrinks (mngmm.py:256-263). Check by logging dets per step (underflow in 384-d?)
- Does one GMM cluster absorb the known samples in S1?
- Novelty count ~4000: how many false positives / false negatives?
- How does S0_All grow with SVI steps? (smoke: 84.86 at 5 steps, same in old and new runs)
- Is New acc lower when cluster_max_min is high?
- Why is S4_New stable across seeds (std 0.47) while other stages are not (std 5-6)?
- Does the M_f proxy (S0 test set) match the paper definition (D_l)? Hint: S5_S0test mean 84.74 = 91.68 - 6.94
- Does code reproduce Table 5 "w/o fine-tuning" S0 = 88.54? Needs a phase-1 variant wrapper
  (no --finetuned, TorchHub CLS token, folder dino-vitb16), tag c100_extract_noft
- Does adding the contrastive term (paper Eq. 3) change S0 / M_d?
- How well does a frozen DINO backbone separate novel defect classes on NEU-DET / GC10-DET?

---

## 20260917_015448__c100_extract
why:    clean rerun, phase 1 (FT 32x4 = 128, extract batch 128, PATCH 5)
result: (no metrics.json yet; copy from runs/INDEX.txt later)
note:   check_phase1 OK (16 PASS). ckpt 888,676 params, 2 pooler keys -> PATCH 5 works.
        fbd_cuda link failed -> butterfly factor 1 (A5, as expected). CE 0.0218 at epoch 9, 1.21 s/it.
        Duration 168.3 min (old run 3: ~195 min). VRAM during FT: 6760 MiB (no hang).
        vs archive: labels identical; features NOT bit-identical, max|diff| 2.8e-5, mean|diff| 1.1e-6
        (train), 2.1e-5 / 1.1e-6 (test) -> GPU non-determinism. Saturated |x|>0.9999: 0.019% both.

## 20260917_131731__c100_smoke
why:    pipeline test after clean rerun, n_epochs 5
result: S0_All 84.86 | S1_New 31.40 | final_All 61.96 | M_d 40.40 | M_f 0.66
note:   OK. jax on GPU, 6 stages, 1.4 min, no warnings. S0_All 84.86 = old smoke (5 steps)
        -> 1e-5 feature diff does not change S0 predictions. S1+ numbers meaningless at 5 steps.

## 20260917_132425__c100_base_s0
why:    clean rerun baseline, seed 0
result: S0_All 91.84 | S1_New 67.90 | final_All 80.37 | M_d 76.92 | M_f 6.54
note:   same numbers as old seed 0 (exp.py test run, 2026-09-15)

## 20260917_133651__c100_base_s1
why:    clean rerun baseline, seed 1
result: S0_All 92.00 | S1_New 80.70 | final_All 81.75 | M_d 79.84 | M_f 6.80
note:   -

## 20260917_134924__c100_base_s2
why:    clean rerun baseline, seed 2
result: S0_All 91.96 | S1_New 76.40 | final_All 78.47 | M_d 74.84 | M_f 7.88
note:   -

## 20260917_140227__c100_base_s3
why:    clean rerun baseline, seed 3
result: S0_All 91.84 | S1_New 76.50 | final_All 80.96 | M_d 78.62 | M_f 7.04
note:   -

## 20260917_141513__c100_base_s4
why:    clean rerun baseline, seed 4; 5-seed summary here
result: S0_All 91.86 | S1_New 66.90 | final_All 77.77 | M_d 72.34 | M_f 7.52
note:   5-seed summary (runs/agg_c100_base.txt), no agg warnings.
        S0 91.90+-0.07, final 79.86+-1.68, M_d 76.51+-3.00, M_f(proxy) 7.16+-0.54. S0 gate PASS (0.22).
        Final metrics: paper inside mean+-2std. Paper above all 5 seeds at S1_All, S1_New (83.10 vs
        max 80.70) and S2_Old. Not resolvable: paper reports 1 run, no logs.
        Seeds 0-2 reproduce old 3-seed means exactly (80.20/77.20/7.07) despite 1e-5 feature diff.
        Variance comes from S1+ (S1_New 66.9-80.7); S0 std only 0.07.
        Scope: phase-2 variance on ONE feature extraction; phase-1 variance not measured.
        Numeric changes vs upstream: PATCH 2 (32x4, BOFT dropout masks differ) + butterfly factor 1.
        Verified: git diff upstream-baseline touches only dino-cifar100.py (+1, PATCH 1),
        sl_finetuned_model.py (+30/-4, PATCH 2+5) and .gitattributes (EOL only); phase-2 code unchanged.

## 20260918_120719__c100_consmoke
why:    smoke test for Eq.3 implementation (NT-Xent + 2-view aug, fold F5, 1 epoch)
result: (no metrics.json yet; copy from runs/INDEX.txt later)
note:   OK, 16 PASS, ckpt 888,676 (no new module, Q5 = no projection head).
        CE 1.2269, CON 0.4779 (log(31) = 3.43 = chance) -> contrastive learns in epoch 0.
        16x8 = effective 128, 800 it/epoch. 26.5 min for 1 epoch + extraction (~21 min fine-tune)
        -> re-estimate: 1 validation run (3 epochs) ~68 min, step 2 total ~22h, not 16h.
        Features from this run are NOT for phase 2 (1 epoch, 40 classes only).

## 20260918_221837__c100_km_smoke
why:
result: S0_All 84.86 | S1_New 39.90 | final_All 63.68 | M_d 44.64 | M_f 0.88
note:

## 20260919_000549__c100_km_s0
why:
result: S0_All 91.84 | S1_New 80.20 | final_All 81.86 | M_d 80.14 | M_f 6.64
note:

## 20260919_001802__c100_km_s1
why:
result: S0_All 92.00 | S1_New 80.60 | final_All 81.60 | M_d 79.66 | M_f 6.74
note:

## 20260919_003013__c100_km_s2
why:
result: S0_All 91.96 | S1_New 79.70 | final_All 81.80 | M_d 79.72 | M_f 6.46
note:

## 20260919_004223__c100_km_s3
why:
result: S0_All 91.84 | S1_New 80.40 | final_All 81.78 | M_d 80.12 | M_f 6.74
note:

## 20260919_005443__c100_km_s4
why:
result: S0_All 91.86 | S1_New 80.50 | final_All 81.78 | M_d 79.76 | M_f 6.46
note:   5-seed k-means summary (runs/agg_c100_km.txt), no agg warnings. vs GMM baseline:
        S0_All identical 91.90+-0.07 (control: k-means must not touch S0) -> OK.
        S1_New 73.68+-6.00 -> 80.28+-0.36 (std / 17). final_All 79.86 -> 81.76. M_d 76.51 -> 79.88.
        M_f 7.16 -> 6.61. Beats paper on final_All (81.23), M_d (79.14), M_f (6.94).
        Only gap left: S1_New 80.28 vs paper 83.10 (7.8 sigma); S2-S4 New now ABOVE paper.
        Confirms D18: GMM(full) 739k params / 4000 samples -> degenerate solutions, ELBO
        anti-correlates with cluster accuracy (r = -0.999 on c100_consmoke/F5).
