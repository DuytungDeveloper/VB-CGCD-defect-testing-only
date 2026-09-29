# LOG

One entry per run, appended by exp run. Fill in "why" and "note".
Before the clean rerun: archive/pre-rerun-20260915/ (not in git) and git tag pre-rerun-20260915

## Open questions
- Are paper numbers inside the 5-seed spread? -> YES for final metrics (S0, final_All, M_d, M_f).
  Paper above all 5 seeds at S1_All, S1_New, S2_Old; not resolvable, paper reports 1 run (see agg_c100_base.txt)
- Are re-extracted features identical to the archived ones? -> NO: max|diff| 2.8e-5; phase 2 numbers
  still identical for seeds 0-2 (see c100_extract, c100_base_s0-2)
- Does PATCH 5 work? -> YES: 888,676 params, 2 pooler keys (see c100_extract)
- Are old-class params identical for ALL old classes, not only class 0? -> YES for probed classes
  0,1,49 (S0-S5) and 50,59 (S1-S5), 4-decimal stats (probe.py ckpt, c100_base_s4). Not all 100 checked.
- Is learned Sigma close to I, or close to the real class covariance? -> close to I: trace/d ~0.98,
  ||S-I||_F ~2 (||I||_F = 19.6), eig [0.87, 1.79]. Not yet compared with empirical class covariance.
- Does clustering choice or restart count drive the results? -> ANSWERED by the 2x2 (see
  c100_gmm10_s4 and D18): restarts drive accuracy, algorithm drives stability.
- Why does S-pass early stop fall across stages (all 5 seeds: 752 -> 597)? No det underflow
  (logdet ~ -10). Old mean det < 1, so the rule is det_new < det_old / sf, and sf =
  sqrt(10/(offset+10)) shrinks -> threshold rises -> earlier stop. Consistent; per-step dets not logged.
- Does one GMM cluster absorb the known samples in S1?
- Novelty count ~4000: how many false positives / false negatives?
- How does S0_All grow with SVI steps? (smoke: 84.86 at 5 steps, same in old and new runs)
- Is New acc lower when cluster_max_min is high?
- Why is S4_New stable across seeds while other stages are not? With k-means std 0.43; with
  GMM n_init=10 S4_New 87.02 (4 of 5 seeds 87.7-88.7) but the lowest S5_New: cross-session
  trade-off? r(S4_New, S5_New) = -0.55, n = 5, not significant.
- Does the M_f proxy (S0 test set) match the paper definition (D_l)? Hint: S5_S0test mean 84.74 = 91.68 - 6.94
- Does code reproduce Table 5 "w/o fine-tuning" S0 = 88.54? Needs a phase-1 variant wrapper
  (no --finetuned, TorchHub CLS token, folder dino-vitb16), tag c100_extract_noft
- Does adding the contrastive term (paper Eq. 3) change S0 / M_d? Grid at tau=0.2 says no
  (monotone decrease with lambda, fold F5). Grid at tau=0.07 and tau=0.5 still missing;
  final answer needs a full 10-epoch 500-img/class run (c100_extract_con) + 5 phase-2 seeds.
- Phase-1 noise floor is still unmeasured: torch.manual_seed(42) is hardcoded, so two runs of
  the same config are identical (f5_l0_t02 == f5_l0_t02_rep). Would need a seed sweep in phase 1.
- Eq. 3 is evaluated with only 30 negatives per anchor (8 GB VRAM, batch 16 x 2 views).
  A negative result cannot separate "Eq. 3 does not help" from "too few negatives".
- How well does a frozen DINO backbone separate novel defect classes on NEU-DET / GC10-DET?

---

## 20260917_015448__c100_extract
why:    clean rerun, phase 1 (FT 32x4 = 128, extract batch 128, PATCH 5)
result: (no metrics.json)
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
        NOTE: this baseline is GMM with n_init=1; see the 2x2 in c100_gmm10_s4 and D18.

## 20260918_120719__c100_consmoke
why:    smoke test for Eq.3 implementation (NT-Xent + 2-view aug, fold F5, 1 epoch)
result: (no metrics.json)
note:   OK, 16 PASS, ckpt 888,676 (no new module, Q5 = no projection head).
        CE 1.2269, CON 0.4779 (log(31) = 3.43 = chance) -> contrastive learns in epoch 0.
        16x8 = effective 128, 800 it/epoch. 26.5 min for 1 epoch + extraction (~21 min fine-tune)
        -> 1 validation run (3 epochs) ~68 min.
        Features from this run are NOT for phase 2 (1 epoch, 40 classes only).
        Later used as the lambda=0.5 point of the 1-epoch probe: km novel acc 94.78 (see c100_con0_probe).

## 20260918_221837__c100_km_smoke
why:    pipeline test for KMeansCluster, n_epochs 5
result: S0_All 84.86 | S1_New 39.90 | final_All 63.68 | M_d 44.64 | M_f 0.88
note:   S0_All 84.86 = GMM smoke exactly -> k-means does not touch S0, as designed.
        dirty from uncommitted ft_data.py / eval_fold.py (phase-1 tools, not read by phase 2).

## 20260919_000549__c100_km_s0
why:    k-means clustering (n_init=10), seed 0
result: S0_All 91.84 | S1_New 80.20 | final_All 81.86 | M_d 80.14 | M_f 6.64
note:   -

## 20260919_001802__c100_km_s1
why:    k-means clustering (n_init=10), seed 1
result: S0_All 92.00 | S1_New 80.60 | final_All 81.60 | M_d 79.66 | M_f 6.74
note:   -

## 20260919_003013__c100_km_s2
why:    k-means clustering (n_init=10), seed 2
result: S0_All 91.96 | S1_New 79.70 | final_All 81.80 | M_d 79.72 | M_f 6.46
note:   -

## 20260919_004223__c100_km_s3
why:    k-means clustering (n_init=10), seed 3
result: S0_All 91.84 | S1_New 80.40 | final_All 81.78 | M_d 80.12 | M_f 6.74
note:   -

## 20260919_005443__c100_km_s4
why:    k-means clustering (n_init=10), seed 4; 5-seed summary here
result: S0_All 91.86 | S1_New 80.50 | final_All 81.78 | M_d 79.76 | M_f 6.46
note:   5-seed k-means summary (runs/agg_c100_km.txt), no agg warnings. vs GMM baseline:
        S0_All identical 91.90+-0.07 (control: k-means must not touch S0) -> OK.
        S1_New 73.68+-6.00 -> 80.28+-0.36 (std / 17). final_All 79.86 -> 81.76. M_d 76.51 -> 79.88.
        M_f 7.16 -> 6.61. Beats paper on final_All (81.23), M_d (79.14), M_f (6.94).
        Only gap left: S1_New 80.28 vs paper 83.10; S2-S4 New now ABOVE paper.
        CORRECTION (2026-09-19): the first version of this note credited the gain to GMM(full)
        being degenerate in 384 dims. The 2x2 (c100_km1, c100_gmm10) disproved that: most of the
        gain comes from n_init=10, not from the algorithm. See D18 and c100_gmm10_s4.

## 20260919_012906__c100_km1_s0
why:    2x2 cell: k-means with n_init=1
result: S0_All 91.84 | S1_New 67.90 | final_All 80.37 | M_d 76.92 | M_f 6.54
note:   -

## 20260919_014105__c100_km1_s1
why:    2x2 cell: k-means with n_init=1
result: S0_All 92.00 | S1_New 80.70 | final_All 81.75 | M_d 79.84 | M_f 6.80
note:   -

## 20260919_015303__c100_km1_s2
why:    2x2 cell: k-means with n_init=1
result: S0_All 91.96 | S1_New 76.40 | final_All 78.47 | M_d 74.84 | M_f 7.88
note:   -

## 20260919_020524__c100_km1_s3
why:    2x2 cell: k-means with n_init=1
result: S0_All 91.84 | S1_New 76.50 | final_All 80.96 | M_d 78.62 | M_f 7.04
note:   -

## 20260919_021729__c100_km1_s4
why:    2x2 cell: k-means with n_init=1, seed 4
result: S0_All 91.86 | S1_New 66.90 | final_All 77.77 | M_d 72.34 | M_f 7.52
note:   c100_km1 reproduces c100_base EXACTLY on all 43 metrics -> with one restart, GMM and
        k-means give the same partition. Direct check on real S1 data (5250 imgs, 384 dims):
        the two partitions differ by 1-2 of 5250 samples.

## 20260919_023837__c100_regress_s0
why:    regression check after adding --gmm_n_init (default 1 must equal upstream)
result: S0_All 91.84 | S1_New 67.90 | final_All 80.37 | M_d 76.92 | M_f 6.54
note:   first of two identical regression runs; see 20260919_095904.

## 20260919_025223__c100_gmm10_s0
why:    2x2 cell: GMM with n_init=10
result: S0_All 91.84 | S1_New 74.30 | final_All 80.50 | M_d 77.96 | M_f 7.10
note:   -

## 20260919_030759__c100_gmm10_s1
why:    2x2 cell: GMM with n_init=10
result: S0_All 92.00 | S1_New 85.20 | final_All 81.26 | M_d 78.94 | M_f 6.96
note:   -

## 20260919_032332__c100_gmm10_s2
why:    2x2 cell: GMM with n_init=10
result: S0_All 91.96 | S1_New 75.50 | final_All 81.79 | M_d 79.62 | M_f 6.62
note:   -

## 20260919_033943__c100_gmm10_s3
why:    2x2 cell: GMM with n_init=10
result: S0_All 91.84 | S1_New 76.50 | final_All 80.77 | M_d 78.30 | M_f 7.08
note:   -

## 20260919_035659__c100_gmm10_s4
why:    2x2 cell: GMM with n_init=10, seed 4; 2x2 summary here
result: S0_All 91.86 | S1_New 75.10 | final_All 82.41 | M_d 80.74 | M_f 6.46
note:   2x2 design, 5 seeds per cell, same features (c100_extract), only clustering changed.
        final_All / M_d:
          GMM      n_init=1  79.86+-1.68 / 76.51+-3.00    n_init=10  81.35+-0.77 / 79.11+-1.11
          k-means  n_init=1  79.86+-1.68 / 76.51+-3.00    n_init=10  81.76+-0.10 / 79.88+-0.23
        restarts:  +1.49 (GMM) / +1.90 (k-means) on final_All; +2.60 / +3.37 on M_d.
        algorithm: +0.00 at n_init=1, +0.41 at n_init=10 -> about 4x smaller, no real interaction.
        stability: algorithm dominates, std of final_All 0.10 (km10) vs 0.77 (gmm10) vs 1.68 (n_init=1).
        control:   S0_All 91.90+-0.07 in all four cells (S0 uses no clustering).
        Best config = k-means with n_init=10, i.e. what Appendix C.3 describes plus restarts. See D18.
        Odd: S4_New 87.02 here (4 of 5 seeds 87.7-88.7) but the lowest S5_New 71.82.

## 20260919_095904__c100_regress_s0
why:    regression check after adding --gmm_n_init (default 1 must equal upstream)
result: S0_All 91.84 | S1_New 67.90 | final_All 80.37 | M_d 76.92 | M_f 6.54
note:   identical to c100_base_s0 on every metric -> the new flag does not change the default path.

## 20260919_103943__c100_con0_probe
why:    lambda=0 control for the Eq.3 probe (fold F5, 1 epoch), to check whether the validation
        metric can separate configurations at all
result: (no metrics.json)
note:   km novel acc 95.23+-0.010 vs 94.78+-0.008 for lambda=0.5 (c100_consmoke) -> Eq.3 worse by 0.44.
        known_acc 89.22 vs 89.63 (opposite direction, so at least one of the two is noise).
        Ceiling: full baseline features (c100_extract, 10 epochs, 50 classes) on F5 give
        km 97.93 and known_acc 99.91 -> only ~2.7 points of head-room; the fold task is easy.
        13.0 min for 1 epoch with 1 view (vs ~21 min with 2 views).

## 20260919_124635__c100_con0_probe
why:    repeat of the lambda=0 probe
result: INTERRUPTED (SIGINT)
note:   stopped after 0.1 min, no features written. Ignore; kept for the record.

## 20260919_175218__f5_l0_t02
why:    grid stage 2a: lambda=0, tau=0.2, fold F5, 3 epochs (baseline point of the grid)
result: (no metrics.json)
note:   CE 0.1796 at epoch 2, no CON (lambda=0 means no contrastive and no augmentation).

## 20260919_185711__f5_l0_t02_rep
why:    repeat of the lambda=0 grid point, intended as the noise floor
result: (no metrics.json)
note:   CE 0.1796, identical to f5_l0_t02 to 4 decimals, and km novel acc identical too.
        Phase 1 is seed-deterministic (torch.manual_seed(42) is hardcoded), so this measures
        repeatability, NOT the noise floor. A real noise floor needs a seed sweep in phase 1.

## 20260919_194732__f5_l025_t02
why:    grid stage 2a: lambda=0.25, tau=0.2
result: (no metrics.json)
note:   CE 0.2611, CON 0.4081 at epoch 2.

## 20260919_204744__f5_l05_t02
why:    grid stage 2a: lambda=0.5, tau=0.2
result: (no metrics.json)
note:   CE 0.2638, CON 0.3718 at epoch 2.

## 20260919_214648__f5_l075_t02
why:    grid stage 2a: lambda=0.75, tau=0.2; grid summary here
result: (no metrics.json)
note:   CE 0.2868, CON 0.3423 at epoch 2. CON falls monotonically with lambda (0.408 / 0.372 /
        0.342), so the contrastive term is being optimised as intended.
        Grid at tau=0.2 (scripts/pick_lambda.py, k-means n_init=10, 5 clustering seeds, fold F5):
          lambda 0     95.48 +-0.066     (and the repeat run: 95.48, identical)
          lambda 0.25  95.42 +-0.031     -0.05
          lambda 0.5   95.16 +-0.012     -0.32
          lambda 0.75  94.81 +-0.031     -0.67
        Monotone decrease; same direction as the 1-epoch probe (-0.44 at lambda 0.5).
        Scope: one fold, one phase-1 seed, tau=0.2 only, 30 negatives per anchor, and the fold
        task is near its ceiling (95.5 of ~97.9). Not yet a conclusion about Eq. 3.
        All 5 grid runs: commit 3f2320f, dirty only scripts/eval_fold.py (evaluation tool).
## 20260919_234953__f5_l025_t007
why:    grid stage 2a: lambda=0.25, tau=0.07, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   -

## 20260920_005051__f5_l05_t007
why:    grid stage 2a: lambda=0.5, tau=0.07, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   -

## 20260920_015334__f5_l075_t007
why:    grid stage 2a: lambda=0.75, tau=0.07, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   -

## 20260920_025511__f5_l025_t05
why:    grid stage 2a: lambda=0.25, tau=0.5, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   -

## 20260920_035516__f5_l05_t05
why:    grid stage 2a: lambda=0.5, tau=0.5, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   -

## 20260920_045513__f5_l075_t05
why:    grid stage 2a: lambda=0.75, tau=0.5, fold F5, 3 epochs, 400 imgs/class
result: (no metrics.json)
note:   validation grid complete, 3 lambda x 3 tau, fold F5, 3 epochs
        (scripts/pick_lambda.py, k-means n_init=10, 5 clustering seeds). Cluster acc on the 4000
        fake-novel images; lambda=0 baseline 95.48:
          lambda \ tau     0.07     0.2      0.5
          0.25            95.31    95.42    95.38
          0.5             95.36    95.16    95.30
          0.75            95.25    94.81    94.64
        No cell beats lambda=0. Effect of lambda (mean -0.10 / -0.20 / -0.58) is 2-3x the effect
        of tau (-0.17 / -0.35 / -0.37). Interaction visible at lambda 0.75 (tau 0.07 -0.22 vs
        tau 0.5 -0.84). WARNING: this criterion later turned out NOT to predict downstream
        results (it said lambda 0.25 was -0.05; at full scale it was +0.19 then -0.13 at n=3).
        The fold task is near its ceiling: 95.5 of ~97.9.

## 20260920_074552__c100_extract_con
why:    phase 1 with Eq. 3: lambda=0.25, tau=0.2, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   337.5 min. First full-scale Eq. 3 run: 500 imgs/class, 50 classes, 10 epochs.
        Batch forced to 16x8 because 32 x 2 views = 7.68 GB > 6.93 GB available (OOM).
        Effective batch still 128. 30 negatives per anchor.

## 20260920_132656__c100_con_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: c100_extract_con)
result: S0_All 92.10 | S1_New 84.30 | final_All 82.31 | M_d 80.60 | M_f 6.58
note:   -

## 20260920_133925__c100_con_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: c100_extract_con)
result: S0_All 91.90 | S1_New 80.00 | final_All 81.94 | M_d 80.08 | M_f 6.58
note:   -

## 20260920_135143__c100_con_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: c100_extract_con)
result: S0_All 91.88 | S1_New 80.30 | final_All 81.72 | M_d 79.84 | M_f 6.74
note:   -

## 20260920_140401__c100_con_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: c100_extract_con)
result: S0_All 91.96 | S1_New 80.90 | final_All 81.82 | M_d 79.68 | M_f 6.52
note:   -

## 20260920_141618__c100_con_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: c100_extract_con)
result: S0_All 91.74 | S1_New 80.40 | final_All 81.81 | M_d 79.96 | M_f 6.52
note:   5-seed summary (agg_c100_con.txt). S0_All 91.92+-0.13, S1_New 81.18+-1.77,
        final_All 81.92+-0.23, M_d 80.03+-0.35, M_f 6.59+-0.09. S1_All 88.40 = paper exactly.
        vs c100_km (lambda 0, batch 32x4): +0.16 final_All. NOT interpretable on its own:
        two variables changed (contrastive AND batch) -> see the control c100_ft16.

## 20260920_144616__c100_extract_ft16
why:    phase-1 CONTROL for Eq. 3: lambda=0, batch 16x8 (isolates batch size from contrastive)
result: (no metrics.json)
note:   -

## 20260920_195205__c100_ft16_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: c100_extract_ft16)
result: S0_All 91.80 | S1_New 80.70 | final_All 81.75 | M_d 80.02 | M_f 6.72
note:   -

## 20260920_200418__c100_ft16_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: c100_extract_ft16)
result: S0_All 91.72 | S1_New 80.40 | final_All 81.71 | M_d 79.90 | M_f 6.64
note:   -

## 20260920_201636__c100_ft16_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: c100_extract_ft16)
result: S0_All 91.72 | S1_New 80.00 | final_All 81.52 | M_d 79.56 | M_f 6.68
note:   -

## 20260920_202855__c100_ft16_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: c100_extract_ft16)
result: S0_All 91.96 | S1_New 81.20 | final_All 81.69 | M_d 79.98 | M_f 7.00
note:   -

## 20260920_204120__c100_ft16_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: c100_extract_ft16)
result: S0_All 91.80 | S1_New 83.40 | final_All 81.97 | M_d 80.46 | M_f 6.72
note:   5-seed summary (agg_c100_ft16.txt). final_All 81.73+-0.16, M_d 79.98+-0.32.
        vs c100_km (same lambda 0, batch 32x4): 81.76 vs 81.73 -> batch size alone changes nothing.
        So the clean Eq. 3 comparison is this run vs c100_con: +0.19 final_All, +0.05 M_d.

## 20260920_224309__c100_seedcheck
why:    regression check after adding FT_SEED: default 42 must reproduce upstream
result: (no metrics.json)
note:   FT_SEED default 42 reproduces c100_con0_probe: mean CE 1.1756 identical,
        features max|diff| 1.15e-5 (GPU noise only). The flag does not change the default path.

## 20260920_235027__c100_extract_ft16_sd43
why:    phase-1 noise measurement: lambda=0, batch 16x8, FT_SEED 43
result: (no metrics.json)
note:   -

## 20260921_041554__c100_extract_ft16_sd44
why:    phase-1 noise measurement: lambda=0, batch 16x8, FT_SEED 44
result: (no metrics.json)
note:   -

## 20260921_123050__c100_ft16sd43_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: ft16 FT_SEED 43)
result: S0_All 91.90 | S1_New 79.90 | final_All 81.99 | M_d 80.08 | M_f 6.32
note:   -

## 20260921_124304__c100_ft16sd43_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: ft16 FT_SEED 43)
result: S0_All 91.78 | S1_New 80.00 | final_All 81.74 | M_d 79.56 | M_f 6.12
note:   -

## 20260921_125510__c100_ft16sd43_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: ft16 FT_SEED 43)
result: S0_All 91.74 | S1_New 78.30 | final_All 81.55 | M_d 79.44 | M_f 6.44
note:   -

## 20260921_130729__c100_ft16sd43_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: ft16 FT_SEED 43)
result: S0_All 91.82 | S1_New 79.90 | final_All 81.58 | M_d 79.44 | M_f 6.30
note:   -

## 20260921_131953__c100_ft16sd43_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: ft16 FT_SEED 43)
result: S0_All 91.78 | S1_New 79.50 | final_All 81.69 | M_d 79.84 | M_f 6.44
note:   -

## 20260921_133213__c100_ft16sd44_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: ft16 FT_SEED 44)
result: S0_All 91.78 | S1_New 79.80 | final_All 81.78 | M_d 79.94 | M_f 6.54
note:   -

## 20260921_134434__c100_ft16sd44_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: ft16 FT_SEED 44)
result: S0_All 91.86 | S1_New 79.30 | final_All 81.82 | M_d 80.08 | M_f 6.62
note:   -

## 20260921_135648__c100_ft16sd44_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: ft16 FT_SEED 44)
result: S0_All 91.88 | S1_New 78.70 | final_All 81.86 | M_d 79.90 | M_f 6.42
note:   -

## 20260921_140936__c100_ft16sd44_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: ft16 FT_SEED 44)
result: S0_All 91.98 | S1_New 79.70 | final_All 81.97 | M_d 80.42 | M_f 6.86
note:   -

## 20260921_142159__c100_ft16sd44_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: ft16 FT_SEED 44)
result: S0_All 91.74 | S1_New 79.20 | final_All 82.04 | M_d 80.30 | M_f 6.48
note:   -

## 20260921_161513__c100_extract_ft16_sd45
why:    phase-1 noise measurement: lambda=0, batch 16x8, FT_SEED 45
result: (no metrics.json)
note:   -

## 20260921_204122__c100_ft16sd45_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: ft16 FT_SEED 45)
result: S0_All 91.82 | S1_New 79.50 | final_All 81.99 | M_d 80.16 | M_f 6.48
note:   -

## 20260921_205352__c100_ft16sd45_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: ft16 FT_SEED 45)
result: S0_All 91.84 | S1_New 79.70 | final_All 81.99 | M_d 80.52 | M_f 6.62
note:   -

## 20260921_210622__c100_ft16sd45_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: ft16 FT_SEED 45)
result: S0_All 91.72 | S1_New 79.20 | final_All 81.78 | M_d 80.12 | M_f 6.64
note:   -

## 20260921_211849__c100_ft16sd45_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: ft16 FT_SEED 45)
result: S0_All 91.62 | S1_New 79.70 | final_All 81.92 | M_d 80.18 | M_f 6.48
note:   -

## 20260921_213122__c100_ft16sd45_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: ft16 FT_SEED 45)
result: S0_All 91.70 | S1_New 79.70 | final_All 81.98 | M_d 80.12 | M_f 6.38
note:   -

## 20260921_214348__c100_extract_con_sd43
why:    extra phase-1 seed for cell (lambda 0.25, tau 0.2): FT_SEED 43
result: (no metrics.json)
note:   -

## 20260922_031414__c100_consd43_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: con lambda 0.25 tau 0.2, FT_SEED 43)
result: S0_All 91.86 | S1_New 79.90 | final_All 81.96 | M_d 79.96 | M_f 6.40
note:   -

## 20260922_032613__c100_consd43_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: con lambda 0.25 tau 0.2, FT_SEED 43)
result: S0_All 91.84 | S1_New 79.70 | final_All 81.82 | M_d 79.94 | M_f 6.56
note:   -

## 20260922_033820__c100_consd43_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: con lambda 0.25 tau 0.2, FT_SEED 43)
result: S0_All 91.88 | S1_New 79.80 | final_All 81.46 | M_d 79.42 | M_f 6.82
note:   -

## 20260922_035025__c100_consd43_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: con lambda 0.25 tau 0.2, FT_SEED 43)
result: S0_All 91.92 | S1_New 80.20 | final_All 81.67 | M_d 79.74 | M_f 6.74
note:   -

## 20260922_040236__c100_consd43_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: con lambda 0.25 tau 0.2, FT_SEED 43)
result: S0_All 91.96 | S1_New 80.00 | final_All 81.72 | M_d 79.84 | M_f 6.70
note:   -

## 20260922_041443__c100_extract_ft16_sd46
why:    phase-1 noise measurement: lambda=0, batch 16x8, FT_SEED 46
result: (no metrics.json)
note:   -

## 20260922_083829__c100_ft16sd46_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: ft16 FT_SEED 46)
result: S0_All 92.12 | S1_New 80.10 | final_All 81.95 | M_d 80.04 | M_f 6.76
note:   -

## 20260922_085042__c100_ft16sd46_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: ft16 FT_SEED 46)
result: S0_All 91.98 | S1_New 79.70 | final_All 82.15 | M_d 80.28 | M_f 6.64
note:   -

## 20260922_090254__c100_ft16sd46_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: ft16 FT_SEED 46)
result: S0_All 91.94 | S1_New 79.30 | final_All 81.71 | M_d 79.80 | M_f 6.86
note:   -

## 20260922_091512__c100_ft16sd46_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: ft16 FT_SEED 46)
result: S0_All 91.98 | S1_New 80.50 | final_All 81.87 | M_d 80.32 | M_f 7.06
note:   -

## 20260922_092734__c100_ft16sd46_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: ft16 FT_SEED 46)
result: S0_All 91.90 | S1_New 80.30 | final_All 82.00 | M_d 80.52 | M_f 6.92
note:   PHASE-1 NOISE FLOOR, the number every phase-1 comparison needs.
        5 phase-1 seeds at lambda=0 (42..46), each = mean of 5 phase-2 seeds:
          final_All  81.73 / 81.71 / 81.89 / 81.93 / 81.94  -> 81.84 +- 0.11
          M_d        79.98 / 79.67 / 80.13 / 80.22 / 80.19  -> 80.04 +- 0.23
          S0_All                                            -> 91.84 +- 0.09
          S1_New     81.14 / 79.52 / 79.34 / 79.56 / 79.98  -> 79.91 +- 0.73
        Consequences: (1) the +0.19 of c100_con is 1.2 sigma, not significant; (2) beating the
        paper IS significant: final_All +0.55 (5 sigma), M_d +0.79 (3.4 sigma); (3) the S1_New
        gap to the paper (2.8 points) is only ~2.8 sigma, not the 7.8 claimed earlier from
        phase-2 spread. n = number of PHASE-1 runs; the 5 phase-2 seeds are not independent units.

## 20260922_093949__c100_extract_con_sd44
why:    extra phase-1 seed for cell (lambda 0.25, tau 0.2): FT_SEED 44
result: (no metrics.json)
note:   -

## 20260922_150955__c100_consd44_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: con lambda 0.25 tau 0.2, FT_SEED 44)
result: S0_All 91.66 | S1_New 79.10 | final_All 81.36 | M_d 79.14 | M_f 6.62
note:   -

## 20260922_152216__c100_consd44_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: con lambda 0.25 tau 0.2, FT_SEED 44)
result: S0_All 91.60 | S1_New 79.10 | final_All 81.35 | M_d 79.32 | M_f 6.80
note:   -

## 20260922_153418__c100_consd44_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: con lambda 0.25 tau 0.2, FT_SEED 44)
result: S0_All 91.52 | S1_New 78.90 | final_All 81.57 | M_d 79.54 | M_f 6.42
note:   -

## 20260922_154627__c100_consd44_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: con lambda 0.25 tau 0.2, FT_SEED 44)
result: S0_All 91.88 | S1_New 79.90 | final_All 81.63 | M_d 79.80 | M_f 6.86
note:   -

## 20260922_155840__c100_consd44_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: con lambda 0.25 tau 0.2, FT_SEED 44)
result: S0_All 91.68 | S1_New 79.70 | final_All 81.51 | M_d 79.50 | M_f 6.60
note:   cell (0.25, 0.2) now has 3 phase-1 seeds: final_All 81.92 / 81.73 / 81.48
        -> 81.71 +- 0.22, i.e. BELOW the lambda=0 mean 81.84. The single-seed +0.19 did not hold.

## 20260922_192320__c100_extract_con_l01_t02_sd42
why:    grid cell at full scale: lambda=0.1, tau=0.2, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   -

## 20260923_011302__c100_con_l01_t02_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.1, tau 0.2, FT_SEED 42)
result: S0_All 92.18 | S1_New 80.50 | final_All 82.12 | M_d 80.00 | M_f 6.50
note:   -

## 20260923_012517__c100_con_l01_t02_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.1, tau 0.2, FT_SEED 42)
result: S0_All 92.20 | S1_New 80.80 | final_All 82.27 | M_d 80.20 | M_f 6.38
note:   -

## 20260923_013718__c100_con_l01_t02_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.1, tau 0.2, FT_SEED 42)
result: S0_All 92.08 | S1_New 80.10 | final_All 82.24 | M_d 80.18 | M_f 6.34
note:   -

## 20260923_014926__c100_con_l01_t02_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.1, tau 0.2, FT_SEED 42)
result: S0_All 92.28 | S1_New 81.10 | final_All 82.14 | M_d 80.10 | M_f 6.58
note:   -

## 20260923_020134__c100_con_l01_t02_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.1, tau 0.2, FT_SEED 42)
result: S0_All 92.00 | S1_New 80.10 | final_All 82.09 | M_d 79.84 | M_f 6.28
note:   -

## 20260923_095021__c100_extract_con_l05_t02_sd42
why:    grid cell at full scale: lambda=0.5, tau=0.2, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   -

## 20260923_151415__c100_con_l05_t02_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.2, FT_SEED 42)
result: S0_All 91.20 | S1_New 79.70 | final_All 81.15 | M_d 79.34 | M_f 6.68
note:   -

## 20260923_152605__c100_con_l05_t02_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.2, FT_SEED 42)
result: S0_All 91.30 | S1_New 79.10 | final_All 81.09 | M_d 79.26 | M_f 6.70
note:   -

## 20260923_153801__c100_con_l05_t02_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.2, FT_SEED 42)
result: S0_All 91.30 | S1_New 79.10 | final_All 81.16 | M_d 79.24 | M_f 6.62
note:   -

## 20260923_154958__c100_con_l05_t02_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.2, FT_SEED 42)
result: S0_All 91.44 | S1_New 79.60 | final_All 81.31 | M_d 79.50 | M_f 6.74
note:   -

## 20260923_160157__c100_con_l05_t02_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.2, FT_SEED 42)
result: S0_All 91.22 | S1_New 79.50 | final_All 81.02 | M_d 79.20 | M_f 6.74
note:   -

## 20260923_181654__c100_extract_con_l01_t02_sd43
why:    grid cell at full scale: lambda=0.1, tau=0.2, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 43
result: (no metrics.json)
note:   -

## 20260923_234257__c100_con_l01_t02_sd43_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.1, tau 0.2, FT_SEED 43)
result: S0_All 92.18 | S1_New 80.50 | final_All 82.06 | M_d 79.96 | M_f 6.34
note:   -

## 20260923_235449__c100_con_l01_t02_sd43_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.1, tau 0.2, FT_SEED 43)
result: S0_All 92.12 | S1_New 79.70 | final_All 82.31 | M_d 80.16 | M_f 6.10
note:   -

## 20260924_000649__c100_con_l01_t02_sd43_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.1, tau 0.2, FT_SEED 43)
result: S0_All 92.22 | S1_New 80.00 | final_All 82.23 | M_d 79.90 | M_f 6.16
note:   -

## 20260924_001857__c100_con_l01_t02_sd43_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.1, tau 0.2, FT_SEED 43)
result: S0_All 92.28 | S1_New 78.10 | final_All 81.76 | M_d 79.78 | M_f 7.02
note:   -

## 20260924_003103__c100_con_l01_t02_sd43_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.1, tau 0.2, FT_SEED 43)
result: S0_All 92.16 | S1_New 80.40 | final_All 82.29 | M_d 80.42 | M_f 6.46
note:   -

## 20260924_004303__c100_extract_con_l01_t02_sd44
why:    grid cell at full scale: lambda=0.1, tau=0.2, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 44
result: (no metrics.json)
note:   -

## 20260924_060930__c100_con_l01_t02_sd44_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.1, tau 0.2, FT_SEED 44)
result: S0_All 91.76 | S1_New 79.80 | final_All 81.68 | M_d 80.04 | M_f 6.90
note:   -

## 20260924_062138__c100_con_l01_t02_sd44_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.1, tau 0.2, FT_SEED 44)
result: S0_All 91.84 | S1_New 79.10 | final_All 81.36 | M_d 79.28 | M_f 6.80
note:   -

## 20260924_063347__c100_con_l01_t02_sd44_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.1, tau 0.2, FT_SEED 44)
result: S0_All 91.80 | S1_New 78.40 | final_All 81.39 | M_d 79.44 | M_f 6.84
note:   -

## 20260924_064552__c100_con_l01_t02_sd44_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.1, tau 0.2, FT_SEED 44)
result: S0_All 92.08 | S1_New 79.40 | final_All 81.55 | M_d 79.70 | M_f 7.02
note:   -

## 20260924_065803__c100_con_l01_t02_sd44_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.1, tau 0.2, FT_SEED 44)
result: S0_All 91.92 | S1_New 79.30 | final_All 81.67 | M_d 79.64 | M_f 6.76
note:   LAMBDA AXIS at tau=0.2 (500 imgs/class, 10 epochs, k-means n_init=10).
        n = phase-1 runs; each value = mean of 5 phase-2 seeds:
          lambda 0     n=5  final 81.84+-0.11  M_d 80.04+-0.23
          lambda 0.1   n=3  final 81.94+-0.36  M_d 79.91+-0.25
          lambda 0.25  n=3  final 81.71+-0.22  M_d 79.76+-0.29
          lambda 0.5   n=1  final 81.15        M_d 79.31
        Slope final_All -1.17 per unit lambda (p=0.027), M_d -1.35 (p=0.011), 12 phase-1 runs.
        LESSON: lambda 0.1 seed 42 alone gave 82.17 (+0.33, looked like a real gain); seed 44 gave
        81.53 and the n=3 mean fell to +0.10. Never conclude from n=1.
        Contrastive also seems to RAISE phase-1 variance: std 0.36 (l0.1) and 0.22 (l0.25) vs 0.11 (l0).
        dirty only scripts/queue_grid.sh (queue driver, not read by phase 1 or phase 2).

## 20260924_084033__c100_extract_con_l05_t007_sd42
why:    grid cell at full scale: lambda=0.5, tau=0.07, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   -

## 20260924_140731__c100_con_l05_t007_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.07, FT_SEED 42)
result: S0_All 92.04 | S1_New 80.70 | final_All 82.19 | M_d 80.48 | M_f 6.68
note:   -

## 20260924_141934__c100_con_l05_t007_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.07, FT_SEED 42)
result: S0_All 92.14 | S1_New 81.00 | final_All 82.20 | M_d 80.46 | M_f 6.74
note:   -

## 20260924_143151__c100_con_l05_t007_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.07, FT_SEED 42)
result: S0_All 92.04 | S1_New 80.80 | final_All 82.00 | M_d 80.24 | M_f 6.80
note:   -

## 20260924_144351__c100_con_l05_t007_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.07, FT_SEED 42)
result: S0_All 92.16 | S1_New 81.50 | final_All 82.23 | M_d 80.52 | M_f 6.78
note:   -

## 20260924_145601__c100_con_l05_t007_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.07, FT_SEED 42)
result: S0_All 92.12 | S1_New 80.70 | final_All 82.05 | M_d 80.22 | M_f 6.72
note:   -

## 20260924_150804__c100_extract_con_l05_t05_sd42
why:    grid cell at full scale: lambda=0.5, tau=0.5, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   -

## 20260924_203440__c100_con_l05_t05_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.5, FT_SEED 42)
result: S0_All 91.14 | S1_New 79.00 | final_All 80.58 | M_d 78.42 | M_f 6.54
note:   -

## 20260924_204659__c100_con_l05_t05_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.5, FT_SEED 42)
result: S0_All 91.12 | S1_New 78.60 | final_All 80.64 | M_d 78.48 | M_f 6.48
note:   -

## 20260924_205913__c100_con_l05_t05_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.5, FT_SEED 42)
result: S0_All 91.18 | S1_New 78.10 | final_All 80.27 | M_d 77.80 | M_f 6.72
note:   -

## 20260924_211155__c100_con_l05_t05_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.5, FT_SEED 42)
result: S0_All 91.28 | S1_New 78.90 | final_All 79.57 | M_d 76.28 | M_f 6.56
note:   -

## 20260924_212534__c100_con_l05_t05_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.5, FT_SEED 42)
result: S0_All 91.18 | S1_New 78.80 | final_All 80.24 | M_d 77.76 | M_f 6.58
note:   TAU AXIS at lambda=0.5, n=1 per cell, vs the lambda=0 baseline 81.84 / 91.84:
          tau 0.07   S0 92.10   final 82.13   M_d 80.38
          tau 0.2    S0 91.29   final 81.15   M_d 79.31
          tau 0.5    S0 91.18   final 80.26   M_d 77.75
        Range on the tau axis is 1.87 points of final_All, vs 0.79 on the lambda axis -> tau is the
        more important hyper-parameter, and it INTERACTS with lambda. The whole lambda sweep was run
        at tau=0.2, which happens to be a harmful setting; that is why the earlier conclusion
        ("Eq. 3 does not help") was premature.

## 20260924_214316__c100_extract_con_l05_t007_sd43
why:    grid cell at full scale: lambda=0.5, tau=0.07, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 43
result: (no metrics.json)
note:   -

## 20260925_033039__c100_con_l05_t007_sd43_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.07, FT_SEED 43)
result: S0_All 92.24 | S1_New 80.70 | final_All 82.14 | M_d 80.18 | M_f 6.50
note:   -

## 20260925_034236__c100_con_l05_t007_sd43_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.07, FT_SEED 43)
result: S0_All 92.08 | S1_New 80.20 | final_All 82.18 | M_d 80.16 | M_f 6.28
note:   -

## 20260925_035429__c100_con_l05_t007_sd43_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.07, FT_SEED 43)
result: S0_All 92.16 | S1_New 80.10 | final_All 82.17 | M_d 80.20 | M_f 6.38
note:   -

## 20260925_040627__c100_con_l05_t007_sd43_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.07, FT_SEED 43)
result: S0_All 92.28 | S1_New 80.50 | final_All 82.09 | M_d 80.18 | M_f 6.56
note:   -

## 20260925_041827__c100_con_l05_t007_sd43_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.07, FT_SEED 43)
result: S0_All 92.14 | S1_New 80.70 | final_All 82.22 | M_d 80.20 | M_f 6.16
note:   -

## 20260925_043020__c100_extract_con_l05_t007_sd44
why:    grid cell at full scale: lambda=0.5, tau=0.07, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 44
result: (no metrics.json)
note:   -

## 20260925_095506__c100_con_l05_t007_sd44_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.07, FT_SEED 44)
result: S0_All 92.06 | S1_New 79.90 | final_All 81.81 | M_d 80.06 | M_f 7.04
note:   -

## 20260925_100705__c100_con_l05_t007_sd44_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.07, FT_SEED 44)
result: S0_All 91.98 | S1_New 79.40 | final_All 81.79 | M_d 80.04 | M_f 7.00
note:   -

## 20260925_101906__c100_con_l05_t007_sd44_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.07, FT_SEED 44)
result: S0_All 92.12 | S1_New 79.40 | final_All 81.55 | M_d 79.56 | M_f 7.08
note:   -

## 20260925_103104__c100_con_l05_t007_sd44_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.07, FT_SEED 44)
result: S0_All 92.20 | S1_New 80.20 | final_All 81.57 | M_d 79.76 | M_f 7.22
note:   -

## 20260925_104308__c100_con_l05_t007_sd44_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.07, FT_SEED 44)
result: S0_All 92.08 | S1_New 79.60 | final_All 82.09 | M_d 80.26 | M_f 6.68
note:   CONCLUSION for Eq. 3 (cell lambda=0.5, tau=0.07, now n=3 phase-1 seeds),
        Welch t-test vs lambda=0 (n=5):
          S0_All     91.84+-0.09 -> 92.12+-0.05   +0.29   t=+5.73  p=0.001   SIGNIFICANT
          final_All  81.84+-0.11 -> 82.02+-0.22   +0.18   t=+1.30  p=0.298
          M_d        80.04+-0.23 -> 80.17+-0.22   +0.13   t=+0.79  p=0.473
          S1_New     79.91+-0.73 -> 80.36+-0.62   +0.45   t=+0.93  p=0.395
        So Eq. 3 with a SMALL tau measurably improves the backbone (S0, supervised session), and all
        three continual metrics move the same way but are not significant at this sample size:
        variance grows along the pipeline (sigma 0.09 at S0 -> 0.22 at final_All -> 0.62 at S1_New).
        Reaching p<0.05 on final_All would need ~14 phase-1 runs per arm (>100 h). Seed 44 gives
        final 81.76 (no gain) yet S0 92.09 like the others: the backbone improves every time, but
        whether that reaches the final metric varies by seed.
        Scope: tau swept at full scale only at lambda=0.5; 30 negatives per anchor; augmentation
        and NT-Xent form are our choices (the paper underspecifies Eq. 2).

## 20260925_111920__c100_extract_con_l05_t003_sd42
why:    tau-axis cell: lambda=0.5, tau=0.03, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   mean CON falls to 0.0008 by epoch 9 (chance = log(31) = 3.43), i.e. the contrastive task is
        SOLVED and L_con has almost no gradient left. CE 0.1293, same as tau=0.07 (0.126).

## 20260925_164557__c100_con_l05_t003_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.03, FT_SEED 42)
result: S0_All 92.04 | S1_New 80.80 | final_All 82.13 | M_d 80.36 | M_f 6.60
note:   -

## 20260925_165802__c100_con_l05_t003_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.03, FT_SEED 42)
result: S0_All 92.10 | S1_New 80.30 | final_All 82.08 | M_d 80.32 | M_f 6.72
note:   -

## 20260925_171001__c100_con_l05_t003_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.03, FT_SEED 42)
result: S0_All 92.12 | S1_New 80.60 | final_All 82.08 | M_d 80.46 | M_f 6.96
note:   -

## 20260925_172216__c100_con_l05_t003_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.03, FT_SEED 42)
result: S0_All 92.24 | S1_New 81.30 | final_All 81.85 | M_d 80.16 | M_f 7.08
note:   -

## 20260925_173426__c100_con_l05_t003_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.03, FT_SEED 42)
result: S0_All 92.16 | S1_New 80.50 | final_All 82.14 | M_d 80.46 | M_f 6.80
note:   -

## 20260925_174631__c100_extract_con_l05_t012_sd42
why:    tau-axis cell: lambda=0.5, tau=0.12, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   -

## 20260925_231551__c100_con_l05_t012_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 0.12, FT_SEED 42)
result: S0_All 91.92 | S1_New 80.30 | final_All 82.04 | M_d 80.40 | M_f 6.64
note:   -

## 20260925_232758__c100_con_l05_t012_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 0.12, FT_SEED 42)
result: S0_All 91.92 | S1_New 79.80 | final_All 82.12 | M_d 80.50 | M_f 6.62
note:   -

## 20260925_234006__c100_con_l05_t012_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 0.12, FT_SEED 42)
result: S0_All 91.92 | S1_New 79.80 | final_All 82.13 | M_d 80.34 | M_f 6.66
note:   -

## 20260925_235216__c100_con_l05_t012_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 0.12, FT_SEED 42)
result: S0_All 91.94 | S1_New 76.70 | final_All 81.36 | M_d 79.42 | M_f 7.08
note:   -

## 20260926_000430__c100_con_l05_t012_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 0.12, FT_SEED 42)
result: S0_All 91.70 | S1_New 80.00 | final_All 81.88 | M_d 80.42 | M_f 6.80
note:   TAU CURVE at lambda=0.5, 5 points now (n=1 except tau 0.07 which has n=3), vs the lambda=0
        baseline S0 91.84+-0.09 / final 81.84+-0.11:
          tau 0.03   S0 92.13   final 82.06   CON 0.0008
          tau 0.07   S0 92.12   final 82.02   CON 0.005    (n=3)
          tau 0.12   S0 91.88   final 81.91
          tau 0.2    S0 91.29   final 81.15   CON 0.34 (fold-F5 value)
          tau 0.5    S0 91.18   final 80.26
        Monotone in tau, but SATURATED at tau <= 0.07 (0.03 and 0.07 are equal within noise).
        The threshold sits between 0.07 and 0.2: at tau 0.12 S0 91.88 is already back at baseline.

## 20260926_002800__c100_extract_con_l05_t50_sd42
why:    tau-axis cell: lambda=0.5, tau=5.0, 500 imgs/class, 10 epochs, batch 16x8, FT_SEED 42
result: (no metrics.json)
note:   CONTROL: tau deliberately huge, so the softmax over 31 candidates is nearly flat.
        mean CON stays at 3.27 for all 10 epochs (chance = 3.43) -> L_con provides essentially no
        gradient, while the augmentation and the 2 forwards are unchanged. CE 0.1262, same as
        tau 0.03 / 0.07. This separates "contrastive active" from "augmentation present".

## 20260926_055420__c100_con_l05_t50_sd42_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: lambda 0.5, tau 5.0, FT_SEED 42)
result: S0_All 92.22 | S1_New 80.10 | final_All 81.88 | M_d 79.72 | M_f 6.62
note:   -

## 20260926_060641__c100_con_l05_t50_sd42_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: lambda 0.5, tau 5.0, FT_SEED 42)
result: S0_All 92.04 | S1_New 80.40 | final_All 82.05 | M_d 79.80 | M_f 6.36
note:   -

## 20260926_061857__c100_con_l05_t50_sd42_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: lambda 0.5, tau 5.0, FT_SEED 42)
result: S0_All 92.06 | S1_New 82.80 | final_All 82.19 | M_d 80.22 | M_f 6.38
note:   -

## 20260926_063114__c100_con_l05_t50_sd42_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: lambda 0.5, tau 5.0, FT_SEED 42)
result: S0_All 92.16 | S1_New 80.60 | final_All 81.77 | M_d 79.36 | M_f 6.48
note:   -

## 20260926_064334__c100_con_l05_t50_sd42_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: lambda 0.5, tau 5.0, FT_SEED 42)
result: S0_All 92.16 | S1_New 80.20 | final_All 81.73 | M_d 79.52 | M_f 6.66
note:   tau=5.0 gives S0 92.13 / final 81.92, i.e. the SAME as tau 0.03 (92.13) and tau 0.07 (92.12),
        although the mechanism is the opposite end of the temperature range.
        Three configurations with L_con disabled all land at S0 ~ 92.1; every configuration where
        L_con is actually active (tau 0.12-0.5) is at or below the baseline 91.84.
        -> tau is a SWITCH, not a weight, and the gain does not come from the contrastive term.

## 20260926_111754__c100_augcheck
why:    regression check after adding FT_AUG: default must reproduce upstream (no augmentation)
result: (no metrics.json)
note:   FT_AUG unset reproduces c100_con0_probe exactly: "augmentation OFF", views=1, mean CE 1.1756
        identical, features max|diff| 1.15e-5 (GPU noise). The new flag leaves the default path alone.

## 20260926_121316__c100_extract_augonly
why:    AUGMENTATION ONLY: FT_AUG=1 with lambda=0 (2 forwards, no L_con, CE weight 1.0), FT_SEED 42
result: (no metrics.json)
note:   The decisive control for D19: augmentation with NO contrastive term at all and CE weight 1.0
        (lambda=0, FT_AUG=1). Both views are forwarded so the dropout / RNG stream matches the
        tau=5.0 run; CE is computed on view 1 only. NOTE: the log line prints views=1 because that
        field follows FT_LAMBDA, not FT_AUG - the real evidence is the "augmentation ON" line.

## 20260926_182158__c100_augonly_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: augmentation only, FT_SEED 42)
result: S0_All 92.18 | S1_New 80.90 | final_All 82.01 | M_d 80.24 | M_f 6.86
note:   -

## 20260926_183412__c100_augonly_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: augmentation only, FT_SEED 42)
result: S0_All 92.18 | S1_New 80.20 | final_All 81.97 | M_d 80.48 | M_f 7.04
note:   -

## 20260926_184618__c100_augonly_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: augmentation only, FT_SEED 42)
result: S0_All 92.12 | S1_New 80.80 | final_All 81.96 | M_d 80.10 | M_f 6.74
note:   -

## 20260926_185829__c100_augonly_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: augmentation only, FT_SEED 42)
result: S0_All 92.32 | S1_New 84.20 | final_All 82.41 | M_d 80.96 | M_f 6.88
note:   -

## 20260926_191036__c100_augonly_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: augmentation only, FT_SEED 42)
result: S0_All 92.18 | S1_New 80.30 | final_All 82.08 | M_d 80.34 | M_f 6.78
note:   5-seed summary, FT_SEED 42: S0_All 92.20+-0.07, final_All 82.09+-0.19, M_d 80.42+-0.33,
        S1_New 81.28+-1.66. Highest single cell in the whole project at this point; beats the paper
        on every metric except S1_New (81.28 vs 83.10). With n=1 this looked like "augmentation is
        the whole story" - seeds 43 and 44 changed that, see c100_augonly_sd44_s4.

## 20260926_200516__c100_extract_augonly_sd43
why:    augmentation only (FT_AUG=1, lambda=0), FT_SEED 43
result: (no metrics.json)
note:   -

## 20260927_015724__c100_augonly_sd43_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: augmentation only, FT_SEED 43)
result: S0_All 92.26 | S1_New 80.30 | final_All 82.08 | M_d 79.82 | M_f 6.38
note:   -

## 20260927_021049__c100_augonly_sd43_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: augmentation only, FT_SEED 43)
result: S0_All 92.20 | S1_New 80.60 | final_All 82.14 | M_d 80.16 | M_f 6.44
note:   -

## 20260927_022510__c100_augonly_sd43_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: augmentation only, FT_SEED 43)
result: S0_All 92.18 | S1_New 80.30 | final_All 82.18 | M_d 79.78 | M_f 6.10
note:   -

## 20260927_023902__c100_augonly_sd43_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: augmentation only, FT_SEED 43)
result: S0_All 92.44 | S1_New 80.20 | final_All 82.13 | M_d 80.04 | M_f 6.62
note:   -

## 20260927_025231__c100_augonly_sd43_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: augmentation only, FT_SEED 43)
result: S0_All 92.30 | S1_New 80.40 | final_All 82.31 | M_d 80.32 | M_f 6.54
note:   FT_SEED 43: S0_All 92.28+-0.10, final_All 82.17+-0.09. Consistent with seed 42.

## 20260927_030528__c100_extract_augonly_sd44
why:    augmentation only (FT_AUG=1, lambda=0), FT_SEED 44
result: (no metrics.json)
note:   -

## 20260927_082727__c100_augonly_sd44_s0
why:    phase 2 on those features (k-means n_init=10), seed 0 (features: augmentation only, FT_SEED 44)
result: S0_All 91.76 | S1_New 79.10 | final_All 81.65 | M_d 79.80 | M_f 6.68
note:   -

## 20260927_083933__c100_augonly_sd44_s1
why:    phase 2 on those features (k-means n_init=10), seed 1 (features: augmentation only, FT_SEED 44)
result: S0_All 91.74 | S1_New 78.40 | final_All 81.62 | M_d 79.82 | M_f 6.58
note:   -

## 20260927_085147__c100_augonly_sd44_s2
why:    phase 2 on those features (k-means n_init=10), seed 2 (features: augmentation only, FT_SEED 44)
result: S0_All 91.72 | S1_New 78.40 | final_All 81.76 | M_d 80.14 | M_f 6.74
note:   -

## 20260927_090400__c100_augonly_sd44_s3
why:    phase 2 on those features (k-means n_init=10), seed 3 (features: augmentation only, FT_SEED 44)
result: S0_All 92.04 | S1_New 78.90 | final_All 80.84 | M_d 78.52 | M_f 7.18
note:   -

## 20260927_091617__c100_augonly_sd44_s4
why:    phase 2 on those features (k-means n_init=10), seed 4 (features: augmentation only, FT_SEED 44)
result: S0_All 91.70 | S1_New 79.20 | final_All 81.33 | M_d 79.14 | M_f 6.70
note:   FINAL PICTURE for Eq. 3 / augmentation, Welch t-test vs the lambda=0 arm (n=5):
                              n   S0_All          p       final_All       p
          lambda 0 (no aug)   5   91.84+-0.09     -       81.84+-0.11     -
          aug only            3   92.09+-0.26   0.229     81.90+-0.40   0.827
          lambda 0.5 tau 0.07 3   92.12+-0.05   0.001     82.02+-0.22   0.298
        Per-seed S0: aug only 92.20 / 92.28 / 91.79 (seed 44 drops to baseline);
                     tau 0.07  92.10 / 92.18 / 92.09 (tight).
        So the two arms have the SAME mean but sigma differs 5x (0.26 vs 0.05), and only the arm
        with a near-zero L_con reaches p < 0.05. Hypothesis: a near-zero L_con acts as a mild
        stabiliser for augmented training. With n=3 per arm this is NOT a result. [INFERENCE]
        CORRECTION: after seed 42 alone I wrote "augmentation is the only cause" - seed 44
        (S0 91.79, final 81.65) shows that arm is unstable. Third time in this project that n=1
        produced a wrong conclusion (see also lambda 0.1 and cell 0.25/0.2).
        No configuration improves the continual metrics significantly; ~14 phase-1 runs per arm
        would be needed for 80% power on final_All (>100 h). Full write-up in D19.
