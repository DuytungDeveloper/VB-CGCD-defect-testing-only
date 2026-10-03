# LOG

One entry per run, appended by exp run. Fill in "why" and "note".
Before the clean rerun: archive/pre-rerun-20260915/ (not in git) and git tag pre-rerun-20260915

## Open questions
Moved to `notes/plan.vi.md` §5 (single list since 2026-09-29).

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

## 20260929_203718__c100_smoke_hungtrain
why: D13 plumbing smoke (5 SVI steps), --hungarian_on train/eval
result: S0_All 84.86 | S1_New 39.90 | final_All 63.68 | M_d 44.64 | M_f 0.88
note: plumbing OK; training permutation-equivariant (loss, train_acc, stop identical); at S2 train/test mappings swap clusters 1/8 (sizes 1136/917): eval -5 Novel samples

## 20260929_203851__c100_smoke_hungeval
why: D13 plumbing smoke (5 SVI steps), --hungarian_on train/eval
result: S0_All 84.86 | S1_New 39.90 | final_All 63.62 | M_d 44.54 | M_f 0.88
note: plumbing OK; training permutation-equivariant (loss, train_acc, stop identical); at S2 train/test mappings swap clusters 1/8 (sizes 1136/917): eval -5 Novel samples

## 20260929_204306__c100_km_hungtrain_s0
why: regression: --hungarian_on train must reproduce c100_km_s0
result: S0_All 91.84 | S1_New 80.20 | final_All 81.86 | M_d 80.14 | M_f 6.64
note: IDENTICAL on 46 metrics; pi equals the smoke run's (clustering independent of SVI steps)

## 20260929_205507__c100_km_evhung_s0
why: D13 fix (Q10b)
result: S0_All 91.84 | S1_New 80.20 | final_All 81.72 | M_d 79.88 | M_f 6.64
note: vs c100_km_s0: differs from S2 on (clusters 1/8 swapped): S2_New -1.3, final_All -0.14

## 20260929_210716__c100_km_evhung_s1
why: D13 fix (Q10b)
result: S0_All 92.00 | S1_New 80.60 | final_All 81.60 | M_d 79.66 | M_f 6.74
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_211926__c100_km_evhung_s2
why: D13 fix (Q10b)
result: S0_All 91.96 | S1_New 79.70 | final_All 81.80 | M_d 79.72 | M_f 6.46
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_213136__c100_km_evhung_s3
why: D13 fix (Q10b)
result: S0_All 91.84 | S1_New 80.40 | final_All 81.78 | M_d 80.12 | M_f 6.74
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_214418__c100_km_evhung_s4
why: D13 fix (Q10b)
result: S0_All 91.86 | S1_New 80.50 | final_All 81.78 | M_d 79.76 | M_f 6.46
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_223749__c100_extract_base400
why: D9 fix: fine-tune on the 20000 labeled images only (FT_LABELED_IDX)
result: (no metrics.json)
note: num_rows 20000, 153 min, check_phase1 OK

## 20260930_011502__c100_km_evhung_base400_s0
why: D9 effect, base400 features, eval mode
result: S0_All 91.68 | S1_New 79.10 | final_All 81.88 | M_d 80.10 | M_f 6.42
note: see the _s4 note

## 20260930_012836__c100_km_evhung_base400_s1
why: D9 effect, base400 features, eval mode
result: S0_All 91.66 | S1_New 82.80 | final_All 82.35 | M_d 80.80 | M_f 6.38
note: outlier seed: balanced S1 clustering (max_min 2.75) -> S1_New 82.80, final 82.35

## 20260930_014215__c100_km_evhung_base400_s2
why: D9 effect, base400 features, eval mode
result: S0_All 91.78 | S1_New 78.40 | final_All 82.08 | M_d 80.00 | M_f 6.16
note: see the _s4 note

## 20260930_015553__c100_km_evhung_base400_s3
why: D9 effect, base400 features, eval mode
result: S0_All 91.84 | S1_New 79.30 | final_All 82.02 | M_d 80.24 | M_f 6.50
note: see the _s4 note

## 20260930_020945__c100_km_evhung_base400_s4
why: D9 effect, base400 features, eval mode
result: S0_All 91.76 | S1_New 79.10 | final_All 81.93 | M_d 80.06 | M_f 6.48
note: 5 seeds: final 82.05+-0.18 vs 81.74 (evhung); S0 -0.16; M_f lower only because S0 is lower; ONE phase-1 run, z~2; commit warning harmless (notes-only commit mid-loop). See D9

## 20260930_031430__neu_extract
why: NEU phase 1: fine-tune on 576 labeled base images (3 x 192), extract 1440 / 360
result: (no metrics.json)
note: CE 1.18 -> 0.0036; shapes OK; pooler saturation 0.026%; diagnostic 5-NN / linear probe 100% on all 6 classes. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_095912__neu_smoke
why: NEU plumbing smoke (5 SVI steps)
result: S0_All 93.89 | S1_New 100.00 | final_All 94.17 | M_d 100.00 | M_f 5.56
note: plumbing OK, but 5 steps cannot reveal the S0 collapse (NaN at ~486). dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_100355__neu_km_evhung_s0
why: NEU baseline, num_dim 384
result: FAILED exit=1
note: FAILED: NaN at S0 step ~486 (n=192 < d=384: MLE of Sigma unbounded; no prior D11; no S0 early stop) -> S0 33.3%, S1 keeps 0/612 -> crash. See D21. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_100522__neu_km_evhung_s1
why: NEU baseline, num_dim 384
result: FAILED exit=1
note: FAILED: NaN at S0 step ~486 (n=192 < d=384: MLE of Sigma unbounded; no prior D11; no S0 early stop) -> S0 33.3%, S1 keeps 0/612 -> crash. See D21. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_100657__neu_km_evhung_s2
why: NEU baseline, num_dim 384
result: FAILED exit=1
note: FAILED: NaN at S0 step ~486 (n=192 < d=384: MLE of Sigma unbounded; no prior D11; no S0 early stop) -> S0 33.3%, S1 keeps 0/612 -> crash. See D21. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_100837__neu_km_evhung_s3
why: NEU baseline, num_dim 384
result: FAILED exit=1
note: FAILED: NaN at S0 step ~486 (n=192 < d=384: MLE of Sigma unbounded; no prior D11; no S0 early stop) -> S0 33.3%, S1 keeps 0/612 -> crash. See D21. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_101004__neu_km_evhung_s4
why: NEU baseline, num_dim 384
result: FAILED exit=1
note: FAILED: NaN at S0 step ~486 (n=192 < d=384: MLE of Sigma unbounded; no prior D11; no S0 early stop) -> S0 33.3%, S1 keeps 0/612 -> crash. See D21. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_114536__neu_km_evhung_d128_s0
why: NEU baseline, num_dim 128 (n/d = 1.5)
result: S0_All 100.00 | S1_New 100.00 | final_All 94.44 | M_d 100.00 | M_f 11.11
note: no NaN, no early stop; see the _s4 note. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_125623__neu_km_evhung_d128_s1
why: NEU baseline, num_dim 128 (n/d = 1.5)
result: S0_All 100.00 | S1_New 100.00 | final_All 94.72 | M_d 100.00 | M_f 10.56
note: no NaN, no early stop; see the _s4 note. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_125846__neu_km_evhung_d128_s2
why: NEU baseline, num_dim 128 (n/d = 1.5)
result: S0_All 100.00 | S1_New 100.00 | final_All 95.28 | M_d 100.00 | M_f 9.44
note: no NaN, no early stop; see the _s4 note. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_130109__neu_km_evhung_d128_s3
why: NEU baseline, num_dim 128 (n/d = 1.5)
result: S0_All 100.00 | S1_New 100.00 | final_All 94.44 | M_d 100.00 | M_f 11.11
note: no NaN, no early stop; see the _s4 note. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_130326__neu_km_evhung_d128_s4
why: NEU baseline, num_dim 128 (n/d = 1.5)
result: S0_All 100.00 | S1_New 100.00 | final_All 94.44 | M_d 100.00 | M_f 11.11
note: 5 seeds: S0 100, S1_New 100, S1_Old 89.33+-0.72; no NaN; early stop never fired (new det ~0.012 vs old ~0.002 after 1000 steps). dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_141133__neu_km_evhung_d128_cm_s0
why: diagnostic: --print_confusion
result: S0_All 100.00 | S1_New 100.00 | final_All 94.44 | M_d 100.00 | M_f 11.11
note: metrics identical to d128_s0; S1 confusion: all 20 forgotten = inclusion (14 -> scratches, 6 -> pitted_surface); crazing, patches 60/60. dirty = README.md only (uncommitted GD2 docs), code as in config.json

## 20260930_150747__neu_extract_frozen
why: label-free end of the label axis: frozen DINO ViT-B/16, CLS token, no fine-tuning
result: (no metrics.json)
note: k-means (PCA 128, k known) cluster acc 100% on all 6 / base / novel; silhouette best k = 6 (thin margin 0.253 vs 0.249); fine-tuned features: 99.0% all, silhouette k = 5. The two frozen runs are bit-identical (15:07 = duplicate). NEU saturates at every label level

## 20260930_172146__neu_extract_frozen
why: label-free end of the label axis: frozen DINO ViT-B/16, CLS token, no fine-tuning
result: (no metrics.json)
note: k-means (PCA 128, k known) cluster acc 100% on all 6 / base / novel; silhouette best k = 6 (thin margin 0.253 vs 0.249); fine-tuned features: 99.0% all, silhouette k = 5. The two frozen runs are bit-identical (15:07 = duplicate). NEU saturates at every label level

## 20260930_201115__neu_extract_frozen_regress
why: regression: generic extractor (0529ea8) must reproduce NEU frozen features
result: (no metrics.json)
note: max |diff| = 0.0 vs 20260930_172146__neu_extract_frozen

## 20260930_204949__gc10_extract_frozen
why: GC10 label-free probe: frozen DINO, CLS, --resize full (2048x1000 -> 224x224)
result: INTERRUPTED (SIGINT)
note: features saved (file sizes match 1833x768 / 455x768 float32 exactly), then hung at exit after an HF ReadTimeout; Ctrl+C -> exit=interrupted. Superseded by 20260930_232359 (bit-identical)

## 20260930_232359__gc10_extract_frozen
why: clean rerun of the GC10 frozen extraction with HF_HUB_OFFLINE=1
result: (no metrics.json)
note: exit 0, bit-identical to 20260930_204949. Probe: k-means acc all 10 = 41.3%, base 51.9%, novel 44.6%; silhouette best k = 2 (true 10); linear probe (all labels) 80.2%, per class [81,74,82,72,90,94,56,67,45,83]. GC10 is NOT saturated

## 20260930_233323__gc10_frozen_km_evhung_d64_s0
why: GC10 baseline: frozen DINO CLS (--resize full 224), labeled S0, k-means n_init 10, eval Hungarian, num_dim 64
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note: see the _s4 note

## 20260930_233811__gc10_frozen_km_evhung_d64_s1
why: GC10 baseline: frozen DINO CLS (--resize full 224), labeled S0, k-means n_init 10, eval Hungarian, num_dim 64
result: S0_All 78.84 | S1_New 56.07 | final_All 62.64 | M_d 56.07 | M_f 10.37
note: see the _s4 note

## 20260930_234024__gc10_frozen_km_evhung_d64_s2
why: GC10 baseline: frozen DINO CLS (--resize full 224), labeled S0, k-means n_init 10, eval Hungarian, num_dim 64
result: S0_All 79.25 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 10.37
note: see the _s4 note

## 20260930_234238__gc10_frozen_km_evhung_d64_s3
why: GC10 baseline: frozen DINO CLS (--resize full 224), labeled S0, k-means n_init 10, eval Hungarian, num_dim 64
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: see the _s4 note

## 20260930_234455__gc10_frozen_km_evhung_d64_s4
why: GC10 baseline: frozen DINO CLS (--resize full 224), labeled S0, k-means n_init 10, eval Hungarian, num_dim 64
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: 5 seeds: S0 78.67+-0.63, final 62.37+-0.96, Old 68.55, New 55.42; balanced base/novel 70.0/37.6 (cm_summary); recall rolled 0, crease 11, incl 14; water->silk 104, incl->silk 100, silk split into rolled/crease clusters; early stop only s0 (step 981, threshold 0.707 x 4.356 = 3.08); no NaN

## 20260930_235006__gc10_frozen_gmm10_evhung_d64_s0
why: clustering ablation on GC10: GMM n_init 10 instead of k-means (same features and settings)
result: S0_All 77.59 | S1_New 49.53 | final_All 59.78 | M_d 49.53 | M_f 8.71
note: see the _s4 note

## 20260930_235228__gc10_frozen_gmm10_evhung_d64_s1
why: clustering ablation on GC10: GMM n_init 10 instead of k-means (same features and settings)
result: S0_All 78.84 | S1_New 57.01 | final_All 63.30 | M_d 57.01 | M_f 9.96
note: see the _s4 note

## 20260930_235459__gc10_frozen_gmm10_evhung_d64_s2
why: clustering ablation on GC10: GMM n_init 10 instead of k-means (same features and settings)
result: S0_All 79.25 | S1_New 52.34 | final_All 60.66 | M_d 52.34 | M_f 11.20
note: see the _s4 note

## 20260930_235723__gc10_frozen_gmm10_evhung_d64_s3
why: clustering ablation on GC10: GMM n_init 10 instead of k-means (same features and settings)
result: S0_All 78.84 | S1_New 54.67 | final_All 61.32 | M_d 54.67 | M_f 11.62
note: see the _s4 note

## 20260930_235953__gc10_frozen_gmm10_evhung_d64_s4
why: clustering ablation on GC10: GMM n_init 10 instead of k-means (same features and settings)
result: S0_All 78.84 | S1_New 50.93 | final_All 60.00 | M_d 50.93 | M_f 10.79
note: 5 seeds: final 61.01+-1.41; balanced novel 37.7+-2.3 (= k-means 37.6); clusters even MORE balanced (max_min 1.66 vs 2.71); rare classes still ~0 -> GMM does not fix imbalance; bottleneck is the features (linear probe weak on the same classes)
        CORRECTION (2026-10-02): "bottleneck is the features" withdrawn. (1) GMM(full, reg_covar 1e-6)
        with n_k < d = 768: log-lik contains 6.91*[(d+1)N - sum n_k^2], so it favours equal-size clusters,
        and EM stays at its k-means init -> gmm10 = the most balanced of 10 single-run k-means partitions
        (post-hoc explanation of max_min 1.66 < 2.71; mechanism not yet checked). The ablation cannot test
        the cluster-capacity hypothesis. (2) Linear probe (gc10_extract_frozen, unweighted LR): balanced
        novel 69.0 vs pipeline 37.6 -> at least 31.4 points are lost in discovery, not in the features.
        Reopened: notes/improvement-2.vi.md, plan.vi.md.

## 20261001_002948__gc10_extract_frozen_224x448
why: resolution probe: frozen DINO at 224x448 (aspect kept), interpolate_pos_encoding
result: (no metrics.json)
note: feat_probe: k-means all/base/novel 47.2/54.3/48.5 (224: 41.3/51.9/44.6); silhouette k=3; linear probe 81.1 (bal 74.7) = flat. Gain comes from the aspect ratio, not from more pixels

## 20261001_003044__gc10_extract_frozen_448x896
why: resolution probe: frozen DINO at 448x896
result: (no metrics.json)
note: feat_probe: k-means 44.4/55.1/47.3; silhouette k=16 (range edge); probe 79.3 (bal 72.8): not better than 224x448

## 20261001_011448__gc10_fr224x448_km_evhung_d64_s0
why: GC10 phase 2 on frozen DINO 224x448 features (proxy k-means had predicted +5.3 balanced novel)
result: S0_All 80.08 | S1_New 51.40 | final_All 58.46 | M_d 51.40 | M_f 15.35
note: see the _s4 note

## 20261001_011705__gc10_fr224x448_km_evhung_d64_s1
why: GC10 phase 2 on frozen DINO 224x448 features (proxy k-means had predicted +5.3 balanced novel)
result: S0_All 80.91 | S1_New 42.52 | final_All 54.29 | M_d 42.52 | M_f 16.18
note: see the _s4 note

## 20261001_011913__gc10_fr224x448_km_evhung_d64_s2
why: GC10 phase 2 on frozen DINO 224x448 features (proxy k-means had predicted +5.3 balanced novel)
result: S0_All 80.91 | S1_New 41.12 | final_All 54.29 | M_d 41.12 | M_f 14.94
note: see the _s4 note

## 20261001_012122__gc10_fr224x448_km_evhung_d64_s3
why: GC10 phase 2 on frozen DINO 224x448 features (proxy k-means had predicted +5.3 balanced novel)
result: S0_All 80.08 | S1_New 40.19 | final_All 53.41 | M_d 40.19 | M_f 14.94
note: see the _s4 note

## 20261001_012335__gc10_fr224x448_km_evhung_d64_s4
why: GC10 phase 2 on frozen DINO 224x448 features (proxy k-means had predicted +5.3 balanced novel)
result: S0_All 80.50 | S1_New 41.12 | final_All 54.29 | M_d 41.12 | M_f 14.52
note: 5 seeds: final 54.95+-2.00 (224: 62.37), New 43.27 (55.42), M_f 15.19 (10.12), S0 80.50 (78.67); balanced novel 38.0 (37.6). silk split (silk->rolled 216 over seeds), incl found 57 (14), water->incl 110. Worse than 224: proxy (own PCA 128 on novel rows) did not predict the pipeline (base-fit PCA 64). inputs.txt back to 224. An accidental 2nd loop (4 runs, interrupted) was deleted before commit


## 20261001_013723__gc10_extract
why: GC10 phase 1 WITH fine-tuning (BOFT, 764 labeled base images, --resize full 224)
result: (no metrics.json)
note: CE 1.69 -> 0.155 (not fully fitted); BOFT factor 1 (A5, fbd_cuda build fails as in every run); feat_probe k-means all/base/novel 59.0/89.3/48.4 (frozen 41.3/51.9/44.6), silhouette k=2, linear probe 82.0 (bal 74.2): flat

## 20261001_014446__gc10_ft_km_evhung_d64_s0
why: GC10 phase 2 on fine-tuned features (full VB-CGCD pipeline: BOFT on labeled base + labeled S0)
result: S0_All 81.33 | S1_New 41.59 | final_All 61.10 | M_d 41.59 | M_f 2.90
note: see the _s4 note

## 20261001_014639__gc10_ft_km_evhung_d64_s1
why: GC10 phase 2 on fine-tuned features (full VB-CGCD pipeline: BOFT on labeled base + labeled S0)
result: S0_All 81.74 | S1_New 43.46 | final_All 62.42 | M_d 43.46 | M_f 2.49
note: see the _s4 note

## 20261001_014833__gc10_ft_km_evhung_d64_s2
why: GC10 phase 2 on fine-tuned features (full VB-CGCD pipeline: BOFT on labeled base + labeled S0)
result: S0_All 81.33 | S1_New 45.33 | final_All 62.64 | M_d 45.33 | M_f 3.32
note: see the _s4 note

## 20261001_015027__gc10_ft_km_evhung_d64_s3
why: GC10 phase 2 on fine-tuned features (full VB-CGCD pipeline: BOFT on labeled base + labeled S0)
result: S0_All 80.91 | S1_New 48.60 | final_All 64.18 | M_d 48.60 | M_f 2.90
note: see the _s4 note

## 20261001_015221__gc10_ft_km_evhung_d64_s4
why: GC10 phase 2 on fine-tuned features (full VB-CGCD pipeline: BOFT on labeled base + labeled S0)
result: S0_All 82.16 | S1_New 35.51 | final_All 58.02 | M_d 35.51 | M_f 4.15
note: 5 seeds: S0 81.49, Old 78.34 (+9.8 vs frozen), New 42.90 (-12.5), M_f 3.15 (10.12), final 61.67 (62.37); balanced base/novel 79.5/33.4 (70.0/37.6). Early stop in 5/5 seeds at 508-617 (frozen 1/5). Retention up, discovery down: silk->rolled 224, silk->crease 122, incl->oil 61. ONE phase-1 run (FT_SEED 42); pooler vs CLS confound

## 20261001_020959__gc10_frozen_km_evhung_d64_regress_s0
why: regression: --s0_labels default must reproduce gc10_frozen_km_evhung_d64_s0
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note: metrics.json identical (all keys); confusion line identical

## 20261001_090854__gc10_fr_nolab_km_evhung_d64_s0
why: GC10 label-free point: frozen DINO + --s0_labels none (S0 = k-means k=5; test Hungarian for evaluation only)
result: S0_All 59.34 | S1_New 51.87 | final_All 53.63 | M_d 51.87 | M_f 4.15
note: see the _s4 note

## 20261001_091109__gc10_fr_nolab_km_evhung_d64_s1
why: GC10 label-free point: frozen DINO + --s0_labels none (S0 = k-means k=5; test Hungarian for evaluation only)
result: S0_All 62.66 | S1_New 32.71 | final_All 46.37 | M_d 32.71 | M_f 4.15
note: see the _s4 note

## 20261001_091319__gc10_fr_nolab_km_evhung_d64_s2
why: GC10 label-free point: frozen DINO + --s0_labels none (S0 = k-means k=5; test Hungarian for evaluation only)
result: S0_All 58.92 | S1_New 56.07 | final_All 55.38 | M_d 56.07 | M_f 4.15
note: see the _s4 note

## 20261001_091533__gc10_fr_nolab_km_evhung_d64_s3
why: GC10 label-free point: frozen DINO + --s0_labels none (S0 = k-means k=5; test Hungarian for evaluation only)
result: S0_All 54.77 | S1_New 56.07 | final_All 52.53 | M_d 56.07 | M_f 5.39
note: see the _s4 note

## 20261001_091748__gc10_fr_nolab_km_evhung_d64_s4
why: GC10 label-free point: frozen DINO + --s0_labels none (S0 = k-means k=5; test Hungarian for evaluation only)
result: S0_All 59.34 | S1_New 56.07 | final_All 55.82 | M_d 56.07 | M_f 3.73
note: 5 seeds: S0 59.00+-2.81, Old 54.69, New 50.56+-10.14, M_f 4.31 (low only because S0 is low), final 52.75; balanced base/novel 58.0/36.3 (labeled-S0 frozen 70.0/37.6, fine-tuned 79.5/33.4): novel discovery does not need S0 labels. welding_line recall 6 (->punch 150, ->cresc 84: co-occurring trio). Smallest S0 cluster 53-67 < num_dim 64 in 3/5 seeds (D21 risk), no NaN

## 20261001_093838__gc10_extract_ftseed0
why: GC10 phase 1 fine-tune replicate (FT_SEED in the tag), same settings as gc10_extract (FT 42)
result: (no metrics.json)
note: see the gc10_ftseedN_km_evhung_d64_s4 note

## 20261001_094417__gc10_ftseed0_km_evhung_d64_s0
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: see the _s4 note

## 20261001_094619__gc10_ftseed0_km_evhung_d64_s1
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: see the _s4 note

## 20261001_094819__gc10_ftseed0_km_evhung_d64_s2
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 84.65 | S1_New 49.07 | final_All 66.37 | M_d 49.07 | M_f 2.90
note: see the _s4 note

## 20261001_095014__gc10_ftseed0_km_evhung_d64_s3
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.06 | S1_New 56.54 | final_All 69.67 | M_d 56.54 | M_f 3.73
note: see the _s4 note

## 20261001_095207__gc10_ftseed0_km_evhung_d64_s4
why: GC10 phase 2 on FT-seed-0 features
result: S0_All 85.48 | S1_New 57.01 | final_All 70.11 | M_d 57.01 | M_f 3.73
note: 5 seeds: S0 85.23, Old 81.33, New 55.51, M_f 3.90, final 69.19; balanced base/novel 82.4/39.4. FT seeds 42/0/1: retention gain robust (Old 78.3/81.3/82.7 vs frozen 68.6, M_f 3.2/3.9/2.2 vs 10.1); discovery NOT robust (bal novel 33.4/39.4/29.2 vs frozen 37.6): phase-1 seed variance dominates

## 20261001_095401__gc10_extract_ftseed1
why: GC10 phase 1 fine-tune replicate (FT_SEED in the tag), same settings as gc10_extract (FT 42)
result: (no metrics.json)
note: see the gc10_ftseedN_km_evhung_d64_s4 note

## 20261001_095933__gc10_ftseed1_km_evhung_d64_s0
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.06 | S1_New 39.25 | final_All 62.20 | M_d 39.25 | M_f 2.49
note: see the _s4 note

## 20261001_100132__gc10_ftseed1_km_evhung_d64_s1
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.06 | S1_New 39.72 | final_All 62.20 | M_d 39.72 | M_f 2.90
note: see the _s4 note

## 20261001_100330__gc10_ftseed1_km_evhung_d64_s2
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 85.06 | S1_New 40.19 | final_All 62.64 | M_d 40.19 | M_f 2.49
note: see the _s4 note

## 20261001_100525__gc10_ftseed1_km_evhung_d64_s3
why: GC10 phase 2 on FT-seed replicate features
result: S0_All 84.23 | S1_New 41.12 | final_All 63.08 | M_d 41.12 | M_f 1.66
note: see the _s4 note

## 20261001_100716__gc10_ftseed1_km_evhung_d64_s4
why: GC10 phase 2 on FT-seed-1 features
result: S0_All 85.48 | S1_New 36.92 | final_All 61.76 | M_d 36.92 | M_f 1.66
note: 5 seeds: S0 84.98, Old 82.74, New 39.44, M_f 2.24, final 62.37; balanced base/novel 83.4/29.2. FT seeds 42/0/1: retention gain robust (Old 78.3/81.3/82.7 vs frozen 68.6, M_f 3.2/3.9/2.2 vs 10.1); discovery NOT robust (bal novel 33.4/39.4/29.2 vs frozen 37.6): phase-1 seed variance dominates

## 20261001_132428__gc10_frozen_km_evhung_d64_regress2_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261001_132728__neu_km_evhung_d384_k64_s0
why: improvement 1 v1 mechanism test: NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_132931__neu_km_evhung_d384_k64_s1
why: improvement 1 v1 mechanism test: NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133148__neu_km_evhung_d384_k64_s2
why: improvement 1 v1 mechanism test: NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133336__neu_km_evhung_d384_k64_s3
why: improvement 1 v1 mechanism test: NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133514__neu_km_evhung_d384_k64_s4
why: improvement 1 v1 mechanism test: NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_135215__gc10_frozen_km_evhung_d64_k64_regress_s0
why: meant as the v2 regression (pooled), but no v1 GC10 run existed: GC10 kappa 64 pooled, seed 0
result: S0_All 78.42 | S1_New 42.06 | final_All 59.34 | M_d 42.06 | M_f 3.73
note: S0 78.42, Old 74.69, New 42.06, M_f 3.73 (kappa 0: 77.59 / 67.63 / 52.80 / 9.96). Learned Sigma eigenvalue mean 1.031 (kappa 0: 1.029) and F-pass new-class dets identical to kappa 0: the prior is inert (D23); differences are side effects

## 20261001_135839__neu_km_evhung_d384_k64_regress_s0
why: v2 code, pooled target: regression against neu_km_evhung_d384_k64_s0 (v1 code)
result: FAILED exit=1
note: NOT identical: loss equal up to step 400, diverges from step 500 (S0 74.44 vs 81.67). Likely GPU numerical non-determinism amplified in the near-singular regime; not re-checked because the SVI prior was dropped (D23)

## 20261001_191430__gc10_frozen_km_evhung_d64_regress3_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261001_192001__gc10_frozen_km_evhung_d64_k64_fix_s0
why:
result: S0_All 78.42 | S1_New 53.27 | final_All 61.32 | M_d 53.27 | M_f 9.96
note:

## 20261001_193036__gc10_frozen_km_evhung_d64_lr10_s0
why:
result: S0_All 79.25 | S1_New 53.27 | final_All 64.40 | M_d 53.27 | M_f 4.98
note:

## 20261001_193256__gc10_frozen_km_evhung_d64_lr10_s1
why:
result: S0_All 79.67 | S1_New 58.41 | final_All 67.03 | M_d 58.41 | M_f 4.98
note:

## 20261001_193516__gc10_frozen_km_evhung_d64_lr10_s2
why:
result: S0_All 78.84 | S1_New 59.81 | final_All 67.69 | M_d 59.81 | M_f 4.15
note:

## 20261001_193728__gc10_frozen_km_evhung_d64_lr10_s3
why:
result: S0_All 79.25 | S1_New 58.41 | final_All 67.03 | M_d 58.41 | M_f 4.56
note:

## 20261001_193944__gc10_frozen_km_evhung_d64_lr10_s4
why:
result: S0_All 79.25 | S1_New 59.81 | final_All 67.69 | M_d 59.81 | M_f 4.56
note:

## 20261001_194532__gc10_frozen_km_evhung_d64_lr100_s0
why:
result: S0_All 79.25 | S1_New 2.80 | final_All 1.32 | M_d 2.80 | M_f 79.25
note:

## 20261001_194727__gc10_frozen_km_evhung_d64_lr100_s1
why:
result: S0_All 80.08 | S1_New 52.80 | final_All 63.74 | M_d 52.80 | M_f 6.64
note:

## 20261001_194940__gc10_frozen_km_evhung_d64_lr100_s2
why:
result: S0_All 79.25 | S1_New 53.74 | final_All 64.40 | M_d 53.74 | M_f 5.39
note:

## 20261001_195155__gc10_frozen_km_evhung_d64_lr100_s3
why:
result: S0_All 80.50 | S1_New 2.80 | final_All 1.32 | M_d 2.80 | M_f 80.50
note:

## 20261001_195333__gc10_frozen_km_evhung_d64_lr100_s4
why:
result: S0_All 80.08 | S1_New 53.27 | final_All 64.40 | M_d 53.27 | M_f 5.81
note:

## 20261002_005645__gc10_frozen_km_evhung_d64_regress4_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261002_010354__gc10_frozen_km_evhung_d64_closed_s0
why:
result: S0_All 81.74 | S1_New 55.14 | final_All 65.93 | M_d 55.14 | M_f 6.22
note:

## 20261002_012306__gc10_frozen_km_evhung_d64_closed_s1
why:
result: S0_All 81.33 | S1_New 56.54 | final_All 66.81 | M_d 56.54 | M_f 5.39
note:

## 20261002_012322__gc10_frozen_km_evhung_d64_closed_s2
why:
result: S0_All 82.99 | S1_New 57.48 | final_All 67.03 | M_d 57.48 | M_f 7.47
note:

## 20261002_012332__gc10_frozen_km_evhung_d64_closed_s3
why:
result: S0_All 81.74 | S1_New 57.94 | final_All 67.91 | M_d 57.94 | M_f 4.98
note:

## 20261002_012342__gc10_frozen_km_evhung_d64_closed_s4
why:
result: S0_All 82.16 | S1_New 56.54 | final_All 67.47 | M_d 56.54 | M_f 4.98
note:

## 20261002_012431__gc10_fr_nolab_km_evhung_d64_closed_s0
why:
result: S0_All 59.75 | S1_New 54.21 | final_All 56.92 | M_d 54.21 | M_f 0.41
note:

## 20261002_012442__gc10_fr_nolab_km_evhung_d64_closed_s1
why:
result: S0_All 64.32 | S1_New 52.80 | final_All 58.68 | M_d 52.80 | M_f 0.41
note:

## 20261002_012453__gc10_fr_nolab_km_evhung_d64_closed_s2
why:
result: S0_All 59.34 | S1_New 56.07 | final_All 57.36 | M_d 56.07 | M_f 0.83
note:

## 20261002_012503__gc10_fr_nolab_km_evhung_d64_closed_s3
why:
result: S0_All 56.43 | S1_New 59.81 | final_All 57.80 | M_d 59.81 | M_f 0.41
note:

## 20261002_012514__gc10_fr_nolab_km_evhung_d64_closed_s4
why:
result: S0_All 59.75 | S1_New 58.88 | final_All 58.90 | M_d 58.88 | M_f 0.83
note:

## 20261002_012617__neu_km_evhung_d128_closed_s0
why:
result: S0_All 98.33 | S1_New 100.00 | final_All 100.00 | M_d 100.00 | M_f -1.67
note:

## 20261002_012628__neu_km_evhung_d128_closed_s1
why:
result: S0_All 98.33 | S1_New 100.00 | final_All 100.00 | M_d 100.00 | M_f -1.67
note:

## 20261002_012639__neu_km_evhung_d128_closed_s2
why:
result: S0_All 99.44 | S1_New 100.00 | final_All 100.00 | M_d 100.00 | M_f -0.56
note:

## 20261002_012649__neu_km_evhung_d128_closed_s3
why:
result: S0_All 98.89 | S1_New 100.00 | final_All 99.72 | M_d 100.00 | M_f -0.56
note:

## 20261002_012701__neu_km_evhung_d128_closed_s4
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 99.44 | M_d 100.00 | M_f -1.11
note:

## 20261002_012728__neu_km_evhung_d384_closed_s0
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 98.06 | M_d 100.00 | M_f 1.67
note:

## 20261002_012818__neu_km_evhung_d384_closed_s1
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 97.78 | M_d 100.00 | M_f 2.22
note:

## 20261002_012900__neu_km_evhung_d384_closed_s2
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 2.78
note:

## 20261002_012913__neu_km_evhung_d384_closed_s3
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 2.78
note:

## 20261002_012946__neu_km_evhung_d384_closed_s4
why:
result: S0_All 97.78 | S1_New 100.00 | final_All 97.78 | M_d 100.00 | M_f 2.22
note:

## 20261002_014718__c100_km_evhung_closed_s0
why:
result: S0_All 84.40 | S1_New 74.40 | final_All 70.68 | M_d 75.34 | M_f 13.80
note:

## 20261002_014834__c100_km_evhung_closed_s1
why:
result: S0_All 84.40 | S1_New 74.00 | final_All 70.82 | M_d 74.84 | M_f 13.28
note:

## 20261002_015738__c100_km_evhung_closed_s2
why:
result: S0_All 84.40 | S1_New 75.20 | final_All 70.83 | M_d 74.88 | M_f 13.20
note:

## 20261002_020323__c100_km_evhung_closed_s3
why:
result: S0_All 84.40 | S1_New 74.80 | final_All 70.37 | M_d 74.68 | M_f 13.66
note:

## 20261002_020551__c100_km_evhung_closed_s4
why:
result: S0_All 84.40 | S1_New 74.70 | final_All 70.56 | M_d 75.08 | M_f 13.74
note:

## 20261002_024424__gc10_frozen_km_evhung_d64_regress5_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261002_025149__gc10_frozen_km_evhung_d64_closed_mask_s0
why:
result: S0_All 81.74 | S1_New 55.14 | final_All 65.93 | M_d 55.14 | M_f 6.22
note:

## 20261002_025212__c100_km_evhung_closed_mask_s0
why:
result: S0_All 87.60 | S1_New 74.40 | final_All 70.68 | M_d 75.34 | M_f 17.00
note:

## 20261002_095429__c100_km_evhung_closed_oracle_s0
why:
result: S0_All 84.40 | S1_New 74.50 | final_All 74.51 | M_d 74.56 | M_f 7.48
note:

## 20261002_100411__c100_km_evhung_closed_mask_oracle_s0
why:
result: S0_All 87.60 | S1_New 74.50 | final_All 74.51 | M_d 74.56 | M_f 10.68
note:

## 20261002_172051__gc10_frozen_km_evhung_d64_regress6_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261002_173759__gc10_frozen_km_evhung_d64_xtab_s0
why:
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note:

## 20261002_174018__gc10_frozen_km_evhung_d64_xtab_s1
why:
result: S0_All 78.84 | S1_New 56.07 | final_All 62.64 | M_d 56.07 | M_f 10.37
note:

## 20261002_174233__gc10_frozen_km_evhung_d64_xtab_s2
why:
result: S0_All 79.25 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 10.37
note:

## 20261002_174449__gc10_frozen_km_evhung_d64_xtab_s3
why:
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note:

## 20261002_174703__gc10_frozen_km_evhung_d64_xtab_s4
why:
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note:

## 2026-10-03 role split
why: laptop = exploration only; official numbers are run on the server with tag prefix srv_
result: laptop runs up to here are archived at tag archive-2026-10-02 (git show archive-2026-10-02:runs/<run>/metrics.json); console.log backup on drive E: vbcgcd-console-2026-10-02.tgz
note: run folders and agg files removed from the working branch; runs/paper, columns.txt, inputs.txt kept
