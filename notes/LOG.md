# LOG

One entry per run, appended by exp run. Fill in "why" and "note".
Before the clean rerun: archive/pre-rerun-20260915/ (not in git) and git tag pre-rerun-20260915.
On 2026-10-15 the folder was packed into a tar on drive E: and removed. Restore from the repo root:
`mkdir -p archive && tar -xf /mnt/e/AI_Storage/backups/vbcgcd-archive-pre-rerun-20260915.tar -C archive`
Codes follow QD22 since 2026-10-19; old codes: `notes/old-codes.vi.md`; the text before the rewrite is at commit `e75889f`.

## Open questions
Moved to `notes/plan.vi.md` §5 on 2026-09-29; since 2026-10-03 (plan.vi.md frozen) in `notes/pl/questions.vi.md`.

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
note:   OK, 16 PASS, ckpt 888,676 (no new module, QD05 = no projection head).
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
why: D13 fix (QD15)
result: S0_All 91.84 | S1_New 80.20 | final_All 81.72 | M_d 79.88 | M_f 6.64
note: vs c100_km_s0: differs from S2 on (clusters 1/8 swapped): S2_New -1.3, final_All -0.14

## 20260929_210716__c100_km_evhung_s1
why: D13 fix (QD15)
result: S0_All 92.00 | S1_New 80.60 | final_All 81.60 | M_d 79.66 | M_f 6.74
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_211926__c100_km_evhung_s2
why: D13 fix (QD15)
result: S0_All 91.96 | S1_New 79.70 | final_All 81.80 | M_d 79.72 | M_f 6.46
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_213136__c100_km_evhung_s3
why: D13 fix (QD15)
result: S0_All 91.84 | S1_New 80.40 | final_All 81.78 | M_d 80.12 | M_f 6.74
note: identical to c100_km on all 46 metrics (mappings agree in every session)

## 20260929_214418__c100_km_evhung_s4
why: D13 fix (QD15)
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
why: E06.01 mechanism test (first version, --cov_prior_kappa): NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_132931__neu_km_evhung_d384_k64_s1
why: E06.01 mechanism test (first version, --cov_prior_kappa): NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133148__neu_km_evhung_d384_k64_s2
why: E06.01 mechanism test (first version, --cov_prior_kappa): NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133336__neu_km_evhung_d384_k64_s3
why: E06.01 mechanism test (first version, --cov_prior_kappa): NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_133514__neu_km_evhung_d384_k64_s4
why: E06.01 mechanism test (first version, --cov_prior_kappa): NEU d 384 (5/5 NaN under MLE) + inverse-Wishart prior, kappa 64, pooled target
result: FAILED exit=1
note: FAILED the pre-registered test: S0 NaN in s2 (step 531) and s3 (515); S0 test decays after step 400 while train stays 100%; S1 F-pass train_acc 0, novel kept 0, crash in every seed. Cause: the optimisation budget (D23), not only the target

## 20261001_135215__gc10_frozen_km_evhung_d64_k64_regress_s0
why: meant as the regression of bd70b17 (pooled target), but no GC10 run of f6d7841 existed: GC10 kappa 64 pooled, seed 0
result: S0_All 78.42 | S1_New 42.06 | final_All 59.34 | M_d 42.06 | M_f 3.73
note: S0 78.42, Old 74.69, New 42.06, M_f 3.73 (kappa 0: 77.59 / 67.63 / 52.80 / 9.96). Learned Sigma eigenvalue mean 1.031 (kappa 0: 1.029) and F-pass new-class dets identical to kappa 0: the prior is inert (D23); differences are side effects

## 20261001_135839__neu_km_evhung_d384_k64_regress_s0
why: code bd70b17, pooled target: regression against neu_km_evhung_d384_k64_s0 (code f6d7841)
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
why: E09.03 regression: --print_cluster_xtab off must reproduce the base run.
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note: Matches the base run exactly.

## 20261002_173759__gc10_frozen_km_evhung_d64_xtab_s0
why: E09.03 (pre-registered in notes/improvement-2.vi.md): cluster x true-class crosstab on the S1 pool.
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note: See the _s4 note.

## 20261002_174018__gc10_frozen_km_evhung_d64_xtab_s1
why: E09.03 (pre-registered in notes/improvement-2.vi.md): cluster x true-class crosstab on the S1 pool.
result: S0_All 78.84 | S1_New 56.07 | final_All 62.64 | M_d 56.07 | M_f 10.37
note: See the _s4 note.

## 20261002_174233__gc10_frozen_km_evhung_d64_xtab_s2
why: E09.03 (pre-registered in notes/improvement-2.vi.md): cluster x true-class crosstab on the S1 pool.
result: S0_All 79.25 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 10.37
note: See the _s4 note.

## 20261002_174449__gc10_frozen_km_evhung_d64_xtab_s3
why: E09.03 (pre-registered in notes/improvement-2.vi.md): cluster x true-class crosstab on the S1 pool.
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: See the _s4 note.

## 20261002_174703__gc10_frozen_km_evhung_d64_xtab_s4
why: E09.03 (pre-registered in notes/improvement-2.vi.md): cluster x true-class crosstab on the S1 pool.
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: Metrics match the base runs. R1 (s1-s4): silk takes 3 of 5 clusters; crease and rolled sit in the waist cluster; two near-pure silk clusters take the names rolled/crease (recall 0). E09.03/P1 true, P2 invalid (no ownership condition), P3 true, P4 false; s0 is mode R0. Details: notes/improvement-2.vi.md, E09.03 results.

## 2026-10-03 role split
why: laptop = exploration only; official numbers are run on the server with tag prefix srv_
result: laptop runs up to here are archived at tag archive-2026-10-02 (git show archive-2026-10-02:runs/<run>/metrics.json); console.log backup on drive E: vbcgcd-console-2026-10-02.tgz
note: run folders and agg files removed from the working branch; runs/paper, columns.txt, inputs.txt kept

## 20261004_132842__gc10_frozen_km_evhung_d64_head_s0
why: integrity check at HEAD 831386c after deleting 11 f5_* feature dirs and 3 bit-identical duplicates; regenerates the k=5 crosstabs for the Q09.11 proxy check (console.log is untracked)
result: S0_All 77.59 | S1_New 52.80 | final_All 60.66 | M_d 52.80 | M_f 9.96
note: see the _s4 note

## 20261004_133059__gc10_frozen_km_evhung_d64_head_s1
why: integrity check at HEAD 831386c after deleting 11 f5_* feature dirs and 3 bit-identical duplicates; regenerates the k=5 crosstabs for the Q09.11 proxy check (console.log is untracked)
result: S0_All 78.84 | S1_New 56.07 | final_All 62.64 | M_d 56.07 | M_f 10.37
note: see the _s4 note

## 20261004_133314__gc10_frozen_km_evhung_d64_head_s2
why: integrity check at HEAD 831386c after deleting 11 f5_* feature dirs and 3 bit-identical duplicates; regenerates the k=5 crosstabs for the Q09.11 proxy check (console.log is untracked)
result: S0_All 79.25 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 10.37
note: see the _s4 note

## 20261004_133529__gc10_frozen_km_evhung_d64_head_s3
why: integrity check at HEAD 831386c after deleting 11 f5_* feature dirs and 3 bit-identical duplicates; regenerates the k=5 crosstabs for the Q09.11 proxy check (console.log is untracked)
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: see the _s4 note

## 20261004_133745__gc10_frozen_km_evhung_d64_head_s4
why: integrity check at HEAD 831386c after deleting 11 f5_* feature dirs and 3 bit-identical duplicates; regenerates the k=5 crosstabs for the Q09.11 proxy check (console.log is untracked)
result: S0_All 78.84 | S1_New 56.07 | final_All 62.86 | M_d 56.07 | M_f 9.96
note: metrics.json identical to archive-2026-10-02 xtab_s0-s4 on every key (5/5 seeds); same feature input hash; env unchanged. The crosstabs exist only in the local console.log files.

## 20261005_013042__gc10_fz_cls5first_img80pct
why: renamed frozen set (cls/img naming, scripts/make_label_splits.py): same command and split as 20260930_232359__gc10_extract_frozen
result: (no metrics.json)
note: bit-identical to 20260930_232359__gc10_extract_frozen (diff -rq; same input hash as the runs gc10_frozen_km_evhung_d64_head), which was deleted; runs/inputs.txt now points here

## 20261005_013115__gc10_ft_cls5first_img80pct_r0
why: phase 1 paper setting (first 5 classes labeled, 80% of their train images = 764), FT_SEED 0, 10 epochs = 60 steps; replaces gc10_extract (FT 42) and gc10_extract_ftseed0/1
result: (no metrics.json)
note: see the _r2 note

## 20261005_013822__gc10_ft_cls5first_img80pct_r1
why: phase 1 paper setting (first 5 classes labeled, 80% of their train images = 764), FT_SEED 1, 10 epochs = 60 steps; replaces gc10_extract (FT 42) and gc10_extract_ftseed0/1
result: (no metrics.json)
note: see the _r2 note

## 20261005_014356__gc10_ft_cls5first_img80pct_r2
why: phase 1 paper setting (first 5 classes labeled, 80% of their train images = 764), FT_SEED 2, 10 epochs = 60 steps; replaces gc10_extract (FT 42) and gc10_extract_ftseed0/1
result: (no metrics.json)
note: CE first->last 1.77->0.154, 1.71->0.151, 1.59->0.156 (old FT 42: 1.69->0.155): not fully fitted in 60 steps. Old dirs gc10_extract, gc10_extract_ftseed0/1 deleted; correction 2026-10-05: phase-1 FT is reproducible with the same FT_SEED (phase 2 on r0/r1 gives the same means as gc10_ftseed0/1), so they can be regenerated with FT_SEED 42/0/1

## 20261005_014929__neu_fz_cls3first_img80pct
why: renamed frozen set: same command as 20260930_172146__neu_extract_frozen, --split made explicit
result: (no metrics.json)
note: bit-identical to 20260930_172146__neu_extract_frozen (diff -rq), which was deleted

## 20261005_014947__neu_ft_cls3first_img80pct_r0
why: phase 1 paper setting (first 3 classes labeled, 80% = 576), FT_SEED 0, 10 epochs = 50 steps; replaces neu_extract (FT 42)
result: (no metrics.json)
note: see the _r2 note

## 20261005_015351__neu_ft_cls3first_img80pct_r1
why: phase 1 paper setting (first 3 classes labeled, 80% = 576), FT_SEED 1, 10 epochs = 50 steps; replaces neu_extract (FT 42)
result: (no metrics.json)
note: see the _r2 note

## 20261005_015755__neu_ft_cls3first_img80pct_r2
why: phase 1 paper setting (first 3 classes labeled, 80% = 576), FT_SEED 2, 10 epochs = 50 steps; replaces neu_extract (FT 42)
result: (no metrics.json)
note: CE first->last 1.00->0.0028, 1.12->0.0028, 1.21->0.0037: fits almost perfectly. Old dir neu_extract deleted; correction 2026-10-05: FT is reproducible on GC10 with the same FT_SEED, so neu_extract can likely be regenerated with FT_SEED 42 (not checked on NEU); the D24 NEU check moves to these sets

## 20261005_021012__gc10_ft_cls2top_img80pct_r0
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by top (silk_spot, water_spot), 80pct images each = 604; FT_SEED 0, FT_EPOCHS 12 = 60 steps (fixed-step rule); split splits/gc10_cls2top_img80pct.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_021534__gc10_ft_cls2top_img80pct_r1
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by top (silk_spot, water_spot), 80pct images each = 604; FT_SEED 1, FT_EPOCHS 12 = 60 steps (fixed-step rule); split splits/gc10_cls2top_img80pct.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_022052__gc10_ft_cls2top_img80pct_r2
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by top (silk_spot, water_spot), 80pct images each = 604; FT_SEED 2, FT_EPOCHS 12 = 60 steps (fixed-step rule); split splits/gc10_cls2top_img80pct.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.6325->0.0141, r1 1.1753->0.0119, r2 1.4420->0.0151

## 20261005_022612__gc10_ft_cls5first_img10_r0
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 10 images each = 50; FT_SEED 0, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img10_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_022933__gc10_ft_cls5first_img10_r1
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 10 images each = 50; FT_SEED 1, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img10_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_023256__gc10_ft_cls5first_img10_r2
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 10 images each = 50; FT_SEED 2, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img10_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 2.5696->0.0047, r1 2.4432->0.0061, r2 2.2645->0.0040

## 20261005_023619__gc10_ft_cls5first_img25_r0
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 25 images each = 125; FT_SEED 0, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img25_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_024220__gc10_ft_cls5first_img25_r1
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 25 images each = 125; FT_SEED 1, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img25_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_024820__gc10_ft_cls5first_img25_r2
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 25 images each = 125; FT_SEED 2, FT_EPOCHS 60 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img25_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 2.6269->0.0099, r1 2.4338->0.0091, r2 2.2676->0.0081

## 20261005_025421__gc10_ft_cls5first_img50_r0
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 50 images each = 250; FT_SEED 0, FT_EPOCHS 30 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img50_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_030004__gc10_ft_cls5first_img50_r1
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 50 images each = 250; FT_SEED 1, FT_EPOCHS 30 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img50_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_030550__gc10_ft_cls5first_img50_r2
why: E02.06, phase 1 label budget by images per class: 5 labeled classes by first (punching_hole, welding_line, crescent_gap, water_spot, oil_spot), 50 images each = 250; FT_SEED 2, FT_EPOCHS 30 = 60 steps (fixed-step rule); split splits/gc10_cls5first_img50_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 2.3530->0.0445, r1 2.2238->0.0372, r2 2.0666->0.0308

## 20261005_031134__gc10_ft_cls5top_img80pct_r0
why: E02.06, phase 1 label budget by labeled classes: 5 labeled classes by top (silk_spot, water_spot, welding_line, crescent_gap, punching_hole), 80pct images each = 1054; FT_SEED 0, FT_EPOCHS 7 = 63 steps (fixed-step rule); split splits/gc10_cls5top_img80pct.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_031654__gc10_ft_cls5top_img80pct_r1
why: E02.06, phase 1 label budget by labeled classes: 5 labeled classes by top (silk_spot, water_spot, welding_line, crescent_gap, punching_hole), 80pct images each = 1054; FT_SEED 1, FT_EPOCHS 7 = 63 steps (fixed-step rule); split splits/gc10_cls5top_img80pct.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_032215__gc10_ft_cls5top_img80pct_r2
why: E02.06, phase 1 label budget by labeled classes: 5 labeled classes by top (silk_spot, water_spot, welding_line, crescent_gap, punching_hole), 80pct images each = 1054; FT_SEED 2, FT_EPOCHS 7 = 63 steps (fixed-step rule); split splits/gc10_cls5top_img80pct.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.4315->0.1290, r1 1.3830->0.1270, r2 1.3013->0.1213

## 20261005_032736__neu_ft_cls2rand_img80pct_r0
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by rand (pitted_surface, rolled-in_scale), 80pct images each = 384; FT_SEED 0, FT_EPOCHS 17 = 51 steps (fixed-step rule); split splits/neu_cls2rand_img80pct_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_033211__neu_ft_cls2rand_img80pct_r1
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by rand (patches, pitted_surface), 80pct images each = 384; FT_SEED 1, FT_EPOCHS 17 = 51 steps (fixed-step rule); split splits/neu_cls2rand_img80pct_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_033645__neu_ft_cls2rand_img80pct_r2
why: E02.06, phase 1 label budget by labeled classes: 2 labeled classes by rand (inclusion, rolled-in_scale), 80pct images each = 384; FT_SEED 2, FT_EPOCHS 17 = 51 steps (fixed-step rule); split splits/neu_cls2rand_img80pct_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.1576->0.0010, r1 1.3582->0.0017, r2 1.3609->0.0010

## 20261005_034120__neu_ft_cls3first_img10_r0
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 10 images each = 30; FT_SEED 0, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img10_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_034251__neu_ft_cls3first_img10_r1
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 10 images each = 30; FT_SEED 1, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img10_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_034421__neu_ft_cls3first_img10_r2
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 10 images each = 30; FT_SEED 2, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img10_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.6504->0.0012, r1 1.9152->0.0014, r2 2.0378->0.0018

## 20261005_034554__neu_ft_cls3first_img25_r0
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 25 images each = 75; FT_SEED 0, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img25_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_034911__neu_ft_cls3first_img25_r1
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 25 images each = 75; FT_SEED 1, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img25_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_035229__neu_ft_cls3first_img25_r2
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 25 images each = 75; FT_SEED 2, FT_EPOCHS 50 = 50 steps (fixed-step rule); split splits/neu_cls3first_img25_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.6699->0.0018, r1 1.8780->0.0016, r2 2.0019->0.0022

## 20261005_035548__neu_ft_cls3first_img50_r0
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 50 images each = 150; FT_SEED 0, FT_EPOCHS 25 = 50 steps (fixed-step rule); split splits/neu_cls3first_img50_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_035845__neu_ft_cls3first_img50_r1
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 50 images each = 150; FT_SEED 1, FT_EPOCHS 25 = 50 steps (fixed-step rule); split splits/neu_cls3first_img50_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_040140__neu_ft_cls3first_img50_r2
why: E02.06, phase 1 label budget by images per class: 3 labeled classes by first (crazing, inclusion, patches), 50 images each = 150; FT_SEED 2, FT_EPOCHS 25 = 50 steps (fixed-step rule); split splits/neu_cls3first_img50_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.5565->0.0033, r1 1.7494->0.0030, r2 1.8873->0.0044

## 20261005_040436__neu_ft_cls3rand_img80pct_r0
why: E02.06, phase 1 label budget by labeled classes: 3 labeled classes by rand (pitted_surface, rolled-in_scale, scratches), 80pct images each = 576; FT_SEED 0, FT_EPOCHS 10 = 50 steps (fixed-step rule); split splits/neu_cls3rand_img80pct_r0.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_040840__neu_ft_cls3rand_img80pct_r1
why: E02.06, phase 1 label budget by labeled classes: 3 labeled classes by rand (inclusion, patches, rolled-in_scale), 80pct images each = 576; FT_SEED 1, FT_EPOCHS 10 = 50 steps (fixed-step rule); split splits/neu_cls3rand_img80pct_r1.json
result: (no metrics.json)
note: see the _r2 note

## 20261005_041245__neu_ft_cls3rand_img80pct_r2
why: E02.06, phase 1 label budget by labeled classes: 3 labeled classes by rand (crazing, inclusion, pitted_surface), 80pct images each = 576; FT_SEED 2, FT_EPOCHS 10 = 50 steps (fixed-step rule); split splits/neu_cls3rand_img80pct_r2.json
result: (no metrics.json)
note: CE first->last per seed: r0 1.0443->0.0030, r1 1.1159->0.0019, r2 1.2220->0.0030

## 20261005_202557__gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s0
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r0 (FT_SEED 0); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: see the _s4 note

## 20261005_202758__gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s1
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r0 (FT_SEED 0); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: see the _s4 note

## 20261005_202956__gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s2
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r0 (FT_SEED 0); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 84.65 | S1_New 49.07 | final_All 66.37 | M_d 49.07 | M_f 2.90
note: see the _s4 note

## 20261005_203153__gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s3
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r0 (FT_SEED 0); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.06 | S1_New 56.54 | final_All 69.67 | M_d 56.54 | M_f 3.73
note: see the _s4 note

## 20261005_203349__gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s4
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r0 (FT_SEED 0); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.48 | S1_New 57.01 | final_All 70.11 | M_d 57.01 | M_f 3.73
note: 5-seed mean S0 85.23, Old 81.33, New 55.51, final 69.19, M_f 3.90, balanced novel 39.4: same 5-seed means as gc10_ftseed0_km_evhung_d64 (same FT_SEED)

## 20261005_203543__gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s0
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r1 (FT_SEED 1); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.06 | S1_New 39.25 | final_All 62.20 | M_d 39.25 | M_f 2.49
note: see the _s4 note

## 20261005_203737__gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s1
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r1 (FT_SEED 1); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.06 | S1_New 39.72 | final_All 62.20 | M_d 39.72 | M_f 2.90
note: see the _s4 note

## 20261005_203933__gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s2
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r1 (FT_SEED 1); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.06 | S1_New 40.19 | final_All 62.64 | M_d 40.19 | M_f 2.49
note: see the _s4 note

## 20261005_204126__gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s3
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r1 (FT_SEED 1); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 84.23 | S1_New 41.12 | final_All 63.08 | M_d 41.12 | M_f 1.66
note: see the _s4 note

## 20261005_204319__gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s4
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r1 (FT_SEED 1); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.48 | S1_New 36.92 | final_All 61.76 | M_d 36.92 | M_f 1.66
note: 5-seed mean S0 84.98, Old 82.74, New 39.44, final 62.37, M_f 2.24, balanced novel 29.2: same 5-seed means as gc10_ftseed1_km_evhung_d64 (same FT_SEED)

## 20261005_204510__gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s0
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r2 (FT_SEED 2); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.06 | S1_New 48.13 | final_All 64.84 | M_d 48.13 | M_f 5.39
note: see the _s4 note

## 20261005_204703__gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s1
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r2 (FT_SEED 2); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 86.31 | S1_New 56.07 | final_All 68.79 | M_d 56.07 | M_f 6.22
note: see the _s4 note

## 20261005_204855__gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s2
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r2 (FT_SEED 2); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 86.31 | S1_New 60.28 | final_All 71.21 | M_d 60.28 | M_f 5.39
note: see the _s4 note

## 20261005_205049__gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s3
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r2 (FT_SEED 2); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.89 | S1_New 46.73 | final_All 65.05 | M_d 46.73 | M_f 4.56
note: see the _s4 note

## 20261005_205241__gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s4
why: phase 2 on paper-setting FT features gc10_ft_cls5first_img80pct_r2 (FT_SEED 2); same command as the archived gc10_ftseed0_km_evhung_d64 runs + --print_confusion --print_cluster_xtab
result: S0_All 85.89 | S1_New 58.88 | final_All 70.77 | M_d 58.88 | M_f 4.56
note: 5-seed mean S0 85.89, Old 80.66, New 54.02, final 68.13, M_f 5.22, balanced novel 36.2

## 20261005_205435__neu_ft_cls3first_img80pct_r0_km_evhung_d128_s0
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r0 (FT_SEED 0); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 95.28 | M_d 100.00 | M_f 9.44
note: see the _s4 note

## 20261005_205657__neu_ft_cls3first_img80pct_r0_km_evhung_d128_s1
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r0 (FT_SEED 0); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 95.28 | M_d 100.00 | M_f 9.44
note: see the _s4 note

## 20261005_205922__neu_ft_cls3first_img80pct_r0_km_evhung_d128_s2
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r0 (FT_SEED 0); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 95.56 | M_d 100.00 | M_f 8.89
note: see the _s4 note

## 20261005_210145__neu_ft_cls3first_img80pct_r0_km_evhung_d128_s3
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r0 (FT_SEED 0); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 95.00 | M_d 100.00 | M_f 10.00
note: see the _s4 note

## 20261005_210407__neu_ft_cls3first_img80pct_r0_km_evhung_d128_s4
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r0 (FT_SEED 0); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 95.56 | M_d 100.00 | M_f 8.89
note: 5-seed mean S0 100.00, Old 90.67, New 100.00, final 95.33, M_f 9.33

## 20261005_210633__neu_ft_cls3first_img80pct_r1_km_evhung_d128_s0
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r1 (FT_SEED 1); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 97.22 | M_d 100.00 | M_f 5.56
note: see the _s4 note

## 20261005_210856__neu_ft_cls3first_img80pct_r1_km_evhung_d128_s1
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r1 (FT_SEED 1); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 5.00
note: see the _s4 note

## 20261005_211120__neu_ft_cls3first_img80pct_r1_km_evhung_d128_s2
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r1 (FT_SEED 1); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 5.00
note: see the _s4 note

## 20261005_211342__neu_ft_cls3first_img80pct_r1_km_evhung_d128_s3
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r1 (FT_SEED 1); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 5.00
note: see the _s4 note

## 20261005_211605__neu_ft_cls3first_img80pct_r1_km_evhung_d128_s4
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r1 (FT_SEED 1); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 97.50 | M_d 100.00 | M_f 5.00
note: 5-seed mean S0 100.00, Old 94.89, New 100.00, final 97.44, M_f 5.11

## 20261005_211828__neu_ft_cls3first_img80pct_r2_km_evhung_d128_s0
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r2 (FT_SEED 2); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 96.94 | M_d 100.00 | M_f 6.11
note: see the _s4 note

## 20261005_212051__neu_ft_cls3first_img80pct_r2_km_evhung_d128_s1
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r2 (FT_SEED 2); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 96.94 | M_d 100.00 | M_f 6.11
note: see the _s4 note

## 20261005_212316__neu_ft_cls3first_img80pct_r2_km_evhung_d128_s2
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r2 (FT_SEED 2); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 96.39 | M_d 100.00 | M_f 7.22
note: see the _s4 note

## 20261005_212538__neu_ft_cls3first_img80pct_r2_km_evhung_d128_s3
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r2 (FT_SEED 2); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 96.67 | M_d 100.00 | M_f 6.67
note: see the _s4 note

## 20261005_212803__neu_ft_cls3first_img80pct_r2_km_evhung_d128_s4
why: phase 2 on paper-setting FT features neu_ft_cls3first_img80pct_r2 (FT_SEED 2); same command as the archived neu_km_evhung_d128 runs + --print_confusion --print_cluster_xtab
result: S0_All 100.00 | S1_New 100.00 | final_All 96.39 | M_d 100.00 | M_f 7.22
note: 5-seed mean S0 100.00, Old 93.33, New 100.00, final 96.67, M_f 6.67

## 20261007_150520__gc10_pl01_s0
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 0 | I2_ok 1 | k5_K 5 | k5_ami 0.2916 | k5_size_cv 0.2176 | k5_purity_mean 0.7012 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4396 | k5_owned_silk 4 | k5_best_silk 0.3333 | k5_recall_owned_silk 0.9976 | k5_purity_best_silk 0.7735 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.6917 | k5_inertia 792075.75 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3788 | k10_size_cv 0.3504 | k10_purity_mean 0.7888 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4942 | k10_owned_silk 6 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9000 | k10_purity_best_silk 0.9811 | k10_owned_inclusion 2 | k10_best_inclusion 0.3630 | k10_recall_owned_inclusion 0.6593 | k10_purity_best_inclusion 0.6447 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.8438 | k10_recall_owned_crease 0.8438 | k10_purity_best_crease 0.3803 | k10_owned_waist 1 | k10_best_waist 0.8404 | k10_recall_owned_waist 0.8404 | k10_purity_best_waist 0.9186 | k10_inertia 644341.12 | k15_K 15 | k15_ami 0.3897 | k15_size_cv 0.4744 | k15_purity_mean 0.7386 | k15_old_in_novel_owned 24 | k15_bal_acc_hungarian 0.3996 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8262 | k15_purity_best_silk 0.9905 | k15_owned_inclusion 4 | k15_best_inclusion 0.3111 | k15_recall_owned_inclusion 0.8889 | k15_purity_best_inclusion 0.5753 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.4688 | k15_purity_best_crease 0.3846 | k15_owned_waist 2 | k15_best_waist 0.5000 | k15_recall_owned_waist 0.8404 | k15_purity_best_waist 0.9216 | k15_inertia 571869.75 | k20_K 20 | k20_ami 0.3843 | k20_size_cv 0.6692 | k20_purity_mean 0.8398 | k20_old_in_novel_owned 50 | k20_bal_acc_hungarian 0.3719 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.8952 | k20_purity_best_silk 0.9905 | k20_owned_inclusion 5 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8963 | k20_purity_best_inclusion 0.5000 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.2453 | k20_owned_crease 1 | k20_best_crease 0.2500 | k20_recall_owned_crease 0.2500 | k20_purity_best_crease 1.00 | k20_owned_waist 4 | k20_best_waist 0.3085 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.8286 | k20_inertia 521446.75 | k30_K 30 | k30_ami 0.3836 | k30_size_cv 0.5466 | k30_purity_mean 0.8334 | k30_old_in_novel_owned 14 | k30_bal_acc_hungarian 0.2690 | k30_owned_silk 13 | k30_best_silk 0.1333 | k30_recall_owned_silk 0.9476 | k30_purity_best_silk 0.9655 | k30_owned_inclusion 7 | k30_best_inclusion 0.2370 | k30_recall_owned_inclusion 0.8889 | k30_purity_best_inclusion 0.7273 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 1 | k30_best_crease 0.2812 | k30_recall_owned_crease 0.2812 | k30_purity_best_crease 0.5000 | k30_owned_waist 5 | k30_best_waist 0.3404 | k30_recall_owned_waist 0.8617 | k30_purity_best_waist 1.00 | k30_inertia 463508.81 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_150615__gc10_pl01_s1
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 1 | I2_ok 1 | k5_K 5 | k5_ami 0.3771 | k5_size_cv 0.4593 | k5_purity_mean 0.7446 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9524 | k5_purity_best_silk 0.6654 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5269 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5644 | k5_inertia 789687.44 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3653 | k10_size_cv 0.4294 | k10_purity_mean 0.7430 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3923 | k10_owned_silk 5 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.8810 | k10_purity_best_silk 0.9542 | k10_owned_inclusion 2 | k10_best_inclusion 0.3556 | k10_recall_owned_inclusion 0.5630 | k10_purity_best_inclusion 0.5393 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.7812 | k10_recall_owned_crease 0.7812 | k10_purity_best_crease 0.2976 | k10_owned_waist 2 | k10_best_waist 0.4681 | k10_recall_owned_waist 0.8617 | k10_purity_best_waist 0.9167 | k10_inertia 650454.81 | k15_K 15 | k15_ami 0.3751 | k15_size_cv 0.4651 | k15_purity_mean 0.7609 | k15_old_in_novel_owned 39 | k15_bal_acc_hungarian 0.3618 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8571 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.8444 | k15_purity_best_inclusion 0.5882 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.2407 | k15_owned_waist 3 | k15_best_waist 0.3404 | k15_recall_owned_waist 0.9043 | k15_purity_best_waist 0.9412 | k15_inertia 576601.00 | k20_K 20 | k20_ami 0.3528 | k20_size_cv 0.5814 | k20_purity_mean 0.7728 | k20_old_in_novel_owned 50 | k20_bal_acc_hungarian 0.3368 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.8619 | k20_purity_best_silk 0.9905 | k20_owned_inclusion 7 | k20_best_inclusion 0.2741 | k20_recall_owned_inclusion 0.9333 | k20_purity_best_inclusion 0.6066 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 2 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.7188 | k20_purity_best_crease 0.41 | k20_owned_waist 3 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.8404 | k20_purity_best_waist 0.9697 | k20_inertia 532006.81 | k30_K 30 | k30_ami 0.4042 | k30_size_cv 0.6803 | k30_purity_mean 0.8538 | k30_old_in_novel_owned 23 | k30_bal_acc_hungarian 0.2759 | k30_owned_silk 12 | k30_best_silk 0.1619 | k30_recall_owned_silk 0.9714 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2741 | k30_recall_owned_inclusion 0.8519 | k30_purity_best_inclusion 0.7708 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 3 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.7500 | k30_purity_best_crease 0.5000 | k30_owned_waist 6 | k30_best_waist 0.1809 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.85 | k30_inertia 459409.88 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_150709__gc10_pl01_s2
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 2 | I2_ok 1 | k5_K 5 | k5_ami 0.3767 | k5_size_cv 0.4732 | k5_purity_mean 0.7449 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9500 | k5_purity_best_silk 0.6530 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5269 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5786 | k5_inertia 790486.19 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3776 | k10_size_cv 0.4865 | k10_purity_mean 0.7777 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4093 | k10_owned_silk 5 | k10_best_silk 0.2881 | k10_recall_owned_silk 0.9095 | k10_purity_best_silk 0.9837 | k10_owned_inclusion 2 | k10_best_inclusion 0.3778 | k10_recall_owned_inclusion 0.7037 | k10_purity_best_inclusion 0.5426 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3261 | k10_owned_waist 2 | k10_best_waist 0.5000 | k10_recall_owned_waist 0.8617 | k10_purity_best_waist 0.9400 | k10_inertia 654201.00 | k15_K 15 | k15_ami 0.3602 | k15_size_cv 0.6524 | k15_purity_mean 0.7659 | k15_old_in_novel_owned 34 | k15_bal_acc_hungarian 0.3207 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8786 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.5037 | k15_purity_best_inclusion 0.4878 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5938 | k15_recall_owned_crease 0.5938 | k15_purity_best_crease 0.2714 | k15_owned_waist 3 | k15_best_waist 0.3404 | k15_recall_owned_waist 0.8191 | k15_purity_best_waist 1.00 | k15_inertia 577367.94 | k20_K 20 | k20_ami 0.3977 | k20_size_cv 0.5922 | k20_purity_mean 0.7999 | k20_old_in_novel_owned 34 | k20_bal_acc_hungarian 0.3067 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9095 | k20_purity_best_silk 0.9905 | k20_owned_inclusion 5 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.6308 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4062 | k20_recall_owned_crease 0.4062 | k20_purity_best_crease 0.42 | k20_owned_waist 4 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.9787 | k20_purity_best_waist 0.9697 | k20_inertia 521830.41 | k30_K 30 | k30_ami 0.3971 | k30_size_cv 0.6551 | k30_purity_mean 0.8569 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.2587 | k30_owned_silk 14 | k30_best_silk 0.1500 | k30_recall_owned_silk 0.9833 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2963 | k30_recall_owned_inclusion 0.8370 | k30_purity_best_inclusion 0.7843 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 2 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.6875 | k30_purity_best_crease 0.5172 | k30_owned_waist 7 | k30_best_waist 0.2021 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 1.00 | k30_inertia 458361.66 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_150802__gc10_pl01_s3
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 3 | I2_ok 1 | k5_K 5 | k5_ami 0.3783 | k5_size_cv 0.4731 | k5_purity_mean 0.7536 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9619 | k5_purity_best_silk 0.6654 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5833 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5444 | k5_inertia 788922.19 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3345 | k10_size_cv 0.4641 | k10_purity_mean 0.7591 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4188 | k10_owned_silk 5 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.8786 | k10_purity_best_silk 0.9905 | k10_owned_inclusion 3 | k10_best_inclusion 0.3037 | k10_recall_owned_inclusion 0.6741 | k10_purity_best_inclusion 0.3868 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.6277 | k10_recall_owned_waist 0.9681 | k10_purity_best_waist 0.6782 | k10_inertia 651894.12 | k15_K 15 | k15_ami 0.4014 | k15_size_cv 0.6005 | k15_purity_mean 0.7849 | k15_old_in_novel_owned 27 | k15_bal_acc_hungarian 0.4754 | k15_owned_silk 8 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8881 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.3259 | k15_recall_owned_inclusion 0.6667 | k15_purity_best_inclusion 0.5176 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.8438 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.5625 | k15_owned_waist 2 | k15_best_waist 0.4894 | k15_recall_owned_waist 0.8511 | k15_purity_best_waist 0.90 | k15_inertia 572945.62 | k20_K 20 | k20_ami 0.4197 | k20_size_cv 0.6734 | k20_purity_mean 0.8272 | k20_old_in_novel_owned 38 | k20_bal_acc_hungarian 0.4078 | k20_owned_silk 10 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9571 | k20_purity_best_silk 0.9905 | k20_owned_inclusion 5 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.8815 | k20_purity_best_inclusion 0.7885 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 2 | k20_best_crease 0.4062 | k20_recall_owned_crease 0.6562 | k20_purity_best_crease 0.25 | k20_owned_waist 2 | k20_best_waist 0.5319 | k20_recall_owned_waist 0.8936 | k20_purity_best_waist 0.9259 | k20_inertia 523842.69 | k30_K 30 | k30_ami 0.4012 | k30_size_cv 0.5025 | k30_purity_mean 0.8742 | k30_old_in_novel_owned 36 | k30_bal_acc_hungarian 0.3134 | k30_owned_silk 15 | k30_best_silk 0.1548 | k30_recall_owned_silk 0.9952 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2370 | k30_recall_owned_inclusion 0.8815 | k30_purity_best_inclusion 0.9697 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.2857 | k30_owned_crease 2 | k30_best_crease 0.3750 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.5455 | k30_owned_waist 5 | k30_best_waist 0.3298 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.89 | k30_inertia 458451.47 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_150855__gc10_pl01_s4
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 4 | I2_ok 1 | k5_K 5 | k5_ami 0.3778 | k5_size_cv 0.4752 | k5_purity_mean 0.7477 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9524 | k5_purity_best_silk 0.6554 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5444 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5644 | k5_inertia 790444.38 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3879 | k10_size_cv 0.4185 | k10_purity_mean 0.7654 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3697 | k10_owned_silk 5 | k10_best_silk 0.2929 | k10_recall_owned_silk 0.8976 | k10_purity_best_silk 0.9840 | k10_owned_inclusion 2 | k10_best_inclusion 0.3556 | k10_recall_owned_inclusion 0.6667 | k10_purity_best_inclusion 0.5926 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.6250 | k10_recall_owned_crease 0.6250 | k10_purity_best_crease 0.2778 | k10_owned_waist 2 | k10_best_waist 0.4574 | k10_recall_owned_waist 0.8830 | k10_purity_best_waist 0.8600 | k10_inertia 647633.19 | k15_K 15 | k15_ami 0.3622 | k15_size_cv 0.6207 | k15_purity_mean 0.7778 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.3485 | k15_owned_silk 8 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 0.99 | k15_owned_inclusion 2 | k15_best_inclusion 0.1778 | k15_recall_owned_inclusion 0.3185 | k15_purity_best_inclusion 0.9231 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.2787 | k15_owned_waist 4 | k15_best_waist 0.2979 | k15_recall_owned_waist 0.9149 | k15_purity_best_waist 1.00 | k15_inertia 574406.31 | k20_K 20 | k20_ami 0.3866 | k20_size_cv 0.5954 | k20_purity_mean 0.7819 | k20_old_in_novel_owned 31 | k20_bal_acc_hungarian 0.2978 | k20_owned_silk 10 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.9071 | k20_purity_best_silk 0.9904 | k20_owned_inclusion 4 | k20_best_inclusion 0.2593 | k20_recall_owned_inclusion 0.8593 | k20_purity_best_inclusion 0.5072 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 0 | k20_best_crease 0.0000 | k20_recall_owned_crease 0.0000 | k20_purity_best_crease 0.00 | k20_owned_waist 5 | k20_best_waist 0.3191 | k20_recall_owned_waist 0.9787 | k20_purity_best_waist 0.9677 | k20_inertia 530238.25 | k30_K 30 | k30_ami 0.4180 | k30_size_cv 0.6893 | k30_purity_mean 0.8700 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.2636 | k30_owned_silk 12 | k30_best_silk 0.1714 | k30_recall_owned_silk 0.9738 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2296 | k30_recall_owned_inclusion 0.8889 | k30_purity_best_inclusion 0.9688 | k30_owned_rolled 1 | k30_best_rolled 0.2353 | k30_recall_owned_rolled 0.2353 | k30_purity_best_rolled 0.8000 | k30_owned_crease 2 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.7812 | k30_purity_best_crease 0.7500 | k30_owned_waist 7 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9574 | k30_purity_best_waist 0.95 | k30_inertia 460927.25 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_150950__gc10_pl01_s5
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 5 | I2_ok 1 | k5_K 5 | k5_ami 0.3684 | k5_size_cv 0.4447 | k5_purity_mean 0.7444 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3585 | k5_owned_silk 3 | k5_best_silk 0.4048 | k5_recall_owned_silk 0.9405 | k5_purity_best_silk 0.6489 | k5_owned_inclusion 1 | k5_best_inclusion 0.3778 | k5_recall_owned_inclusion 0.3778 | k5_purity_best_inclusion 0.4951 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5860 | k5_inertia 796313.00 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3781 | k10_size_cv 0.4921 | k10_purity_mean 0.7429 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4192 | k10_owned_silk 5 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.9167 | k10_purity_best_silk 0.9766 | k10_owned_inclusion 2 | k10_best_inclusion 0.3185 | k10_recall_owned_inclusion 0.6000 | k10_purity_best_inclusion 0.5119 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5625 | k10_recall_owned_crease 0.5625 | k10_purity_best_crease 0.3158 | k10_owned_waist 2 | k10_best_waist 0.5426 | k10_recall_owned_waist 0.9255 | k10_purity_best_waist 0.7846 | k10_inertia 657305.00 | k15_K 15 | k15_ami 0.3901 | k15_size_cv 0.5895 | k15_purity_mean 0.7714 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4466 | k15_owned_silk 8 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8857 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.3481 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.5595 | k15_owned_rolled 1 | k15_best_rolled 0.8235 | k15_recall_owned_rolled 0.8235 | k15_purity_best_rolled 0.2333 | k15_owned_crease 0 | k15_best_crease 0.0000 | k15_recall_owned_crease 0.0000 | k15_purity_best_crease 0.0000 | k15_owned_waist 2 | k15_best_waist 0.5638 | k15_recall_owned_waist 0.9043 | k15_purity_best_waist 0.93 | k15_inertia 578589.19 | k20_K 20 | k20_ami 0.4049 | k20_size_cv 0.5743 | k20_purity_mean 0.8272 | k20_old_in_novel_owned 37 | k20_bal_acc_hungarian 0.3521 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9571 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2963 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 1.00 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.61 | k20_owned_waist 5 | k20_best_waist 0.3085 | k20_recall_owned_waist 0.9681 | k20_purity_best_waist 1.00 | k20_inertia 527432.69 | k30_K 30 | k30_ami 0.4391 | k30_size_cv 0.7649 | k30_purity_mean 0.8753 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3134 | k30_owned_silk 14 | k30_best_silk 0.2476 | k30_recall_owned_silk 0.9952 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2296 | k30_recall_owned_inclusion 0.8889 | k30_purity_best_inclusion 1.00 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.2759 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.6875 | k30_purity_best_crease 0.6842 | k30_owned_waist 5 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.87 | k30_inertia 457006.09 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_151044__gc10_pl01_s6
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 6 | I2_ok 1 | k5_K 5 | k5_ami 0.3707 | k5_size_cv 0.4913 | k5_purity_mean 0.7478 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9500 | k5_purity_best_silk 0.6387 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5326 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5935 | k5_inertia 790158.94 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3590 | k10_size_cv 0.3753 | k10_purity_mean 0.7516 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4473 | k10_owned_silk 7 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.9905 | k10_owned_inclusion 1 | k10_best_inclusion 0.3556 | k10_recall_owned_inclusion 0.3556 | k10_purity_best_inclusion 0.6667 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.5312 | k10_purity_best_crease 0.3617 | k10_owned_waist 1 | k10_best_waist 0.9255 | k10_recall_owned_waist 0.9255 | k10_purity_best_waist 0.7768 | k10_inertia 651018.44 | k15_K 15 | k15_ami 0.3726 | k15_size_cv 0.4994 | k15_purity_mean 0.7919 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.3762 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8690 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.3111 | k15_recall_owned_inclusion 0.8815 | k15_purity_best_inclusion 0.8077 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.4688 | k15_purity_best_crease 0.4286 | k15_owned_waist 3 | k15_best_waist 0.3830 | k15_recall_owned_waist 0.9255 | k15_purity_best_waist 1.00 | k15_inertia 573201.88 | k20_K 20 | k20_ami 0.3806 | k20_size_cv 0.5734 | k20_purity_mean 0.7660 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.3146 | k20_owned_silk 9 | k20_best_silk 0.1810 | k20_recall_owned_silk 0.9476 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.6000 | k20_purity_best_inclusion 0.97 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 0 | k20_best_crease 0.0000 | k20_recall_owned_crease 0.0000 | k20_purity_best_crease 0.00 | k20_owned_waist 5 | k20_best_waist 0.2872 | k20_recall_owned_waist 0.9681 | k20_purity_best_waist 0.84 | k20_inertia 526136.88 | k30_K 30 | k30_ami 0.3906 | k30_size_cv 0.6564 | k30_purity_mean 0.8469 | k30_old_in_novel_owned 29 | k30_bal_acc_hungarian 0.2363 | k30_owned_silk 13 | k30_best_silk 0.1381 | k30_recall_owned_silk 0.9548 | k30_purity_best_silk 0.95 | k30_owned_inclusion 6 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.8815 | k30_purity_best_inclusion 0.80 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 2 | k30_best_crease 0.2812 | k30_recall_owned_crease 0.5312 | k30_purity_best_crease 1.00 | k30_owned_waist 7 | k30_best_waist 0.2234 | k30_recall_owned_waist 0.9255 | k30_purity_best_waist 1.00 | k30_inertia 455095.81 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_151138__gc10_pl01_s7
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 7 | I2_ok 1 | k5_K 5 | k5_ami 0.3724 | k5_size_cv 0.5076 | k5_purity_mean 0.7574 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3584 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.9643 | k5_purity_best_silk 0.6354 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5833 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5935 | k5_inertia 790543.31 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3302 | k10_size_cv 0.4255 | k10_purity_mean 0.7998 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3548 | k10_owned_silk 6 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9190 | k10_purity_best_silk 0.9905 | k10_owned_inclusion 2 | k10_best_inclusion 0.3556 | k10_recall_owned_inclusion 0.6593 | k10_purity_best_inclusion 0.6234 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.6064 | k10_recall_owned_waist 0.9468 | k10_purity_best_waist 0.6064 | k10_inertia 648691.31 | k15_K 15 | k15_ami 0.3704 | k15_size_cv 0.4895 | k15_purity_mean 0.7761 | k15_old_in_novel_owned 40 | k15_bal_acc_hungarian 0.3301 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8524 | k15_purity_best_silk 0.99 | k15_owned_inclusion 4 | k15_best_inclusion 0.2815 | k15_recall_owned_inclusion 0.8741 | k15_purity_best_inclusion 0.5758 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5625 | k15_recall_owned_crease 0.5625 | k15_purity_best_crease 0.3103 | k15_owned_waist 2 | k15_best_waist 0.5000 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 0.90 | k15_inertia 570946.44 | k20_K 20 | k20_ami 0.3988 | k20_size_cv 0.4198 | k20_purity_mean 0.8473 | k20_old_in_novel_owned 50 | k20_bal_acc_hungarian 0.3517 | k20_owned_silk 11 | k20_best_silk 0.1357 | k20_recall_owned_silk 0.9881 | k20_purity_best_silk 0.95 | k20_owned_inclusion 4 | k20_best_inclusion 0.3852 | k20_recall_owned_inclusion 0.8519 | k20_purity_best_inclusion 0.90 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.2558 | k20_owned_crease 1 | k20_best_crease 0.2500 | k20_recall_owned_crease 0.2500 | k20_purity_best_crease 0.36 | k20_owned_waist 3 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.97 | k20_inertia 528390.62 | k30_K 30 | k30_ami 0.3766 | k30_size_cv 0.6476 | k30_purity_mean 0.8470 | k30_old_in_novel_owned 33 | k30_bal_acc_hungarian 0.2760 | k30_owned_silk 14 | k30_best_silk 0.1738 | k30_recall_owned_silk 0.9429 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2370 | k30_recall_owned_inclusion 0.9037 | k30_purity_best_inclusion 0.70 | k30_owned_rolled 1 | k30_best_rolled 0.4118 | k30_recall_owned_rolled 0.4118 | k30_purity_best_rolled 0.2692 | k30_owned_crease 2 | k30_best_crease 0.3125 | k30_recall_owned_crease 0.5625 | k30_purity_best_crease 1.00 | k30_owned_waist 6 | k30_best_waist 0.2447 | k30_recall_owned_waist 0.9149 | k30_purity_best_waist 0.92 | k30_inertia 462264.50 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_151231__gc10_pl01_s8
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 8 | I2_ok 1 | k5_K 5 | k5_ami 0.3806 | k5_size_cv 0.4780 | k5_purity_mean 0.7521 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3579 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.9643 | k5_purity_best_silk 0.6554 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5833 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5714 | k5_inertia 787272.12 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3767 | k10_size_cv 0.4150 | k10_purity_mean 0.7510 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3934 | k10_owned_silk 5 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.8667 | k10_purity_best_silk 0.9690 | k10_owned_inclusion 2 | k10_best_inclusion 0.3407 | k10_recall_owned_inclusion 0.6296 | k10_purity_best_inclusion 0.4842 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.8125 | k10_recall_owned_crease 0.8125 | k10_purity_best_crease 0.3095 | k10_owned_waist 2 | k10_best_waist 0.4574 | k10_recall_owned_waist 0.8511 | k10_purity_best_waist 0.9556 | k10_inertia 646193.31 | k15_K 15 | k15_ami 0.4015 | k15_size_cv 0.5512 | k15_purity_mean 0.8059 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4454 | k15_owned_silk 8 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8976 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.4148 | k15_recall_owned_inclusion 0.8519 | k15_purity_best_inclusion 0.8485 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.4194 | k15_owned_waist 3 | k15_best_waist 0.3936 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.90 | k15_inertia 573123.25 | k20_K 20 | k20_ami 0.4086 | k20_size_cv 0.6170 | k20_purity_mean 0.8047 | k20_old_in_novel_owned 30 | k20_bal_acc_hungarian 0.3238 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9429 | k20_purity_best_silk 0.99 | k20_owned_inclusion 5 | k20_best_inclusion 0.2370 | k20_recall_owned_inclusion 0.8963 | k20_purity_best_inclusion 0.68 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.5000 | k20_recall_owned_crease 0.5000 | k20_purity_best_crease 0.47 | k20_owned_waist 3 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.97 | k20_inertia 526748.44 | k30_K 30 | k30_ami 0.3837 | k30_size_cv 0.6534 | k30_purity_mean 0.8442 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.3052 | k30_owned_silk 14 | k30_best_silk 0.1310 | k30_recall_owned_silk 0.9286 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2963 | k30_recall_owned_inclusion 0.8444 | k30_purity_best_inclusion 0.65 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.64 | k30_owned_waist 7 | k30_best_waist 0.3085 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 1.00 | k30_inertia 455408.75 | checks_ok 1
note: see the gc10_pl01_s9 note

## 20261007_151324__gc10_pl01_s9
why: E09.04: k-means with K' in {5,10,15,20,30} on the S1 pool from the pipeline's RNG state; frozen features (fz); pre-registered in notes/pl/experiments.vi.md (commit e02f0af)
result: seed 9 | I2_ok 1 | k5_K 5 | k5_ami 0.3868 | k5_size_cv 0.4837 | k5_purity_mean 0.7529 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3584 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.9643 | k5_purity_best_silk 0.6519 | k5_owned_inclusion 1 | k5_best_inclusion 0.3630 | k5_recall_owned_inclusion 0.3630 | k5_purity_best_inclusion 0.5632 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9787 | k5_recall_owned_waist 0.9787 | k5_purity_best_waist 0.5823 | k5_inertia 790225.62 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3701 | k10_size_cv 0.4537 | k10_purity_mean 0.7270 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3710 | k10_owned_silk 6 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.9690 | k10_owned_inclusion 1 | k10_best_inclusion 0.2222 | k10_recall_owned_inclusion 0.2222 | k10_purity_best_inclusion 0.5455 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5938 | k10_recall_owned_crease 0.5938 | k10_purity_best_crease 0.2879 | k10_owned_waist 2 | k10_best_waist 0.4681 | k10_recall_owned_waist 0.8936 | k10_purity_best_waist 0.8148 | k10_inertia 650245.94 | k15_K 15 | k15_ami 0.3771 | k15_size_cv 0.5505 | k15_purity_mean 0.7425 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4061 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8643 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3481 | k15_recall_owned_inclusion 0.7185 | k15_purity_best_inclusion 0.5054 | k15_owned_rolled 1 | k15_best_rolled 0.6471 | k15_recall_owned_rolled 0.6471 | k15_purity_best_rolled 0.2683 | k15_owned_crease 1 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.4688 | k15_purity_best_crease 0.4286 | k15_owned_waist 3 | k15_best_waist 0.3191 | k15_recall_owned_waist 0.7872 | k15_purity_best_waist 1.00 | k15_inertia 574505.62 | k20_K 20 | k20_ami 0.3914 | k20_size_cv 0.4002 | k20_purity_mean 0.8152 | k20_old_in_novel_owned 38 | k20_bal_acc_hungarian 0.3606 | k20_owned_silk 11 | k20_best_silk 0.1333 | k20_recall_owned_silk 0.9405 | k20_purity_best_silk 0.93 | k20_owned_inclusion 5 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.8815 | k20_purity_best_inclusion 0.68 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4062 | k20_recall_owned_crease 0.4062 | k20_purity_best_crease 0.26 | k20_owned_waist 2 | k20_best_waist 0.5000 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.96 | k20_inertia 528136.50 | k30_K 30 | k30_ami 0.3986 | k30_size_cv 0.7830 | k30_purity_mean 0.8595 | k30_old_in_novel_owned 39 | k30_bal_acc_hungarian 0.3037 | k30_owned_silk 14 | k30_best_silk 0.2476 | k30_recall_owned_silk 0.9595 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2370 | k30_recall_owned_inclusion 0.8889 | k30_purity_best_inclusion 0.68 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.0000 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7812 | k30_purity_best_crease 0.48 | k30_owned_waist 7 | k30_best_waist 0.1809 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 0.89 | k30_inertia 460048.78 | checks_ok 1
note: E09.04 verdict DROP at K'=20: rolled_pit 2/10, crease 1/10 seeds (need 6/10); C1-C3 pass 10/10. Details: notes/pl/experiments.vi.md

## 20261007_151532__gc10_pl01b_r0_s0
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 0 | I2_ok 1 | k5_K 5 | k5_ami 0.4435 | k5_size_cv 0.5712 | k5_purity_mean 0.6686 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5750 | k5_owned_silk 2 | k5_best_silk 0.5238 | k5_recall_owned_silk 0.9286 | k5_purity_best_silk 0.9649 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.4831 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3333 | k5_owned_waist 1 | k5_best_waist 0.8617 | k5_recall_owned_waist 0.8617 | k5_purity_best_waist 0.9000 | k5_inertia 82352.23 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.3980 | k10_size_cv 0.3842 | k10_purity_mean 0.7019 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4200 | k10_owned_silk 4 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.8405 | k10_purity_best_silk 0.9916 | k10_owned_inclusion 3 | k10_best_inclusion 0.2593 | k10_recall_owned_inclusion 0.6593 | k10_purity_best_inclusion 0.4268 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5625 | k10_recall_owned_crease 0.5625 | k10_purity_best_crease 0.3462 | k10_owned_waist 2 | k10_best_waist 0.5106 | k10_recall_owned_waist 0.9787 | k10_purity_best_waist 0.7385 | k10_inertia 68571.34 | k15_K 15 | k15_ami 0.4273 | k15_size_cv 0.4393 | k15_purity_mean 0.7921 | k15_old_in_novel_owned 37 | k15_bal_acc_hungarian 0.3611 | k15_owned_silk 8 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9571 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2519 | k15_recall_owned_inclusion 0.6296 | k15_purity_best_inclusion 0.6939 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5625 | k15_recall_owned_crease 0.5625 | k15_purity_best_crease 0.4737 | k15_owned_waist 2 | k15_best_waist 0.5106 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.77 | k15_inertia 59792.89 | k20_K 20 | k20_ami 0.4827 | k20_size_cv 0.6272 | k20_purity_mean 0.8623 | k20_old_in_novel_owned 26 | k20_bal_acc_hungarian 0.4726 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9333 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.58 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.4815 | k20_owned_crease 2 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.9062 | k20_purity_best_crease 0.70 | k20_owned_waist 2 | k20_best_waist 0.4681 | k20_recall_owned_waist 0.9255 | k20_purity_best_waist 1.00 | k20_inertia 54730.95 | k30_K 30 | k30_ami 0.4371 | k30_size_cv 0.6597 | k30_purity_mean 0.8417 | k30_old_in_novel_owned 18 | k30_bal_acc_hungarian 0.3154 | k30_owned_silk 12 | k30_best_silk 0.1690 | k30_recall_owned_silk 0.9667 | k30_purity_best_silk 0.99 | k30_owned_inclusion 5 | k30_best_inclusion 0.2148 | k30_recall_owned_inclusion 0.7407 | k30_purity_best_inclusion 0.83 | k30_owned_rolled 1 | k30_best_rolled 0.5294 | k30_recall_owned_rolled 0.5294 | k30_purity_best_rolled 0.3750 | k30_owned_crease 3 | k30_best_crease 0.3125 | k30_recall_owned_crease 0.8125 | k30_purity_best_crease 0.91 | k30_owned_waist 5 | k30_best_waist 0.3511 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 1.00 | k30_inertia 48074.38 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_151625__gc10_pl01b_r0_s1
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 1 | I2_ok 1 | k5_K 5 | k5_ami 0.4490 | k5_size_cv 0.5518 | k5_purity_mean 0.6656 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5687 | k5_owned_silk 2 | k5_best_silk 0.5238 | k5_recall_owned_silk 0.9286 | k5_purity_best_silk 0.9692 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.4674 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8438 | k5_recall_owned_crease 0.8438 | k5_purity_best_crease 0.3068 | k5_owned_waist 1 | k5_best_waist 0.8617 | k5_recall_owned_waist 0.8617 | k5_purity_best_waist 0.9101 | k5_inertia 82649.41 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4554 | k10_size_cv 0.4406 | k10_purity_mean 0.7570 | k10_old_in_novel_owned 36 | k10_bal_acc_hungarian 0.4746 | k10_owned_silk 5 | k10_best_silk 0.2738 | k10_recall_owned_silk 0.9286 | k10_purity_best_silk 0.9914 | k10_owned_inclusion 2 | k10_best_inclusion 0.3704 | k10_recall_owned_inclusion 0.6741 | k10_purity_best_inclusion 0.5435 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5625 | k10_recall_owned_crease 0.5625 | k10_purity_best_crease 0.3158 | k10_owned_waist 1 | k10_best_waist 0.8723 | k10_recall_owned_waist 0.8723 | k10_purity_best_waist 0.9318 | k10_inertia 69302.27 | k15_K 15 | k15_ami 0.4536 | k15_size_cv 0.4293 | k15_purity_mean 0.7872 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.4049 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9119 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.6963 | k15_purity_best_inclusion 0.5968 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6875 | k15_recall_owned_crease 0.6875 | k15_purity_best_crease 0.5946 | k15_owned_waist 2 | k15_best_waist 0.5213 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.92 | k15_inertia 59908.95 | k20_K 20 | k20_ami 0.4496 | k20_size_cv 0.4601 | k20_purity_mean 0.8260 | k20_old_in_novel_owned 21 | k20_bal_acc_hungarian 0.4313 | k20_owned_silk 9 | k20_best_silk 0.1548 | k20_recall_owned_silk 0.9738 | k20_purity_best_silk 0.98 | k20_owned_inclusion 5 | k20_best_inclusion 0.2444 | k20_recall_owned_inclusion 0.6889 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.5938 | k20_purity_best_crease 0.86 | k20_owned_waist 3 | k20_best_waist 0.4574 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 0.93 | k20_inertia 55312.72 | k30_K 30 | k30_ami 0.4363 | k30_size_cv 0.5679 | k30_purity_mean 0.8556 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3753 | k30_owned_silk 14 | k30_best_silk 0.1405 | k30_recall_owned_silk 0.9810 | k30_purity_best_silk 0.98 | k30_owned_inclusion 5 | k30_best_inclusion 0.2074 | k30_recall_owned_inclusion 0.7333 | k30_purity_best_inclusion 1.00 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.4583 | k30_owned_crease 2 | k30_best_crease 0.5625 | k30_recall_owned_crease 0.9062 | k30_purity_best_crease 0.56 | k30_owned_waist 5 | k30_best_waist 0.3191 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.88 | k30_inertia 48769.50 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_151718__gc10_pl01b_r0_s2
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 2 | I2_ok 1 | k5_K 5 | k5_ami 0.4221 | k5_size_cv 0.4612 | k5_purity_mean 0.6710 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5709 | k5_owned_silk 3 | k5_best_silk 0.5452 | k5_recall_owned_silk 1.00 | k5_purity_best_silk 0.8980 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3011 | k5_owned_waist 1 | k5_best_waist 0.8723 | k5_recall_owned_waist 0.8723 | k5_purity_best_waist 0.9535 | k5_inertia 83112.66 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4192 | k10_size_cv 0.4234 | k10_purity_mean 0.7390 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4142 | k10_owned_silk 5 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.9452 | k10_purity_best_silk 0.9752 | k10_owned_inclusion 2 | k10_best_inclusion 0.2815 | k10_recall_owned_inclusion 0.5333 | k10_purity_best_inclusion 0.5429 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3333 | k10_owned_waist 2 | k10_best_waist 0.5106 | k10_recall_owned_waist 0.9787 | k10_purity_best_waist 0.8276 | k10_inertia 69190.51 | k15_K 15 | k15_ami 0.4401 | k15_size_cv 0.4058 | k15_purity_mean 0.7895 | k15_old_in_novel_owned 24 | k15_bal_acc_hungarian 0.3796 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.8714 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.3037 | k15_recall_owned_inclusion 0.9111 | k15_purity_best_inclusion 0.5190 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6562 | k15_recall_owned_crease 0.6562 | k15_purity_best_crease 0.4565 | k15_owned_waist 2 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8511 | k15_purity_best_waist 0.91 | k15_inertia 60352.54 | k20_K 20 | k20_ami 0.4589 | k20_size_cv 0.6450 | k20_purity_mean 0.8086 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.4352 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.8595 | k20_purity_best_silk 1.00 | k20_owned_inclusion 5 | k20_best_inclusion 0.2963 | k20_recall_owned_inclusion 0.9259 | k20_purity_best_inclusion 0.53 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.3929 | k20_owned_crease 2 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.8125 | k20_purity_best_crease 0.76 | k20_owned_waist 3 | k20_best_waist 0.3936 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.97 | k20_inertia 55080.65 | k30_K 30 | k30_ami 0.4421 | k30_size_cv 0.7591 | k30_purity_mean 0.8412 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.3375 | k30_owned_silk 11 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9429 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2000 | k30_recall_owned_inclusion 0.7926 | k30_purity_best_inclusion 0.96 | k30_owned_rolled 1 | k30_best_rolled 0.5294 | k30_recall_owned_rolled 0.5294 | k30_purity_best_rolled 0.4737 | k30_owned_crease 1 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.5000 | k30_purity_best_crease 0.80 | k30_owned_waist 7 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 48519.00 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_151811__gc10_pl01b_r0_s3
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 3 | I2_ok 1 | k5_K 5 | k5_ami 0.3460 | k5_size_cv 0.5034 | k5_purity_mean 0.7178 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3151 | k5_owned_silk 3 | k5_best_silk 0.4048 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.6137 | k5_owned_inclusion 1 | k5_best_inclusion 0.3407 | k5_recall_owned_inclusion 0.3407 | k5_purity_best_inclusion 0.5111 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.8298 | k5_recall_owned_waist 0.8298 | k5_purity_best_waist 0.5000 | k5_inertia 82477.73 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4483 | k10_size_cv 0.6359 | k10_purity_mean 0.7621 | k10_old_in_novel_owned 26 | k10_bal_acc_hungarian 0.4540 | k10_owned_silk 4 | k10_best_silk 0.3048 | k10_recall_owned_silk 0.9476 | k10_purity_best_silk 0.7111 | k10_owned_inclusion 2 | k10_best_inclusion 0.2889 | k10_recall_owned_inclusion 0.5630 | k10_purity_best_inclusion 0.5493 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.8125 | k10_recall_owned_crease 0.8125 | k10_purity_best_crease 0.5652 | k10_owned_waist 2 | k10_best_waist 0.4681 | k10_recall_owned_waist 0.8830 | k10_purity_best_waist 0.9362 | k10_inertia 68934.99 | k15_K 15 | k15_ami 0.4689 | k15_size_cv 0.5473 | k15_purity_mean 0.8033 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.4620 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9190 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.5873 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6250 | k15_recall_owned_crease 0.6250 | k15_purity_best_crease 0.9091 | k15_owned_waist 3 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.93 | k15_inertia 59236.22 | k20_K 20 | k20_ami 0.4475 | k20_size_cv 0.6360 | k20_purity_mean 0.8173 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4256 | k20_owned_silk 8 | k20_best_silk 0.2500 | k20_recall_owned_silk 0.9738 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2593 | k20_recall_owned_inclusion 0.6296 | k20_purity_best_inclusion 0.95 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.3871 | k20_owned_crease 1 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.5938 | k20_purity_best_crease 0.95 | k20_owned_waist 4 | k20_best_waist 0.3191 | k20_recall_owned_waist 0.9362 | k20_purity_best_waist 0.86 | k20_inertia 53967.82 | k30_K 30 | k30_ami 0.4327 | k30_size_cv 0.7830 | k30_purity_mean 0.8344 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.3324 | k30_owned_silk 11 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9310 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.8444 | k30_purity_best_inclusion 0.97 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.4211 | k30_owned_crease 3 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.7812 | k30_purity_best_crease 1.00 | k30_owned_waist 7 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 47379.87 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_151905__gc10_pl01b_r0_s4
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 4 | I2_ok 1 | k5_K 5 | k5_ami 0.3457 | k5_size_cv 0.4844 | k5_purity_mean 0.7168 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3167 | k5_owned_silk 3 | k5_best_silk 0.4024 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.6213 | k5_owned_inclusion 1 | k5_best_inclusion 0.3407 | k5_recall_owned_inclusion 0.3407 | k5_purity_best_inclusion 0.4842 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.8404 | k5_recall_owned_waist 0.8404 | k5_purity_best_waist 0.5032 | k5_inertia 82926.21 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4400 | k10_size_cv 0.3792 | k10_purity_mean 0.7135 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.5190 | k10_owned_silk 5 | k10_best_silk 0.2905 | k10_recall_owned_silk 0.9524 | k10_purity_best_silk 0.9531 | k10_owned_inclusion 2 | k10_best_inclusion 0.2593 | k10_recall_owned_inclusion 0.5037 | k10_purity_best_inclusion 0.8333 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 2 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.8750 | k10_purity_best_crease 0.3148 | k10_owned_waist 1 | k10_best_waist 0.8191 | k10_recall_owned_waist 0.8191 | k10_purity_best_waist 0.9747 | k10_inertia 69334.23 | k15_K 15 | k15_ami 0.4492 | k15_size_cv 0.4378 | k15_purity_mean 0.7801 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.3653 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.9556 | k15_purity_best_inclusion 0.5200 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6562 | k15_recall_owned_crease 0.6562 | k15_purity_best_crease 0.4773 | k15_owned_waist 2 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8617 | k15_purity_best_waist 0.93 | k15_inertia 59803.38 | k20_K 20 | k20_ami 0.4615 | k20_size_cv 0.5921 | k20_purity_mean 0.8144 | k20_old_in_novel_owned 15 | k20_bal_acc_hungarian 0.4255 | k20_owned_silk 7 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.8905 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2296 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.74 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.5217 | k20_owned_crease 1 | k20_best_crease 0.6250 | k20_recall_owned_crease 0.6250 | k20_purity_best_crease 0.74 | k20_owned_waist 4 | k20_best_waist 0.3191 | k20_recall_owned_waist 0.9149 | k20_purity_best_waist 1.00 | k20_inertia 54619.96 | k30_K 30 | k30_ami 0.4463 | k30_size_cv 0.6064 | k30_purity_mean 0.8599 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.3451 | k30_owned_silk 13 | k30_best_silk 0.1476 | k30_recall_owned_silk 0.9810 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.8333 | k30_owned_crease 3 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.9062 | k30_purity_best_crease 0.74 | k30_owned_waist 4 | k30_best_waist 0.3298 | k30_recall_owned_waist 0.9255 | k30_purity_best_waist 0.94 | k30_inertia 48489.25 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_151959__gc10_pl01b_r0_s5
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 5 | I2_ok 1 | k5_K 5 | k5_ami 0.4545 | k5_size_cv 0.5687 | k5_purity_mean 0.6733 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5745 | k5_owned_silk 2 | k5_best_silk 0.5214 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.9648 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.5181 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3043 | k5_owned_waist 1 | k5_best_waist 0.8617 | k5_recall_owned_waist 0.8617 | k5_purity_best_waist 0.9101 | k5_inertia 82728.47 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4604 | k10_size_cv 0.4242 | k10_purity_mean 0.7322 | k10_old_in_novel_owned 35 | k10_bal_acc_hungarian 0.4553 | k10_owned_silk 4 | k10_best_silk 0.2786 | k10_recall_owned_silk 0.9167 | k10_purity_best_silk 0.9915 | k10_owned_inclusion 3 | k10_best_inclusion 0.2667 | k10_recall_owned_inclusion 0.6963 | k10_purity_best_inclusion 0.5373 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.6875 | k10_recall_owned_crease 0.6875 | k10_purity_best_crease 0.3188 | k10_owned_waist 1 | k10_best_waist 0.8085 | k10_recall_owned_waist 0.8085 | k10_purity_best_waist 0.9383 | k10_inertia 68760.48 | k15_K 15 | k15_ami 0.4557 | k15_size_cv 0.4615 | k15_purity_mean 0.7943 | k15_old_in_novel_owned 20 | k15_bal_acc_hungarian 0.3795 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8286 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.9407 | k15_purity_best_inclusion 0.5065 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5938 | k15_recall_owned_crease 0.5938 | k15_purity_best_crease 0.8636 | k15_owned_waist 2 | k15_best_waist 0.5319 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.81 | k15_inertia 59592.66 | k20_K 20 | k20_ami 0.4497 | k20_size_cv 0.5719 | k20_purity_mean 0.8178 | k20_old_in_novel_owned 20 | k20_bal_acc_hungarian 0.3993 | k20_owned_silk 10 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.9714 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2370 | k20_recall_owned_inclusion 0.6667 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.94 | k20_owned_waist 3 | k20_best_waist 0.4574 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 0.93 | k20_inertia 54788.99 | k30_K 30 | k30_ami 0.4431 | k30_size_cv 0.8606 | k30_purity_mean 0.8364 | k30_old_in_novel_owned 15 | k30_bal_acc_hungarian 0.3479 | k30_owned_silk 10 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9238 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2000 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.93 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.7059 | k30_owned_crease 3 | k30_best_crease 0.3438 | k30_recall_owned_crease 0.8125 | k30_purity_best_crease 0.92 | k30_owned_waist 5 | k30_best_waist 0.2447 | k30_recall_owned_waist 0.9362 | k30_purity_best_waist 1.00 | k30_inertia 48674.97 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152053__gc10_pl01b_r0_s6
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 6 | I2_ok 1 | k5_K 5 | k5_ami 0.4566 | k5_size_cv 0.6179 | k5_purity_mean 0.6777 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5731 | k5_owned_silk 2 | k5_best_silk 0.5310 | k5_recall_owned_silk 0.95 | k5_purity_best_silk 0.9696 | k5_owned_inclusion 1 | k5_best_inclusion 0.2963 | k5_recall_owned_inclusion 0.2963 | k5_purity_best_inclusion 0.5479 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8438 | k5_recall_owned_crease 0.8438 | k5_purity_best_crease 0.3103 | k5_owned_waist 1 | k5_best_waist 0.8617 | k5_recall_owned_waist 0.8617 | k5_purity_best_waist 0.9101 | k5_inertia 82403.79 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4721 | k10_size_cv 0.3809 | k10_purity_mean 0.7288 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.5954 | k10_owned_silk 4 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.9000 | k10_purity_best_silk 0.9916 | k10_owned_inclusion 3 | k10_best_inclusion 0.2741 | k10_recall_owned_inclusion 0.7185 | k10_purity_best_inclusion 0.7872 | k10_owned_rolled 1 | k10_best_rolled 0.7059 | k10_recall_owned_rolled 0.7059 | k10_purity_best_rolled 0.3750 | k10_owned_crease 1 | k10_best_crease 0.8438 | k10_recall_owned_crease 0.8438 | k10_purity_best_crease 0.4821 | k10_owned_waist 1 | k10_best_waist 0.8723 | k10_recall_owned_waist 0.8723 | k10_purity_best_waist 0.9535 | k10_inertia 68303.83 | k15_K 15 | k15_ami 0.4605 | k15_size_cv 0.4333 | k15_purity_mean 0.8000 | k15_old_in_novel_owned 22 | k15_bal_acc_hungarian 0.4189 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8429 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.9333 | k15_purity_best_inclusion 0.5556 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.9062 | k15_recall_owned_crease 0.9062 | k15_purity_best_crease 0.5370 | k15_owned_waist 2 | k15_best_waist 0.4681 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.94 | k15_inertia 58790.20 | k20_K 20 | k20_ami 0.4327 | k20_size_cv 0.5884 | k20_purity_mean 0.7998 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3510 | k20_owned_silk 9 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.9286 | k20_purity_best_silk 0.99 | k20_owned_inclusion 5 | k20_best_inclusion 0.2296 | k20_recall_owned_inclusion 0.8074 | k20_purity_best_inclusion 0.76 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.6250 | k20_recall_owned_crease 0.6250 | k20_purity_best_crease 0.62 | k20_owned_waist 3 | k20_best_waist 0.4787 | k20_recall_owned_waist 0.9362 | k20_purity_best_waist 0.87 | k20_inertia 54355.29 | k30_K 30 | k30_ami 0.4380 | k30_size_cv 0.7832 | k30_purity_mean 0.8313 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.3603 | k30_owned_silk 12 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9405 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.7556 | k30_purity_best_inclusion 0.68 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.7143 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.7812 | k30_purity_best_crease 0.93 | k30_owned_waist 6 | k30_best_waist 0.3085 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 48038.78 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152147__gc10_pl01b_r0_s7
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 7 | I2_ok 1 | k5_K 5 | k5_ami 0.4505 | k5_size_cv 0.5642 | k5_purity_mean 0.6678 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5692 | k5_owned_silk 2 | k5_best_silk 0.5262 | k5_recall_owned_silk 0.94 | k5_purity_best_silk 0.9651 | k5_owned_inclusion 1 | k5_best_inclusion 0.3111 | k5_recall_owned_inclusion 0.3111 | k5_purity_best_inclusion 0.5000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8438 | k5_recall_owned_crease 0.8438 | k5_purity_best_crease 0.2967 | k5_owned_waist 1 | k5_best_waist 0.8617 | k5_recall_owned_waist 0.8617 | k5_purity_best_waist 0.9000 | k5_inertia 82808.49 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4068 | k10_size_cv 0.3436 | k10_purity_mean 0.7534 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4019 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9286 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3037 | k10_recall_owned_inclusion 0.5704 | k10_purity_best_inclusion 0.5256 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3061 | k10_owned_waist 2 | k10_best_waist 0.5213 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8167 | k10_inertia 69242.64 | k15_K 15 | k15_ami 0.4445 | k15_size_cv 0.4750 | k15_purity_mean 0.8038 | k15_old_in_novel_owned 38 | k15_bal_acc_hungarian 0.4295 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9190 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.7037 | k15_purity_best_inclusion 0.9250 | k15_owned_rolled 1 | k15_best_rolled 0.6471 | k15_recall_owned_rolled 0.6471 | k15_purity_best_rolled 0.3056 | k15_owned_crease 1 | k15_best_crease 0.3125 | k15_recall_owned_crease 0.3125 | k15_purity_best_crease 0.9091 | k15_owned_waist 2 | k15_best_waist 0.5106 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.66 | k15_inertia 61068.93 | k20_K 20 | k20_ami 0.4452 | k20_size_cv 0.6109 | k20_purity_mean 0.7937 | k20_old_in_novel_owned 16 | k20_bal_acc_hungarian 0.3454 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.8857 | k20_purity_best_silk 0.99 | k20_owned_inclusion 5 | k20_best_inclusion 0.2815 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.55 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.70 | k20_owned_waist 4 | k20_best_waist 0.3511 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 1.00 | k20_inertia 54663.23 | k30_K 30 | k30_ami 0.4115 | k30_size_cv 0.7368 | k30_purity_mean 0.8174 | k30_old_in_novel_owned 17 | k30_bal_acc_hungarian 0.3396 | k30_owned_silk 13 | k30_best_silk 0.2500 | k30_recall_owned_silk 0.9619 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2000 | k30_recall_owned_inclusion 0.6815 | k30_purity_best_inclusion 1.00 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.8889 | k30_owned_crease 2 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.79 | k30_owned_waist 5 | k30_best_waist 0.3085 | k30_recall_owned_waist 0.9468 | k30_purity_best_waist 1.00 | k30_inertia 48611.40 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152241__gc10_pl01b_r0_s8
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 8 | I2_ok 1 | k5_K 5 | k5_ami 0.4557 | k5_size_cv 0.5596 | k5_purity_mean 0.6712 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5766 | k5_owned_silk 2 | k5_best_silk 0.5214 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.9733 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.5000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3077 | k5_owned_waist 1 | k5_best_waist 0.8723 | k5_recall_owned_waist 0.8723 | k5_purity_best_waist 0.9111 | k5_inertia 82404.61 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4308 | k10_size_cv 0.4765 | k10_purity_mean 0.7330 | k10_old_in_novel_owned 37 | k10_bal_acc_hungarian 0.3745 | k10_owned_silk 4 | k10_best_silk 0.2786 | k10_recall_owned_silk 0.9310 | k10_purity_best_silk 0.95 | k10_owned_inclusion 2 | k10_best_inclusion 0.3111 | k10_recall_owned_inclusion 0.5704 | k10_purity_best_inclusion 0.5185 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.5312 | k10_purity_best_crease 0.3333 | k10_owned_waist 2 | k10_best_waist 0.4574 | k10_recall_owned_waist 0.9149 | k10_purity_best_waist 0.9348 | k10_inertia 68865.84 | k15_K 15 | k15_ami 0.4403 | k15_size_cv 0.5420 | k15_purity_mean 0.8047 | k15_old_in_novel_owned 23 | k15_bal_acc_hungarian 0.3899 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9214 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2667 | k15_recall_owned_inclusion 0.6963 | k15_purity_best_inclusion 0.5902 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.8438 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.5400 | k15_owned_waist 3 | k15_best_waist 0.4149 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 0.97 | k15_inertia 59936.47 | k20_K 20 | k20_ami 0.4718 | k20_size_cv 0.5888 | k20_purity_mean 0.8213 | k20_old_in_novel_owned 21 | k20_bal_acc_hungarian 0.4234 | k20_owned_silk 8 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.9667 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.6815 | k20_purity_best_inclusion 0.97 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.5000 | k20_owned_crease 1 | k20_best_crease 0.5312 | k20_recall_owned_crease 0.5312 | k20_purity_best_crease 0.94 | k20_owned_waist 4 | k20_best_waist 0.3830 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 1.00 | k20_inertia 54248.24 | k30_K 30 | k30_ami 0.4554 | k30_size_cv 0.8510 | k30_purity_mean 0.8443 | k30_old_in_novel_owned 15 | k30_bal_acc_hungarian 0.3804 | k30_owned_silk 11 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9786 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.7778 | k30_purity_best_inclusion 0.97 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.6875 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.89 | k30_owned_waist 6 | k30_best_waist 0.2872 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.96 | k30_inertia 48333.43 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152335__gc10_pl01b_r0_s9
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 0); pre-registered (commit 885e2dc)
result: seed 9 | I2_ok 1 | k5_K 5 | k5_ami 0.3523 | k5_size_cv 0.4494 | k5_purity_mean 0.7241 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3256 | k5_owned_silk 3 | k5_best_silk 0.4024 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.6450 | k5_owned_inclusion 1 | k5_best_inclusion 0.3852 | k5_recall_owned_inclusion 0.3852 | k5_purity_best_inclusion 0.5098 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.8404 | k5_recall_owned_waist 0.8404 | k5_purity_best_waist 0.4907 | k5_inertia 82795.67 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4384 | k10_size_cv 0.3662 | k10_purity_mean 0.7574 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4168 | k10_owned_silk 4 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.9048 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.2741 | k10_recall_owned_inclusion 0.7259 | k10_purity_best_inclusion 0.5068 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.7188 | k10_recall_owned_crease 0.7188 | k10_purity_best_crease 0.3433 | k10_owned_waist 2 | k10_best_waist 0.4574 | k10_recall_owned_waist 0.8936 | k10_purity_best_waist 0.9149 | k10_inertia 68328.05 | k15_K 15 | k15_ami 0.4397 | k15_size_cv 0.4487 | k15_purity_mean 0.7892 | k15_old_in_novel_owned 38 | k15_bal_acc_hungarian 0.4402 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8310 | k15_purity_best_silk 0.99 | k15_owned_inclusion 4 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.9333 | k15_purity_best_inclusion 0.5128 | k15_owned_rolled 1 | k15_best_rolled 0.6471 | k15_recall_owned_rolled 0.6471 | k15_purity_best_rolled 0.2500 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.8500 | k15_owned_waist 2 | k15_best_waist 0.4787 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.90 | k15_inertia 59682.41 | k20_K 20 | k20_ami 0.4672 | k20_size_cv 0.5773 | k20_purity_mean 0.8176 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4237 | k20_owned_silk 8 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.9238 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2444 | k20_recall_owned_inclusion 0.8148 | k20_purity_best_inclusion 0.97 | k20_owned_rolled 1 | k20_best_rolled 0.5882 | k20_recall_owned_rolled 0.5882 | k20_purity_best_rolled 0.4762 | k20_owned_crease 2 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.9375 | k20_purity_best_crease 0.73 | k20_owned_waist 3 | k20_best_waist 0.4468 | k20_recall_owned_waist 0.8617 | k20_purity_best_waist 0.98 | k20_inertia 54362.70 | k30_K 30 | k30_ami 0.4282 | k30_size_cv 0.7677 | k30_purity_mean 0.8368 | k30_old_in_novel_owned 14 | k30_bal_acc_hungarian 0.3701 | k30_owned_silk 13 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.9310 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2296 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.58 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.4167 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.6562 | k30_purity_best_crease 0.84 | k30_owned_waist 5 | k30_best_waist 0.2872 | k30_recall_owned_waist 0.8830 | k30_purity_best_waist 0.84 | k30_inertia 47977.07 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152430__gc10_pl01b_r1_s0
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 0 | I2_ok 1 | k5_K 5 | k5_ami 0.4015 | k5_size_cv 0.5407 | k5_purity_mean 0.6985 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5294 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6082 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2308 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8857 | k5_inertia 89441.91 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4845 | k10_size_cv 0.4338 | k10_purity_mean 0.7648 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3797 | k10_owned_silk 4 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.2889 | k10_recall_owned_inclusion 0.7037 | k10_purity_best_inclusion 0.6842 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2800 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8413 | k10_inertia 73728.16 | k15_K 15 | k15_ami 0.4438 | k15_size_cv 0.4916 | k15_purity_mean 0.8052 | k15_old_in_novel_owned 40 | k15_bal_acc_hungarian 0.3938 | k15_owned_silk 6 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.8548 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.9556 | k15_purity_best_inclusion 0.6557 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3333 | k15_owned_waist 3 | k15_best_waist 0.5532 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.83 | k15_inertia 64982.90 | k20_K 20 | k20_ami 0.4610 | k20_size_cv 0.6013 | k20_purity_mean 0.8283 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.4238 | k20_owned_silk 9 | k20_best_silk 0.2500 | k20_recall_owned_silk 0.9500 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.74 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.3421 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.64 | k20_owned_waist 3 | k20_best_waist 0.4149 | k20_recall_owned_waist 0.8723 | k20_purity_best_waist 0.97 | k20_inertia 59532.36 | k30_K 30 | k30_ami 0.4467 | k30_size_cv 0.7902 | k30_purity_mean 0.8432 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3664 | k30_owned_silk 13 | k30_best_silk 0.2476 | k30_recall_owned_silk 0.9857 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.7407 | k30_purity_best_inclusion 0.85 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.8462 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.82 | k30_owned_waist 5 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9468 | k30_purity_best_waist 0.92 | k30_inertia 52287.20 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152524__gc10_pl01b_r1_s1
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 1 | I2_ok 1 | k5_K 5 | k5_ami 0.3956 | k5_size_cv 0.5015 | k5_purity_mean 0.6924 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5245 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6286 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2234 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8692 | k5_inertia 89757.89 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4378 | k10_size_cv 0.3849 | k10_purity_mean 0.8096 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3534 | k10_owned_silk 5 | k10_best_silk 0.2500 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3111 | k10_recall_owned_inclusion 0.6148 | k10_purity_best_inclusion 0.5676 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5938 | k10_recall_owned_crease 0.5938 | k10_purity_best_crease 0.2923 | k10_owned_waist 2 | k10_best_waist 0.5532 | k10_recall_owned_waist 0.9681 | k10_purity_best_waist 0.8525 | k10_inertia 74755.51 | k15_K 15 | k15_ami 0.4716 | k15_size_cv 0.5910 | k15_purity_mean 0.7839 | k15_old_in_novel_owned 19 | k15_bal_acc_hungarian 0.4091 | k15_owned_silk 6 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3037 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.9111 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 2 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.8125 | k15_purity_best_crease 0.4545 | k15_owned_waist 2 | k15_best_waist 0.6702 | k15_recall_owned_waist 0.8404 | k15_purity_best_waist 0.97 | k15_inertia 65860.51 | k20_K 20 | k20_ami 0.4633 | k20_size_cv 0.6762 | k20_purity_mean 0.8416 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3968 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9881 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.6074 | k20_purity_best_inclusion 0.91 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.5000 | k20_owned_crease 2 | k20_best_crease 0.4062 | k20_recall_owned_crease 0.7500 | k20_purity_best_crease 0.72 | k20_owned_waist 4 | k20_best_waist 0.2766 | k20_recall_owned_waist 0.9468 | k20_purity_best_waist 0.90 | k20_inertia 59189.99 | k30_K 30 | k30_ami 0.4426 | k30_size_cv 0.8826 | k30_purity_mean 0.8214 | k30_old_in_novel_owned 21 | k30_bal_acc_hungarian 0.3739 | k30_owned_silk 12 | k30_best_silk 0.2500 | k30_recall_owned_silk 0.9857 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.7704 | k30_purity_best_inclusion 0.87 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.7500 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.68 | k30_owned_waist 6 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.96 | k30_inertia 52723.62 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152619__gc10_pl01b_r1_s2
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 2 | I2_ok 1 | k5_K 5 | k5_ami 0.3989 | k5_size_cv 0.5253 | k5_purity_mean 0.6972 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5323 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6146 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2165 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8692 | k5_inertia 89581.09 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4373 | k10_size_cv 0.4026 | k10_purity_mean 0.7708 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4121 | k10_owned_silk 5 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3704 | k10_recall_owned_inclusion 0.5556 | k10_purity_best_inclusion 0.6667 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.3043 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8413 | k10_inertia 74510.82 | k15_K 15 | k15_ami 0.4764 | k15_size_cv 0.4735 | k15_purity_mean 0.8154 | k15_old_in_novel_owned 20 | k15_bal_acc_hungarian 0.4448 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9405 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.6889 | k15_purity_best_inclusion 0.7018 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.8438 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.5510 | k15_owned_waist 2 | k15_best_waist 0.4468 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.98 | k15_inertia 65262.88 | k20_K 20 | k20_ami 0.4495 | k20_size_cv 0.5609 | k20_purity_mean 0.8319 | k20_old_in_novel_owned 25 | k20_bal_acc_hungarian 0.4551 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9095 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.93 | k20_owned_rolled 1 | k20_best_rolled 0.8235 | k20_recall_owned_rolled 0.8235 | k20_purity_best_rolled 0.3889 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.71 | k20_owned_waist 3 | k20_best_waist 0.4468 | k20_recall_owned_waist 0.9255 | k20_purity_best_waist 0.89 | k20_inertia 59920.51 | k30_K 30 | k30_ami 0.4377 | k30_size_cv 0.6856 | k30_purity_mean 0.8354 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.2570 | k30_owned_silk 12 | k30_best_silk 0.1738 | k30_recall_owned_silk 0.9881 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.7926 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 1 | k30_best_rolled 0.3529 | k30_recall_owned_rolled 0.3529 | k30_purity_best_rolled 0.3750 | k30_owned_crease 2 | k30_best_crease 0.3438 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.73 | k30_owned_waist 6 | k30_best_waist 0.1702 | k30_recall_owned_waist 0.8511 | k30_purity_best_waist 0.94 | k30_inertia 52804.92 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152714__gc10_pl01b_r1_s3
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 3 | I2_ok 1 | k5_K 5 | k5_ami 0.3968 | k5_size_cv 0.5197 | k5_purity_mean 0.7443 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3506 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.96 | k5_purity_best_silk 0.6367 | k5_owned_inclusion 1 | k5_best_inclusion 0.3111 | k5_recall_owned_inclusion 0.3111 | k5_purity_best_inclusion 0.5385 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.5813 | k5_inertia 89382.69 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4686 | k10_size_cv 0.5457 | k10_purity_mean 0.7620 | k10_old_in_novel_owned 21 | k10_bal_acc_hungarian 0.3956 | k10_owned_silk 4 | k10_best_silk 0.2905 | k10_recall_owned_silk 0.9595 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.3333 | k10_recall_owned_inclusion 0.6222 | k10_purity_best_inclusion 0.6818 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8154 | k10_inertia 74249.21 | k15_K 15 | k15_ami 0.4451 | k15_size_cv 0.5069 | k15_purity_mean 0.8067 | k15_old_in_novel_owned 37 | k15_bal_acc_hungarian 0.3700 | k15_owned_silk 7 | k15_best_silk 0.2548 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.6889 | k15_purity_best_inclusion 0.6897 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3714 | k15_owned_waist 3 | k15_best_waist 0.5106 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.83 | k15_inertia 64676.45 | k20_K 20 | k20_ami 0.4640 | k20_size_cv 0.6677 | k20_purity_mean 0.8450 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3750 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9286 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.56 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 2 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.7812 | k20_purity_best_crease 0.70 | k20_owned_waist 3 | k20_best_waist 0.4894 | k20_recall_owned_waist 0.9574 | k20_purity_best_waist 0.88 | k20_inertia 58834.02 | k30_K 30 | k30_ami 0.4336 | k30_size_cv 0.5648 | k30_purity_mean 0.8516 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3673 | k30_owned_silk 14 | k30_best_silk 0.1357 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2667 | k30_recall_owned_inclusion 0.6963 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 1 | k30_best_rolled 0.8235 | k30_recall_owned_rolled 0.8235 | k30_purity_best_rolled 0.5385 | k30_owned_crease 2 | k30_best_crease 0.3125 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.91 | k30_owned_waist 6 | k30_best_waist 0.2979 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.97 | k30_inertia 51927.97 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152808__gc10_pl01b_r1_s4
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 4 | I2_ok 1 | k5_K 5 | k5_ami 0.3482 | k5_size_cv 0.2987 | k5_purity_mean 0.7415 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4059 | k5_owned_silk 3 | k5_best_silk 0.2976 | k5_recall_owned_silk 0.77 | k5_purity_best_silk 0.9766 | k5_owned_inclusion 1 | k5_best_inclusion 0.7111 | k5_recall_owned_inclusion 0.7111 | k5_purity_best_inclusion 0.4923 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.4745 | k5_inertia 90109.13 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4874 | k10_size_cv 0.4522 | k10_purity_mean 0.7403 | k10_old_in_novel_owned 26 | k10_bal_acc_hungarian 0.5288 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.8548 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.4519 | k10_recall_owned_inclusion 0.9481 | k10_purity_best_inclusion 0.5304 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.5312 | k10_purity_best_crease 0.3469 | k10_owned_waist 1 | k10_best_waist 0.8511 | k10_recall_owned_waist 0.8511 | k10_purity_best_waist 0.9756 | k10_inertia 75282.20 | k15_K 15 | k15_ami 0.4541 | k15_size_cv 0.5991 | k15_purity_mean 0.7795 | k15_old_in_novel_owned 29 | k15_bal_acc_hungarian 0.4601 | k15_owned_silk 7 | k15_best_silk 0.2952 | k15_recall_owned_silk 0.9286 | k15_purity_best_silk 0.99 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.7926 | k15_purity_best_inclusion 0.8864 | k15_owned_rolled 1 | k15_best_rolled 0.8824 | k15_recall_owned_rolled 0.8824 | k15_purity_best_rolled 0.5556 | k15_owned_crease 0 | k15_best_crease 0.0000 | k15_recall_owned_crease 0.0000 | k15_purity_best_crease 0.0000 | k15_owned_waist 2 | k15_best_waist 0.5213 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.88 | k15_inertia 66652.62 | k20_K 20 | k20_ami 0.4647 | k20_size_cv 0.6525 | k20_purity_mean 0.8311 | k20_old_in_novel_owned 19 | k20_bal_acc_hungarian 0.3521 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.9357 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2741 | k20_recall_owned_inclusion 0.8593 | k20_purity_best_inclusion 0.90 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.8125 | k20_recall_owned_crease 0.8125 | k20_purity_best_crease 0.45 | k20_owned_waist 4 | k20_best_waist 0.3085 | k20_recall_owned_waist 0.8191 | k20_purity_best_waist 0.97 | k20_inertia 59384.49 | k30_K 30 | k30_ami 0.4542 | k30_size_cv 0.6044 | k30_purity_mean 0.8886 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.3670 | k30_owned_silk 12 | k30_best_silk 0.1405 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 0.97 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.8296 | k30_purity_best_inclusion 0.87 | k30_owned_rolled 1 | k30_best_rolled 0.8235 | k30_recall_owned_rolled 0.8235 | k30_purity_best_rolled 0.7778 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.7500 | k30_purity_best_crease 0.68 | k30_owned_waist 7 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 52359.21 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152901__gc10_pl01b_r1_s5
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 5 | I2_ok 1 | k5_K 5 | k5_ami 0.3944 | k5_size_cv 0.5252 | k5_purity_mean 0.6752 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5088 | k5_owned_silk 3 | k5_best_silk 0.4167 | k5_recall_owned_silk 0.96 | k5_purity_best_silk 0.6119 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.5312 | k5_recall_owned_crease 0.5312 | k5_purity_best_crease 0.2024 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.7881 | k5_inertia 89834.42 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4694 | k10_size_cv 0.4787 | k10_purity_mean 0.7694 | k10_old_in_novel_owned 36 | k10_bal_acc_hungarian 0.4688 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.9571 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3333 | k10_recall_owned_inclusion 0.6444 | k10_purity_best_inclusion 0.8824 | k10_owned_rolled 1 | k10_best_rolled 0.8824 | k10_recall_owned_rolled 0.8824 | k10_purity_best_rolled 0.2727 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.4894 | k10_recall_owned_waist 0.9574 | k10_purity_best_waist 0.9200 | k10_inertia 74842.12 | k15_K 15 | k15_ami 0.4251 | k15_size_cv 0.4738 | k15_purity_mean 0.7844 | k15_old_in_novel_owned 37 | k15_bal_acc_hungarian 0.3645 | k15_owned_silk 7 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.6519 | k15_purity_best_inclusion 0.6491 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.3750 | k15_recall_owned_crease 0.3750 | k15_purity_best_crease 0.3158 | k15_owned_waist 3 | k15_best_waist 0.4894 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.84 | k15_inertia 65004.39 | k20_K 20 | k20_ami 0.4784 | k20_size_cv 0.6115 | k20_purity_mean 0.8484 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.4379 | k20_owned_silk 9 | k20_best_silk 0.2500 | k20_recall_owned_silk 1.00 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2593 | k20_recall_owned_inclusion 0.7852 | k20_purity_best_inclusion 0.80 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.3171 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.79 | k20_owned_waist 3 | k20_best_waist 0.4468 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.98 | k20_inertia 58963.38 | k30_K 30 | k30_ami 0.4509 | k30_size_cv 0.6665 | k30_purity_mean 0.8518 | k30_old_in_novel_owned 15 | k30_bal_acc_hungarian 0.2894 | k30_owned_silk 13 | k30_best_silk 0.1833 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2593 | k30_recall_owned_inclusion 0.7630 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 1 | k30_best_rolled 0.3529 | k30_recall_owned_rolled 0.3529 | k30_purity_best_rolled 1.00 | k30_owned_crease 2 | k30_best_crease 0.3750 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.80 | k30_owned_waist 6 | k30_best_waist 0.2766 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.93 | k30_inertia 52462.79 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_152955__gc10_pl01b_r1_s6
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 6 | I2_ok 1 | k5_K 5 | k5_ami 0.3872 | k5_size_cv 0.5616 | k5_purity_mean 0.6801 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5133 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.96 | k5_purity_best_silk 0.5946 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.5312 | k5_recall_owned_crease 0.5312 | k5_purity_best_crease 0.2099 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.7949 | k5_inertia 89304.12 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4467 | k10_size_cv 0.4023 | k10_purity_mean 0.7321 | k10_old_in_novel_owned 31 | k10_bal_acc_hungarian 0.3734 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.9357 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.2741 | k10_recall_owned_inclusion 0.6889 | k10_purity_best_inclusion 0.7400 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.5532 | k10_recall_owned_waist 0.9787 | k10_purity_best_waist 0.8254 | k10_inertia 73716.32 | k15_K 15 | k15_ami 0.4562 | k15_size_cv 0.5903 | k15_purity_mean 0.7895 | k15_old_in_novel_owned 22 | k15_bal_acc_hungarian 0.3982 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.6741 | k15_purity_best_inclusion 0.8409 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5625 | k15_recall_owned_crease 0.5625 | k15_purity_best_crease 0.4865 | k15_owned_waist 3 | k15_best_waist 0.4362 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.98 | k15_inertia 64547.31 | k20_K 20 | k20_ami 0.4533 | k20_size_cv 0.6513 | k20_purity_mean 0.8074 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.3998 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.99 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2741 | k20_recall_owned_inclusion 0.6148 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.4074 | k20_owned_crease 2 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.7188 | k20_purity_best_crease 0.71 | k20_owned_waist 3 | k20_best_waist 0.3617 | k20_recall_owned_waist 0.8085 | k20_purity_best_waist 0.94 | k20_inertia 58864.41 | k30_K 30 | k30_ami 0.4396 | k30_size_cv 0.7935 | k30_purity_mean 0.8467 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.3171 | k30_owned_silk 12 | k30_best_silk 0.2500 | k30_recall_owned_silk 0.99 | k30_purity_best_silk 1.00 | k30_owned_inclusion 7 | k30_best_inclusion 0.2593 | k30_recall_owned_inclusion 0.8296 | k30_purity_best_inclusion 0.81 | k30_owned_rolled 1 | k30_best_rolled 0.3529 | k30_recall_owned_rolled 0.3529 | k30_purity_best_rolled 0.50 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.84 | k30_owned_waist 6 | k30_best_waist 0.2234 | k30_recall_owned_waist 0.9149 | k30_purity_best_waist 1.00 | k30_inertia 52104.28 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153050__gc10_pl01b_r1_s7
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 7 | I2_ok 1 | k5_K 5 | k5_ami 0.3924 | k5_size_cv 0.4928 | k5_purity_mean 0.6766 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4975 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6331 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.5312 | k5_recall_owned_crease 0.5312 | k5_purity_best_crease 0.1848 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8158 | k5_inertia 89703.62 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4304 | k10_size_cv 0.4326 | k10_purity_mean 0.7678 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4180 | k10_owned_silk 5 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3407 | k10_recall_owned_inclusion 0.5185 | k10_purity_best_inclusion 0.6667 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2979 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8281 | k10_inertia 74666.70 | k15_K 15 | k15_ami 0.4408 | k15_size_cv 0.5326 | k15_purity_mean 0.8037 | k15_old_in_novel_owned 29 | k15_bal_acc_hungarian 0.4385 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9738 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.6815 | k15_purity_best_inclusion 0.9286 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.7812 | k15_recall_owned_crease 0.7812 | k15_purity_best_crease 0.5814 | k15_owned_waist 3 | k15_best_waist 0.4043 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 0.95 | k15_inertia 65624.04 | k20_K 20 | k20_ami 0.4624 | k20_size_cv 0.6607 | k20_purity_mean 0.8200 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.4067 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.98 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2815 | k20_recall_owned_inclusion 0.6222 | k20_purity_best_inclusion 0.95 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.4615 | k20_owned_crease 2 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.7500 | k20_purity_best_crease 0.75 | k20_owned_waist 4 | k20_best_waist 0.3298 | k20_recall_owned_waist 0.9149 | k20_purity_best_waist 0.94 | k20_inertia 59483.44 | k30_K 30 | k30_ami 0.4419 | k30_size_cv 0.7927 | k30_purity_mean 0.8532 | k30_old_in_novel_owned 17 | k30_bal_acc_hungarian 0.3772 | k30_owned_silk 13 | k30_best_silk 0.2500 | k30_recall_owned_silk 0.99 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2593 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 1.00 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.40 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.6562 | k30_purity_best_crease 0.74 | k30_owned_waist 5 | k30_best_waist 0.3511 | k30_recall_owned_waist 0.9149 | k30_purity_best_waist 0.94 | k30_inertia 52383.09 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153143__gc10_pl01b_r1_s8
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 8 | I2_ok 1 | k5_K 5 | k5_ami 0.3955 | k5_size_cv 0.5032 | k5_purity_mean 0.6921 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5230 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6263 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2211 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8611 | k5_inertia 89335.09 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4606 | k10_size_cv 0.4363 | k10_purity_mean 0.7588 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3916 | k10_owned_silk 4 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.9405 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.3111 | k10_recall_owned_inclusion 0.7185 | k10_purity_best_inclusion 0.7241 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5938 | k10_recall_owned_crease 0.5938 | k10_purity_best_crease 0.3276 | k10_owned_waist 2 | k10_best_waist 0.7553 | k10_recall_owned_waist 0.9468 | k10_purity_best_waist 0.9467 | k10_inertia 73996.64 | k15_K 15 | k15_ami 0.4338 | k15_size_cv 0.5101 | k15_purity_mean 0.8162 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4599 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8952 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.3185 | k15_recall_owned_inclusion 0.9111 | k15_purity_best_inclusion 0.4886 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.4688 | k15_purity_best_crease 0.3488 | k15_owned_waist 3 | k15_best_waist 0.5000 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.89 | k15_inertia 65385.09 | k20_K 20 | k20_ami 0.4534 | k20_size_cv 0.6383 | k20_purity_mean 0.8184 | k20_old_in_novel_owned 37 | k20_bal_acc_hungarian 0.3962 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.93 | k20_purity_best_silk 1.00 | k20_owned_inclusion 5 | k20_best_inclusion 0.2815 | k20_recall_owned_inclusion 0.9037 | k20_purity_best_inclusion 0.90 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.3636 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.48 | k20_owned_waist 4 | k20_best_waist 0.3085 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.91 | k20_inertia 59039.02 | k30_K 30 | k30_ami 0.4532 | k30_size_cv 0.8966 | k30_purity_mean 0.8418 | k30_old_in_novel_owned 17 | k30_bal_acc_hungarian 0.2878 | k30_owned_silk 12 | k30_best_silk 0.2500 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 4 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.6963 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.89 | k30_owned_crease 4 | k30_best_crease 0.2188 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.70 | k30_owned_waist 6 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.96 | k30_inertia 52014.73 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153238__gc10_pl01b_r1_s9
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 1); pre-registered (commit 885e2dc)
result: seed 9 | I2_ok 1 | k5_K 5 | k5_ami 0.4005 | k5_size_cv 0.5059 | k5_purity_mean 0.6954 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5230 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6263 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2211 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.8942 | k5_inertia 89461.79 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4381 | k10_size_cv 0.4151 | k10_purity_mean 0.7715 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4121 | k10_owned_silk 5 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3778 | k10_recall_owned_inclusion 0.5481 | k10_purity_best_inclusion 0.6892 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2857 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8413 | k10_inertia 74433.98 | k15_K 15 | k15_ami 0.4643 | k15_size_cv 0.5558 | k15_purity_mean 0.7960 | k15_old_in_novel_owned 18 | k15_bal_acc_hungarian 0.4033 | k15_owned_silk 6 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.9548 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3185 | k15_recall_owned_inclusion 0.7037 | k15_purity_best_inclusion 0.9149 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5625 | k15_recall_owned_crease 0.5625 | k15_purity_best_crease 0.6923 | k15_owned_waist 3 | k15_best_waist 0.4149 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.95 | k15_inertia 65260.42 | k20_K 20 | k20_ami 0.4562 | k20_size_cv 0.6554 | k20_purity_mean 0.8049 | k20_old_in_novel_owned 39 | k20_bal_acc_hungarian 0.4236 | k20_owned_silk 9 | k20_best_silk 0.2500 | k20_recall_owned_silk 0.97 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2963 | k20_recall_owned_inclusion 0.6741 | k20_purity_best_inclusion 0.91 | k20_owned_rolled 1 | k20_best_rolled 0.8235 | k20_recall_owned_rolled 0.8235 | k20_purity_best_rolled 0.4828 | k20_owned_crease 2 | k20_best_crease 0.3438 | k20_recall_owned_crease 0.6875 | k20_purity_best_crease 0.48 | k20_owned_waist 3 | k20_best_waist 0.4043 | k20_recall_owned_waist 0.8617 | k20_purity_best_waist 0.95 | k20_inertia 59491.36 | k30_K 30 | k30_ami 0.4423 | k30_size_cv 0.7376 | k30_purity_mean 0.8351 | k30_old_in_novel_owned 18 | k30_bal_acc_hungarian 0.3534 | k30_owned_silk 12 | k30_best_silk 0.1738 | k30_recall_owned_silk 0.97 | k30_purity_best_silk 0.96 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.7630 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 1 | k30_best_rolled 0.7647 | k30_recall_owned_rolled 0.7647 | k30_purity_best_rolled 0.52 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.6875 | k30_purity_best_crease 0.81 | k30_owned_waist 6 | k30_best_waist 0.1702 | k30_recall_owned_waist 0.8830 | k30_purity_best_waist 0.94 | k30_inertia 52490.08 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153333__gc10_pl01b_r2_s0
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 0 | I2_ok 1 | k5_K 5 | k5_ami 0.4403 | k5_size_cv 0.4040 | k5_purity_mean 0.7020 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4580 | k5_owned_silk 2 | k5_best_silk 0.5405 | k5_recall_owned_silk 0.79 | k5_purity_best_silk 0.9660 | k5_owned_inclusion 2 | k5_best_inclusion 0.6074 | k5_recall_owned_inclusion 0.9111 | k5_purity_best_inclusion 0.4881 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.5886 | k5_inertia 92754.14 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4394 | k10_size_cv 0.3881 | k10_purity_mean 0.7804 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3854 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5926 | k10_purity_best_inclusion 0.6557 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2857 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7838 | k10_inertia 76462.56 | k15_K 15 | k15_ami 0.4423 | k15_size_cv 0.5495 | k15_purity_mean 0.8142 | k15_old_in_novel_owned 36 | k15_bal_acc_hungarian 0.4096 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8833 | k15_purity_best_silk 0.98 | k15_owned_inclusion 3 | k15_best_inclusion 0.3852 | k15_recall_owned_inclusion 0.8519 | k15_purity_best_inclusion 0.5000 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.7812 | k15_recall_owned_crease 0.7812 | k15_purity_best_crease 0.4808 | k15_owned_waist 3 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 1.00 | k15_inertia 67977.64 | k20_K 20 | k20_ami 0.4520 | k20_size_cv 0.6528 | k20_purity_mean 0.8107 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.3714 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.89 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.8667 | k20_purity_best_inclusion 0.55 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.4583 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.64 | k20_owned_waist 5 | k20_best_waist 0.2234 | k20_recall_owned_waist 0.9362 | k20_purity_best_waist 0.95 | k20_inertia 62269.23 | k30_K 30 | k30_ami 0.4297 | k30_size_cv 0.6204 | k30_purity_mean 0.8426 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.3824 | k30_owned_silk 14 | k30_best_silk 0.2000 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2296 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.86 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.67 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7500 | k30_purity_best_crease 0.70 | k30_owned_waist 4 | k30_best_waist 0.2766 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.96 | k30_inertia 54717.25 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153428__gc10_pl01b_r2_s1
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 1 | I2_ok 1 | k5_K 5 | k5_ami 0.4078 | k5_size_cv 0.4997 | k5_purity_mean 0.7390 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3496 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6473 | k5_owned_inclusion 1 | k5_best_inclusion 0.3037 | k5_recall_owned_inclusion 0.3037 | k5_purity_best_inclusion 0.4767 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.5962 | k5_inertia 92578.15 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4434 | k10_size_cv 0.4028 | k10_purity_mean 0.7872 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3914 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3037 | k10_recall_owned_inclusion 0.6000 | k10_purity_best_inclusion 0.5942 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2979 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8056 | k10_inertia 76947.14 | k15_K 15 | k15_ami 0.4464 | k15_size_cv 0.4018 | k15_purity_mean 0.8147 | k15_old_in_novel_owned 36 | k15_bal_acc_hungarian 0.3180 | k15_owned_silk 8 | k15_best_silk 0.2024 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.7556 | k15_purity_best_inclusion 0.7551 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4375 | k15_recall_owned_crease 0.4375 | k15_purity_best_crease 0.3500 | k15_owned_waist 2 | k15_best_waist 0.6170 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.79 | k15_inertia 68395.61 | k20_K 20 | k20_ami 0.4443 | k20_size_cv 0.5823 | k20_purity_mean 0.8351 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3475 | k20_owned_silk 9 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.95 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2667 | k20_recall_owned_inclusion 0.7407 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.5000 | k20_recall_owned_crease 0.5000 | k20_purity_best_crease 0.46 | k20_owned_waist 5 | k20_best_waist 0.2660 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 0.93 | k20_inertia 62545.11 | k30_K 30 | k30_ami 0.4357 | k30_size_cv 0.7439 | k30_purity_mean 0.8673 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.3177 | k30_owned_silk 14 | k30_best_silk 0.2024 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.8296 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.00 | k30_owned_crease 2 | k30_best_crease 0.5312 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.74 | k30_owned_waist 6 | k30_best_waist 0.1915 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 1.00 | k30_inertia 55272.96 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153522__gc10_pl01b_r2_s2
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 2 | I2_ok 1 | k5_K 5 | k5_ami 0.4238 | k5_size_cv 0.3988 | k5_purity_mean 0.7763 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3881 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.7149 | k5_owned_inclusion 1 | k5_best_inclusion 0.4963 | k5_recall_owned_inclusion 0.4963 | k5_purity_best_inclusion 0.6321 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.9894 | k5_recall_owned_waist 0.9894 | k5_purity_best_waist 0.5849 | k5_inertia 92739.23 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4431 | k10_size_cv 0.4191 | k10_purity_mean 0.7863 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3826 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5926 | k10_purity_best_inclusion 0.7018 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.3111 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7733 | k10_inertia 76585.34 | k15_K 15 | k15_ami 0.4466 | k15_size_cv 0.3516 | k15_purity_mean 0.8124 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4136 | k15_owned_silk 8 | k15_best_silk 0.2024 | k15_recall_owned_silk 0.9619 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2815 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.8085 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 2 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.3261 | k15_owned_waist 2 | k15_best_waist 0.4681 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.98 | k15_inertia 68539.39 | k20_K 20 | k20_ami 0.4354 | k20_size_cv 0.6105 | k20_purity_mean 0.8021 | k20_old_in_novel_owned 28 | k20_bal_acc_hungarian 0.3575 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.90 | k20_purity_best_silk 0.99 | k20_owned_inclusion 5 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.8593 | k20_purity_best_inclusion 0.79 | k20_owned_rolled 1 | k20_best_rolled 0.4706 | k20_recall_owned_rolled 0.4706 | k20_purity_best_rolled 0.3333 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.58 | k20_owned_waist 4 | k20_best_waist 0.3511 | k20_recall_owned_waist 0.9149 | k20_purity_best_waist 0.97 | k20_inertia 62356.21 | k30_K 30 | k30_ami 0.4611 | k30_size_cv 0.6968 | k30_purity_mean 0.8989 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3950 | k30_owned_silk 13 | k30_best_silk 0.2000 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.8222 | k30_purity_best_inclusion 0.82 | k30_owned_rolled 1 | k30_best_rolled 0.7647 | k30_recall_owned_rolled 0.7647 | k30_purity_best_rolled 0.72 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.94 | k30_owned_waist 6 | k30_best_waist 0.2660 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.96 | k30_inertia 54609.66 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153617__gc10_pl01b_r2_s3
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 3 | I2_ok 1 | k5_K 5 | k5_ami 0.3786 | k5_size_cv 0.2436 | k5_purity_mean 0.7347 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4109 | k5_owned_silk 3 | k5_best_silk 0.3310 | k5_recall_owned_silk 0.82 | k5_purity_best_silk 0.7596 | k5_owned_inclusion 1 | k5_best_inclusion 0.6296 | k5_recall_owned_inclusion 0.6296 | k5_purity_best_inclusion 0.4802 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 1.00 | k5_recall_owned_waist 1.00 | k5_purity_best_waist 0.5629 | k5_inertia 92630.83 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4645 | k10_size_cv 0.4416 | k10_purity_mean 0.7824 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4165 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.8476 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.4074 | k10_recall_owned_inclusion 0.9704 | k10_purity_best_inclusion 0.4867 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3333 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7838 | k10_inertia 76431.62 | k15_K 15 | k15_ami 0.4664 | k15_size_cv 0.5476 | k15_purity_mean 0.8049 | k15_old_in_novel_owned 23 | k15_bal_acc_hungarian 0.3925 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9095 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.8593 | k15_purity_best_inclusion 0.5735 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.7083 | k15_owned_waist 2 | k15_best_waist 0.6596 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.79 | k15_inertia 68098.40 | k20_K 20 | k20_ami 0.4493 | k20_size_cv 0.5345 | k20_purity_mean 0.8121 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4288 | k20_owned_silk 9 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.98 | k20_purity_best_silk 0.98 | k20_owned_inclusion 4 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.7333 | k20_purity_best_inclusion 0.95 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.5000 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.62 | k20_owned_waist 3 | k20_best_waist 0.4043 | k20_recall_owned_waist 0.8936 | k20_purity_best_waist 0.90 | k20_inertia 61994.48 | k30_K 30 | k30_ami 0.4306 | k30_size_cv 0.6533 | k30_purity_mean 0.8301 | k30_old_in_novel_owned 23 | k30_bal_acc_hungarian 0.3979 | k30_owned_silk 13 | k30_best_silk 0.2000 | k30_recall_owned_silk 0.97 | k30_purity_best_silk 1.00 | k30_owned_inclusion 7 | k30_best_inclusion 0.1852 | k30_recall_owned_inclusion 0.7333 | k30_purity_best_inclusion 0.86 | k30_owned_rolled 1 | k30_best_rolled 0.7647 | k30_recall_owned_rolled 0.7647 | k30_purity_best_rolled 0.65 | k30_owned_crease 1 | k30_best_crease 0.5312 | k30_recall_owned_crease 0.5312 | k30_purity_best_crease 0.89 | k30_owned_waist 6 | k30_best_waist 0.3085 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.91 | k30_inertia 54626.82 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153711__gc10_pl01b_r2_s4
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 4 | I2_ok 1 | k5_K 5 | k5_ami 0.4275 | k5_size_cv 0.4028 | k5_purity_mean 0.7776 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3881 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.7120 | k5_owned_inclusion 1 | k5_best_inclusion 0.4963 | k5_recall_owned_inclusion 0.4963 | k5_purity_best_inclusion 0.6321 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5849 | k5_inertia 92764.02 | I1_internal_ok 1 | I1_has_ref 1 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4579 | k10_size_cv 0.3704 | k10_purity_mean 0.7877 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4747 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.8524 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.4074 | k10_recall_owned_inclusion 0.9704 | k10_purity_best_inclusion 0.5046 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2917 | k10_owned_waist 1 | k10_best_waist 0.9894 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8692 | k10_inertia 76921.80 | k15_K 15 | k15_ami 0.4396 | k15_size_cv 0.4845 | k15_purity_mean 0.8256 | k15_old_in_novel_owned 35 | k15_bal_acc_hungarian 0.3430 | k15_owned_silk 6 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.3630 | k15_recall_owned_inclusion 0.8889 | k15_purity_best_inclusion 0.7424 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.5000 | k15_owned_waist 4 | k15_best_waist 0.3404 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.86 | k15_inertia 69094.20 | k20_K 20 | k20_ami 0.4508 | k20_size_cv 0.5242 | k20_purity_mean 0.8147 | k20_old_in_novel_owned 19 | k20_bal_acc_hungarian 0.2983 | k20_owned_silk 9 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.98 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2444 | k20_recall_owned_inclusion 0.7481 | k20_purity_best_inclusion 0.94 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.62 | k20_owned_waist 4 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 1.00 | k20_inertia 63233.57 | k30_K 30 | k30_ami 0.4337 | k30_size_cv 0.7542 | k30_purity_mean 0.8464 | k30_old_in_novel_owned 12 | k30_bal_acc_hungarian 0.3519 | k30_owned_silk 11 | k30_best_silk 0.2024 | k30_recall_owned_silk 0.94 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.8148 | k30_purity_best_inclusion 0.77 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.71 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.5000 | k30_purity_best_crease 0.82 | k30_owned_waist 6 | k30_best_waist 0.1915 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 0.78 | k30_inertia 55369.11 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153806__gc10_pl01b_r2_s5
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 5 | I2_ok 1 | k5_K 5 | k5_ami 0.3846 | k5_size_cv 0.2554 | k5_purity_mean 0.7324 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4064 | k5_owned_silk 3 | k5_best_silk 0.3310 | k5_recall_owned_silk 0.82 | k5_purity_best_silk 0.7316 | k5_owned_inclusion 1 | k5_best_inclusion 0.6074 | k5_recall_owned_inclusion 0.6074 | k5_purity_best_inclusion 0.4881 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 1.00 | k5_recall_owned_waist 1.00 | k5_purity_best_waist 0.5465 | k5_inertia 93277.00 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4502 | k10_size_cv 0.3579 | k10_purity_mean 0.7808 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4742 | k10_owned_silk 6 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9524 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3778 | k10_recall_owned_inclusion 0.6741 | k10_purity_best_inclusion 0.6296 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2642 | k10_owned_waist 1 | k10_best_waist 0.9574 | k10_recall_owned_waist 0.9574 | k10_purity_best_waist 0.9184 | k10_inertia 77263.58 | k15_K 15 | k15_ami 0.4545 | k15_size_cv 0.4322 | k15_purity_mean 0.7907 | k15_old_in_novel_owned 38 | k15_bal_acc_hungarian 0.4076 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2593 | k15_recall_owned_inclusion 0.7333 | k15_purity_best_inclusion 0.6364 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 2 | k15_best_crease 0.5000 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.3721 | k15_owned_waist 2 | k15_best_waist 0.5426 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 1.00 | k15_inertia 67942.22 | k20_K 20 | k20_ami 0.4477 | k20_size_cv 0.6204 | k20_purity_mean 0.8105 | k20_old_in_novel_owned 19 | k20_bal_acc_hungarian 0.4192 | k20_owned_silk 9 | k20_best_silk 0.2000 | k20_recall_owned_silk 0.99 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2667 | k20_recall_owned_inclusion 0.5481 | k20_purity_best_inclusion 0.86 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.3429 | k20_owned_crease 1 | k20_best_crease 0.5000 | k20_recall_owned_crease 0.5000 | k20_purity_best_crease 0.84 | k20_owned_waist 3 | k20_best_waist 0.3936 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.97 | k20_inertia 62372.70 | k30_K 30 | k30_ami 0.4253 | k30_size_cv 0.5254 | k30_purity_mean 0.8533 | k30_old_in_novel_owned 14 | k30_bal_acc_hungarian 0.3207 | k30_owned_silk 16 | k30_best_silk 0.1357 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 0.97 | k30_owned_inclusion 5 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.8074 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.00 | k30_owned_crease 1 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.4688 | k30_purity_best_crease 0.71 | k30_owned_waist 5 | k30_best_waist 0.2766 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.96 | k30_inertia 55180.83 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153900__gc10_pl01b_r2_s6
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 6 | I2_ok 1 | k5_K 5 | k5_ami 0.3821 | k5_size_cv 0.2534 | k5_purity_mean 0.7320 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4064 | k5_owned_silk 3 | k5_best_silk 0.3310 | k5_recall_owned_silk 0.82 | k5_purity_best_silk 0.7316 | k5_owned_inclusion 1 | k5_best_inclusion 0.6074 | k5_recall_owned_inclusion 0.6074 | k5_purity_best_inclusion 0.4740 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 1.00 | k5_recall_owned_waist 1.00 | k5_purity_best_waist 0.5663 | k5_inertia 92667.59 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4506 | k10_size_cv 0.4264 | k10_purity_mean 0.7892 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3958 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5852 | k10_purity_best_inclusion 0.6452 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2917 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7632 | k10_inertia 76567.80 | k15_K 15 | k15_ami 0.4644 | k15_size_cv 0.4822 | k15_purity_mean 0.8137 | k15_old_in_novel_owned 23 | k15_bal_acc_hungarian 0.3879 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9452 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2519 | k15_recall_owned_inclusion 0.7481 | k15_purity_best_inclusion 0.6182 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.6800 | k15_owned_waist 2 | k15_best_waist 0.6170 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.78 | k15_inertia 67146.62 | k20_K 20 | k20_ami 0.4536 | k20_size_cv 0.5342 | k20_purity_mean 0.8163 | k20_old_in_novel_owned 16 | k20_bal_acc_hungarian 0.4602 | k20_owned_silk 8 | k20_best_silk 0.2000 | k20_recall_owned_silk 0.91 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2296 | k20_recall_owned_inclusion 0.8074 | k20_purity_best_inclusion 0.63 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.7059 | k20_owned_crease 1 | k20_best_crease 0.7188 | k20_recall_owned_crease 0.7188 | k20_purity_best_crease 0.68 | k20_owned_waist 3 | k20_best_waist 0.4468 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.93 | k20_inertia 62411.95 | k30_K 30 | k30_ami 0.4337 | k30_size_cv 0.5841 | k30_purity_mean 0.8776 | k30_old_in_novel_owned 23 | k30_bal_acc_hungarian 0.3437 | k30_owned_silk 13 | k30_best_silk 0.1238 | k30_recall_owned_silk 0.93 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.9407 | k30_purity_best_inclusion 0.74 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.71 | k30_owned_crease 2 | k30_best_crease 0.5312 | k30_recall_owned_crease 0.6875 | k30_purity_best_crease 0.89 | k30_owned_waist 7 | k30_best_waist 0.2234 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.91 | k30_inertia 55221.14 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_153955__gc10_pl01b_r2_s7
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 7 | I2_ok 1 | k5_K 5 | k5_ami 0.3810 | k5_size_cv 0.2490 | k5_purity_mean 0.7362 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4109 | k5_owned_silk 3 | k5_best_silk 0.3310 | k5_recall_owned_silk 0.82 | k5_purity_best_silk 0.7554 | k5_owned_inclusion 1 | k5_best_inclusion 0.6296 | k5_recall_owned_inclusion 0.6296 | k5_purity_best_inclusion 0.4802 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 1.00 | k5_recall_owned_waist 1.00 | k5_purity_best_waist 0.5595 | k5_inertia 92979.04 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4338 | k10_size_cv 0.3872 | k10_purity_mean 0.7783 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3840 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5926 | k10_purity_best_inclusion 0.6780 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2979 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7945 | k10_inertia 76871.25 | k15_K 15 | k15_ami 0.4514 | k15_size_cv 0.4631 | k15_purity_mean 0.7952 | k15_old_in_novel_owned 28 | k15_bal_acc_hungarian 0.4316 | k15_owned_silk 8 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9286 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3630 | k15_recall_owned_inclusion 0.8148 | k15_purity_best_inclusion 0.6049 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.7188 | k15_recall_owned_crease 0.7188 | k15_purity_best_crease 0.6216 | k15_owned_waist 2 | k15_best_waist 0.5957 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.93 | k15_inertia 69006.50 | k20_K 20 | k20_ami 0.4422 | k20_size_cv 0.6083 | k20_purity_mean 0.8225 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.4030 | k20_owned_silk 9 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.91 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.3333 | k20_recall_owned_inclusion 0.9037 | k20_purity_best_inclusion 0.58 | k20_owned_rolled 1 | k20_best_rolled 0.4706 | k20_recall_owned_rolled 0.4706 | k20_purity_best_rolled 0.2963 | k20_owned_crease 1 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.5938 | k20_purity_best_crease 0.76 | k20_owned_waist 3 | k20_best_waist 0.4149 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.97 | k20_inertia 62696.21 | k30_K 30 | k30_ami 0.4425 | k30_size_cv 0.7234 | k30_purity_mean 0.8671 | k30_old_in_novel_owned 15 | k30_bal_acc_hungarian 0.3421 | k30_owned_silk 14 | k30_best_silk 0.2048 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2148 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.00 | k30_owned_crease 1 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.4688 | k30_purity_best_crease 0.71 | k30_owned_waist 5 | k30_best_waist 0.2340 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.92 | k30_inertia 55275.64 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_154050__gc10_pl01b_r2_s8
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 8 | I2_ok 1 | k5_K 5 | k5_ami 0.4000 | k5_size_cv 0.5133 | k5_purity_mean 0.7406 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3491 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.96 | k5_purity_best_silk 0.6413 | k5_owned_inclusion 1 | k5_best_inclusion 0.3037 | k5_recall_owned_inclusion 0.3037 | k5_purity_best_inclusion 0.5190 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5776 | k5_inertia 92436.12 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4059 | k10_size_cv 0.5217 | k10_purity_mean 0.7939 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4027 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.7810 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.5926 | k10_recall_owned_inclusion 0.8889 | k10_purity_best_inclusion 0.4734 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5000 | k10_recall_owned_crease 0.5000 | k10_purity_best_crease 0.2712 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7838 | k10_inertia 77028.89 | k15_K 15 | k15_ami 0.4329 | k15_size_cv 0.4590 | k15_purity_mean 0.8088 | k15_old_in_novel_owned 38 | k15_bal_acc_hungarian 0.3266 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9500 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2519 | k15_recall_owned_inclusion 0.7333 | k15_purity_best_inclusion 0.7234 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3095 | k15_owned_waist 3 | k15_best_waist 0.5532 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.81 | k15_inertia 67415.09 | k20_K 20 | k20_ami 0.4244 | k20_size_cv 0.5092 | k20_purity_mean 0.8421 | k20_old_in_novel_owned 23 | k20_bal_acc_hungarian 0.3603 | k20_owned_silk 10 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.94 | k20_purity_best_silk 0.98 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8519 | k20_purity_best_inclusion 0.65 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.7188 | k20_recall_owned_crease 0.7188 | k20_purity_best_crease 0.50 | k20_owned_waist 3 | k20_best_waist 0.4149 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.97 | k20_inertia 61973.13 | k30_K 30 | k30_ami 0.4652 | k30_size_cv 0.8881 | k30_purity_mean 0.8385 | k30_old_in_novel_owned 8 | k30_bal_acc_hungarian 0.3725 | k30_owned_silk 10 | k30_best_silk 0.2476 | k30_recall_owned_silk 0.97 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.7778 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.79 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.6562 | k30_purity_best_crease 0.84 | k30_owned_waist 6 | k30_best_waist 0.2234 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 0.95 | k30_inertia 55011.21 | checks_ok 1
note: see the gc10_pl01b_r2_s9 note

## 20261007_154144__gc10_pl01b_r2_s9
why: E09.05: same as E09.04 on the paper setting (fine-tuned features, FT seed 2); pre-registered (commit 885e2dc)
result: seed 9 | I2_ok 1 | k5_K 5 | k5_ami 0.3939 | k5_size_cv 0.5038 | k5_purity_mean 0.7348 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3491 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.95 | k5_purity_best_silk 0.6390 | k5_owned_inclusion 1 | k5_best_inclusion 0.3037 | k5_recall_owned_inclusion 0.3037 | k5_purity_best_inclusion 0.4607 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.6000 | k5_inertia 92861.46 | I1_internal_ok 1 | I1_has_ref 0 | I1_ok 1 | I3_ok 1 | k10_K 10 | k10_ami 0.4260 | k10_size_cv 0.5663 | k10_purity_mean 0.8007 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3682 | k10_owned_silk 5 | k10_best_silk 0.3262 | k10_recall_owned_silk 0.9548 | k10_purity_best_silk 0.76 | k10_owned_inclusion 2 | k10_best_inclusion 0.3704 | k10_recall_owned_inclusion 0.6593 | k10_purity_best_inclusion 0.7246 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3000 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8056 | k10_inertia 77176.24 | k15_K 15 | k15_ami 0.4283 | k15_size_cv 0.4870 | k15_purity_mean 0.8057 | k15_old_in_novel_owned 38 | k15_bal_acc_hungarian 0.3560 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9071 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3185 | k15_recall_owned_inclusion 0.8593 | k15_purity_best_inclusion 0.6232 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3333 | k15_owned_waist 3 | k15_best_waist 0.5745 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.87 | k15_inertia 67751.86 | k20_K 20 | k20_ami 0.4353 | k20_size_cv 0.4983 | k20_purity_mean 0.8266 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4652 | k20_owned_silk 9 | k20_best_silk 0.2000 | k20_recall_owned_silk 0.94 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.7630 | k20_purity_best_inclusion 0.77 | k20_owned_rolled 1 | k20_best_rolled 0.8235 | k20_recall_owned_rolled 0.8235 | k20_purity_best_rolled 0.5185 | k20_owned_crease 1 | k20_best_crease 0.6250 | k20_recall_owned_crease 0.6250 | k20_purity_best_crease 0.62 | k20_owned_waist 3 | k20_best_waist 0.4255 | k20_recall_owned_waist 0.8830 | k20_purity_best_waist 0.95 | k20_inertia 62195.71 | k30_K 30 | k30_ami 0.4309 | k30_size_cv 0.6823 | k30_purity_mean 0.8467 | k30_old_in_novel_owned 17 | k30_bal_acc_hungarian 0.3649 | k30_owned_silk 14 | k30_best_silk 0.2000 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2370 | k30_recall_owned_inclusion 0.7630 | k30_purity_best_inclusion 0.91 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.75 | k30_owned_crease 1 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.4688 | k30_purity_best_crease 0.94 | k30_owned_waist 6 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 0.95 | k30_inertia 54267.00 | checks_ok 1
note: E09.05 verdict LIVE at K'=20: rolled_pit 6/8/5, crease 8/1/6 seeds (r0/r1/r2), each class passes 2/3 FT seeds; C1-C3 pass 30/30. Details: notes/pl/experiments.vi.md

## 20261020_024405__gc10_v2_ft_r0_s0
why: QD21/C1-C3: default chain of lab/replay.py v2, gc10 ft r0, seed 0; C1 against gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s0; check outputs in runs/20261020_024405__gc10_v2_ft_r0_s0/check_c*.txt
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: C1 OK; C2 OK; C3 OK

## 20261020_083928__gc10_v2_ft_r0_s1
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r0, seed 1; C1 against gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s1; output in runs/20261020_083928__gc10_v2_ft_r0_s1/check_c1.txt
result: S0_All 85.48 | S1_New 57.48 | final_All 69.89 | M_d 57.48 | M_f 4.56
note: C1 OK

## 20261020_084204__gc10_v2_ft_r0_s2
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r0, seed 2; C1 against gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s2; output in runs/20261020_084204__gc10_v2_ft_r0_s2/check_c1.txt
result: S0_All 84.65 | S1_New 49.07 | final_All 66.37 | M_d 49.07 | M_f 2.90
note: C1 OK

## 20261020_084432__gc10_v2_ft_r0_s3
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r0, seed 3; C1 against gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s3; output in runs/20261020_084432__gc10_v2_ft_r0_s3/check_c1.txt
result: S0_All 85.06 | S1_New 56.54 | final_All 69.67 | M_d 56.54 | M_f 3.73
note: C1 OK

## 20261020_084658__gc10_v2_ft_r0_s4
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r0, seed 4; C1 against gc10_ft_cls5first_img80pct_r0_km_evhung_d64_s4; output in runs/20261020_084658__gc10_v2_ft_r0_s4/check_c1.txt
result: S0_All 85.48 | S1_New 57.01 | final_All 70.11 | M_d 57.01 | M_f 3.73
note: C1 OK

## 20261020_084925__gc10_v2_ft_r1_s0
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r1, seed 0; C1 against gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s0; output in runs/20261020_084925__gc10_v2_ft_r1_s0/check_c1.txt
result: S0_All 85.06 | S1_New 39.25 | final_All 62.20 | M_d 39.25 | M_f 2.49
note: C1 OK

## 20261020_085149__gc10_v2_ft_r1_s1
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r1, seed 1; C1 against gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s1; output in runs/20261020_085149__gc10_v2_ft_r1_s1/check_c1.txt
result: S0_All 85.06 | S1_New 39.72 | final_All 62.20 | M_d 39.72 | M_f 2.90
note: C1 OK

## 20261020_085418__gc10_v2_ft_r1_s2
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r1, seed 2; C1 against gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s2; output in runs/20261020_085418__gc10_v2_ft_r1_s2/check_c1.txt
result: S0_All 85.06 | S1_New 40.19 | final_All 62.64 | M_d 40.19 | M_f 2.49
note: C1 OK

## 20261020_085643__gc10_v2_ft_r1_s3
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r1, seed 3; C1 against gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s3; output in runs/20261020_085643__gc10_v2_ft_r1_s3/check_c1.txt
result: S0_All 84.23 | S1_New 41.12 | final_All 63.08 | M_d 41.12 | M_f 1.66
note: C1 OK

## 20261020_085907__gc10_v2_ft_r1_s4
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r1, seed 4; C1 against gc10_ft_cls5first_img80pct_r1_km_evhung_d64_s4; output in runs/20261020_085907__gc10_v2_ft_r1_s4/check_c1.txt
result: S0_All 85.48 | S1_New 36.92 | final_All 61.76 | M_d 36.92 | M_f 1.66
note: C1 OK

## 20261020_090130__gc10_v2_ft_r2_s0
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r2, seed 0; C1 against gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s0; output in runs/20261020_090130__gc10_v2_ft_r2_s0/check_c1.txt
result: S0_All 85.06 | S1_New 48.13 | final_All 64.84 | M_d 48.13 | M_f 5.39
note: C1 OK

## 20261020_090354__gc10_v2_ft_r2_s1
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r2, seed 1; C1 against gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s1; output in runs/20261020_090354__gc10_v2_ft_r2_s1/check_c1.txt
result: S0_All 86.31 | S1_New 56.07 | final_All 68.79 | M_d 56.07 | M_f 6.22
note: C1 OK

## 20261020_090622__gc10_v2_ft_r2_s2
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r2, seed 2; C1 against gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s2; output in runs/20261020_090622__gc10_v2_ft_r2_s2/check_c1.txt
result: S0_All 86.31 | S1_New 60.28 | final_All 71.21 | M_d 60.28 | M_f 5.39
note: C1 OK

## 20261020_090846__gc10_v2_ft_r2_s3
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r2, seed 3; C1 against gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s3; output in runs/20261020_090846__gc10_v2_ft_r2_s3/check_c1.txt
result: S0_All 85.89 | S1_New 46.73 | final_All 65.05 | M_d 46.73 | M_f 4.56
note: C1 OK

## 20261020_091107__gc10_v2_ft_r2_s4
why: QD21/C1: default chain of lab/replay.py v2, gc10 ft r2, seed 4; C1 against gc10_ft_cls5first_img80pct_r2_km_evhung_d64_s4; output in runs/20261020_091107__gc10_v2_ft_r2_s4/check_c1.txt
result: S0_All 85.89 | S1_New 58.88 | final_All 70.77 | M_d 58.88 | M_f 4.56
note: C1 OK

## 20261021_015918__gc10_e09_05_v2_r0_s0
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r0_s0, compared with gc10_pl01b_r0_s0; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 0 | k5_K 5 | k5_ami 0.4435 | k5_size_cv 0.5712 | k5_purity_mean 0.6686 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5750 | k5_owned_silk 2 | k5_best_silk 0.5238 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.9649 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.4831 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3333 | k5_owned_waist 1 | k5_best_waist 0.86 | k5_recall_owned_waist 0.86 | k5_purity_best_waist 0.9000 | k5_inertia 82352.23 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.3980 | k10_size_cv 0.3842 | k10_purity_mean 0.7019 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4200 | k10_owned_silk 4 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.8405 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.2593 | k10_recall_owned_inclusion 0.6593 | k10_purity_best_inclusion 0.4268 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5625 | k10_recall_owned_crease 0.5625 | k10_purity_best_crease 0.3462 | k10_owned_waist 2 | k10_best_waist 0.5106 | k10_recall_owned_waist 0.9787 | k10_purity_best_waist 0.7385 | k10_inertia 68571.34 | k15_K 15 | k15_ami 0.4273 | k15_size_cv 0.4393 | k15_purity_mean 0.7921 | k15_old_in_novel_owned 37 | k15_bal_acc_hungarian 0.3611 | k15_owned_silk 8 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9571 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2519 | k15_recall_owned_inclusion 0.6296 | k15_purity_best_inclusion 0.6939 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5625 | k15_recall_owned_crease 0.5625 | k15_purity_best_crease 0.4737 | k15_owned_waist 2 | k15_best_waist 0.5106 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.77 | k15_inertia 59792.89 | k20_K 20 | k20_ami 0.4827 | k20_size_cv 0.6272 | k20_purity_mean 0.8623 | k20_old_in_novel_owned 26 | k20_bal_acc_hungarian 0.4726 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.93 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.58 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.4815 | k20_owned_crease 2 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.9062 | k20_purity_best_crease 0.70 | k20_owned_waist 2 | k20_best_waist 0.4681 | k20_recall_owned_waist 0.9255 | k20_purity_best_waist 1.00 | k20_inertia 54730.95 | k30_K 30 | k30_ami 0.4371 | k30_size_cv 0.6597 | k30_purity_mean 0.8417 | k30_old_in_novel_owned 18 | k30_bal_acc_hungarian 0.3154 | k30_owned_silk 12 | k30_best_silk 0.1690 | k30_recall_owned_silk 0.97 | k30_purity_best_silk 0.99 | k30_owned_inclusion 5 | k30_best_inclusion 0.2148 | k30_recall_owned_inclusion 0.7407 | k30_purity_best_inclusion 0.83 | k30_owned_rolled 1 | k30_best_rolled 0.5294 | k30_recall_owned_rolled 0.5294 | k30_purity_best_rolled 0.38 | k30_owned_crease 3 | k30_best_crease 0.3125 | k30_recall_owned_crease 0.8125 | k30_purity_best_crease 0.91 | k30_owned_waist 5 | k30_best_waist 0.3511 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 1.00 | k30_inertia 48074.38 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_015932__gc10_e09_05_v2_r0_s1
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r0_s1, compared with gc10_pl01b_r0_s1; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 1 | k5_K 5 | k5_ami 0.4490 | k5_size_cv 0.5518 | k5_purity_mean 0.6656 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5687 | k5_owned_silk 2 | k5_best_silk 0.5238 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.9692 | k5_owned_inclusion 1 | k5_best_inclusion 0.3185 | k5_recall_owned_inclusion 0.3185 | k5_purity_best_inclusion 0.4674 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8438 | k5_recall_owned_crease 0.8438 | k5_purity_best_crease 0.3068 | k5_owned_waist 1 | k5_best_waist 0.86 | k5_recall_owned_waist 0.86 | k5_purity_best_waist 0.9101 | k5_inertia 82649.41 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4554 | k10_size_cv 0.4406 | k10_purity_mean 0.7570 | k10_old_in_novel_owned 36 | k10_bal_acc_hungarian 0.4746 | k10_owned_silk 5 | k10_best_silk 0.2738 | k10_recall_owned_silk 0.9286 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3704 | k10_recall_owned_inclusion 0.6741 | k10_purity_best_inclusion 0.5435 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5625 | k10_recall_owned_crease 0.5625 | k10_purity_best_crease 0.3158 | k10_owned_waist 1 | k10_best_waist 0.8723 | k10_recall_owned_waist 0.8723 | k10_purity_best_waist 0.9318 | k10_inertia 69302.27 | k15_K 15 | k15_ami 0.4536 | k15_size_cv 0.4293 | k15_purity_mean 0.7872 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.4049 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9119 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.6963 | k15_purity_best_inclusion 0.5968 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6875 | k15_recall_owned_crease 0.6875 | k15_purity_best_crease 0.5946 | k15_owned_waist 2 | k15_best_waist 0.5213 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.92 | k15_inertia 59908.95 | k20_K 20 | k20_ami 0.4496 | k20_size_cv 0.4601 | k20_purity_mean 0.8260 | k20_old_in_novel_owned 21 | k20_bal_acc_hungarian 0.4313 | k20_owned_silk 9 | k20_best_silk 0.1548 | k20_recall_owned_silk 0.97 | k20_purity_best_silk 0.98 | k20_owned_inclusion 5 | k20_best_inclusion 0.2444 | k20_recall_owned_inclusion 0.6889 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.5938 | k20_purity_best_crease 0.86 | k20_owned_waist 3 | k20_best_waist 0.4574 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 0.93 | k20_inertia 55312.71 | k30_K 30 | k30_ami 0.4363 | k30_size_cv 0.5679 | k30_purity_mean 0.8556 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3753 | k30_owned_silk 14 | k30_best_silk 0.1405 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 0.98 | k30_owned_inclusion 5 | k30_best_inclusion 0.2074 | k30_recall_owned_inclusion 0.7333 | k30_purity_best_inclusion 1.00 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.46 | k30_owned_crease 2 | k30_best_crease 0.5625 | k30_recall_owned_crease 0.9062 | k30_purity_best_crease 0.56 | k30_owned_waist 5 | k30_best_waist 0.3191 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.88 | k30_inertia 48769.50 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_015943__gc10_e09_05_v2_r0_s2
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r0_s2, compared with gc10_pl01b_r0_s2; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 2 | k5_K 5 | k5_ami 0.4221 | k5_size_cv 0.4612 | k5_purity_mean 0.6710 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5709 | k5_owned_silk 3 | k5_best_silk 0.5452 | k5_recall_owned_silk 1.00 | k5_purity_best_silk 0.8980 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.8750 | k5_recall_owned_crease 0.8750 | k5_purity_best_crease 0.3011 | k5_owned_waist 1 | k5_best_waist 0.87 | k5_recall_owned_waist 0.87 | k5_purity_best_waist 0.9535 | k5_inertia 83112.65 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4192 | k10_size_cv 0.4234 | k10_purity_mean 0.7390 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4142 | k10_owned_silk 5 | k10_best_silk 0.2810 | k10_recall_owned_silk 0.9452 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.2815 | k10_recall_owned_inclusion 0.5333 | k10_purity_best_inclusion 0.5429 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3333 | k10_owned_waist 2 | k10_best_waist 0.5106 | k10_recall_owned_waist 0.9787 | k10_purity_best_waist 0.8276 | k10_inertia 69190.51 | k15_K 15 | k15_ami 0.4401 | k15_size_cv 0.4058 | k15_purity_mean 0.7895 | k15_old_in_novel_owned 24 | k15_bal_acc_hungarian 0.3796 | k15_owned_silk 7 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.8714 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.3037 | k15_recall_owned_inclusion 0.9111 | k15_purity_best_inclusion 0.5190 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6562 | k15_recall_owned_crease 0.6562 | k15_purity_best_crease 0.4565 | k15_owned_waist 2 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8511 | k15_purity_best_waist 0.91 | k15_inertia 60352.54 | k20_K 20 | k20_ami 0.4589 | k20_size_cv 0.6450 | k20_purity_mean 0.8086 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.4352 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.86 | k20_purity_best_silk 1.00 | k20_owned_inclusion 5 | k20_best_inclusion 0.2963 | k20_recall_owned_inclusion 0.9259 | k20_purity_best_inclusion 0.53 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.3929 | k20_owned_crease 2 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.8125 | k20_purity_best_crease 0.76 | k20_owned_waist 3 | k20_best_waist 0.3936 | k20_recall_owned_waist 0.9043 | k20_purity_best_waist 0.97 | k20_inertia 55080.66 | k30_K 30 | k30_ami 0.4421 | k30_size_cv 0.7591 | k30_purity_mean 0.8412 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.3375 | k30_owned_silk 11 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.94 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2000 | k30_recall_owned_inclusion 0.7926 | k30_purity_best_inclusion 0.96 | k30_owned_rolled 1 | k30_best_rolled 0.5294 | k30_recall_owned_rolled 0.5294 | k30_purity_best_rolled 0.47 | k30_owned_crease 1 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.5000 | k30_purity_best_crease 0.80 | k30_owned_waist 7 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 48519.00 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_015952__gc10_e09_05_v2_r0_s3
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r0_s3, compared with gc10_pl01b_r0_s3; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 3 | k5_K 5 | k5_ami 0.3460 | k5_size_cv 0.5034 | k5_purity_mean 0.7178 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3151 | k5_owned_silk 3 | k5_best_silk 0.4048 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.6137 | k5_owned_inclusion 1 | k5_best_inclusion 0.3407 | k5_recall_owned_inclusion 0.3407 | k5_purity_best_inclusion 0.5111 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.83 | k5_recall_owned_waist 0.83 | k5_purity_best_waist 0.5000 | k5_inertia 82477.73 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4483 | k10_size_cv 0.6359 | k10_purity_mean 0.7621 | k10_old_in_novel_owned 26 | k10_bal_acc_hungarian 0.4540 | k10_owned_silk 4 | k10_best_silk 0.3048 | k10_recall_owned_silk 0.9476 | k10_purity_best_silk 0.71 | k10_owned_inclusion 2 | k10_best_inclusion 0.2889 | k10_recall_owned_inclusion 0.5630 | k10_purity_best_inclusion 0.5493 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.8125 | k10_recall_owned_crease 0.8125 | k10_purity_best_crease 0.5652 | k10_owned_waist 2 | k10_best_waist 0.4681 | k10_recall_owned_waist 0.8830 | k10_purity_best_waist 0.9362 | k10_inertia 68934.98 | k15_K 15 | k15_ami 0.4689 | k15_size_cv 0.5473 | k15_purity_mean 0.8033 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.4620 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9190 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.5873 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6250 | k15_recall_owned_crease 0.6250 | k15_purity_best_crease 0.9091 | k15_owned_waist 3 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.93 | k15_inertia 59236.22 | k20_K 20 | k20_ami 0.4475 | k20_size_cv 0.6360 | k20_purity_mean 0.8173 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4256 | k20_owned_silk 8 | k20_best_silk 0.2500 | k20_recall_owned_silk 0.97 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2593 | k20_recall_owned_inclusion 0.6296 | k20_purity_best_inclusion 0.95 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.3871 | k20_owned_crease 1 | k20_best_crease 0.5938 | k20_recall_owned_crease 0.5938 | k20_purity_best_crease 0.95 | k20_owned_waist 4 | k20_best_waist 0.3191 | k20_recall_owned_waist 0.9362 | k20_purity_best_waist 0.86 | k20_inertia 53967.83 | k30_K 30 | k30_ami 0.4327 | k30_size_cv 0.7830 | k30_purity_mean 0.8344 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.3324 | k30_owned_silk 11 | k30_best_silk 0.2452 | k30_recall_owned_silk 0.93 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.8444 | k30_purity_best_inclusion 0.97 | k30_owned_rolled 1 | k30_best_rolled 0.4706 | k30_recall_owned_rolled 0.4706 | k30_purity_best_rolled 0.42 | k30_owned_crease 3 | k30_best_crease 0.4688 | k30_recall_owned_crease 0.7812 | k30_purity_best_crease 1.00 | k30_owned_waist 7 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 47379.87 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020003__gc10_e09_05_v2_r0_s4
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r0_s4, compared with gc10_pl01b_r0_s4; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 4 | k5_K 5 | k5_ami 0.3457 | k5_size_cv 0.4844 | k5_purity_mean 0.7168 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3167 | k5_owned_silk 3 | k5_best_silk 0.4024 | k5_recall_owned_silk 0.93 | k5_purity_best_silk 0.6213 | k5_owned_inclusion 1 | k5_best_inclusion 0.3407 | k5_recall_owned_inclusion 0.3407 | k5_purity_best_inclusion 0.4842 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.84 | k5_recall_owned_waist 0.84 | k5_purity_best_waist 0.5032 | k5_inertia 82926.20 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4400 | k10_size_cv 0.3792 | k10_purity_mean 0.7135 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.5190 | k10_owned_silk 5 | k10_best_silk 0.2905 | k10_recall_owned_silk 0.9524 | k10_purity_best_silk 0.95 | k10_owned_inclusion 2 | k10_best_inclusion 0.2593 | k10_recall_owned_inclusion 0.5037 | k10_purity_best_inclusion 0.8333 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 2 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.8750 | k10_purity_best_crease 0.3148 | k10_owned_waist 1 | k10_best_waist 0.8191 | k10_recall_owned_waist 0.8191 | k10_purity_best_waist 0.9747 | k10_inertia 69334.23 | k15_K 15 | k15_ami 0.4492 | k15_size_cv 0.4378 | k15_purity_mean 0.7801 | k15_old_in_novel_owned 21 | k15_bal_acc_hungarian 0.3653 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.9556 | k15_purity_best_inclusion 0.5200 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.6562 | k15_recall_owned_crease 0.6562 | k15_purity_best_crease 0.4773 | k15_owned_waist 2 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8617 | k15_purity_best_waist 0.93 | k15_inertia 59803.38 | k20_K 20 | k20_ami 0.4615 | k20_size_cv 0.5921 | k20_purity_mean 0.8144 | k20_old_in_novel_owned 15 | k20_bal_acc_hungarian 0.4255 | k20_owned_silk 7 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.89 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2296 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.74 | k20_owned_rolled 1 | k20_best_rolled 0.7059 | k20_recall_owned_rolled 0.7059 | k20_purity_best_rolled 0.5217 | k20_owned_crease 1 | k20_best_crease 0.6250 | k20_recall_owned_crease 0.6250 | k20_purity_best_crease 0.74 | k20_owned_waist 4 | k20_best_waist 0.3191 | k20_recall_owned_waist 0.9149 | k20_purity_best_waist 1.00 | k20_inertia 54619.96 | k30_K 30 | k30_ami 0.4463 | k30_size_cv 0.6064 | k30_purity_mean 0.8599 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.3451 | k30_owned_silk 13 | k30_best_silk 0.1476 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 1 | k30_best_rolled 0.5882 | k30_recall_owned_rolled 0.5882 | k30_purity_best_rolled 0.83 | k30_owned_crease 3 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.9062 | k30_purity_best_crease 0.74 | k30_owned_waist 4 | k30_best_waist 0.3298 | k30_recall_owned_waist 0.9255 | k30_purity_best_waist 0.94 | k30_inertia 48489.25 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020012__gc10_e09_05_v2_r1_s0
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r1_s0, compared with gc10_pl01b_r1_s0; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 0 | k5_K 5 | k5_ami 0.4015 | k5_size_cv 0.5407 | k5_purity_mean 0.6985 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5294 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6082 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2308 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.8857 | k5_inertia 89441.90 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4845 | k10_size_cv 0.4338 | k10_purity_mean 0.7648 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3797 | k10_owned_silk 4 | k10_best_silk 0.2976 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.2889 | k10_recall_owned_inclusion 0.7037 | k10_purity_best_inclusion 0.6842 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2800 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8413 | k10_inertia 73728.16 | k15_K 15 | k15_ami 0.4438 | k15_size_cv 0.4916 | k15_purity_mean 0.8052 | k15_old_in_novel_owned 40 | k15_bal_acc_hungarian 0.3938 | k15_owned_silk 6 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.8548 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.9556 | k15_purity_best_inclusion 0.6557 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3333 | k15_owned_waist 3 | k15_best_waist 0.5532 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.83 | k15_inertia 64982.90 | k20_K 20 | k20_ami 0.4610 | k20_size_cv 0.6013 | k20_purity_mean 0.8283 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.4238 | k20_owned_silk 9 | k20_best_silk 0.2500 | k20_recall_owned_silk 0.95 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.8444 | k20_purity_best_inclusion 0.74 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.3421 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.64 | k20_owned_waist 3 | k20_best_waist 0.4149 | k20_recall_owned_waist 0.8723 | k20_purity_best_waist 0.97 | k20_inertia 59532.36 | k30_K 30 | k30_ami 0.4467 | k30_size_cv 0.7902 | k30_purity_mean 0.8432 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3664 | k30_owned_silk 13 | k30_best_silk 0.2476 | k30_recall_owned_silk 0.99 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.7407 | k30_purity_best_inclusion 0.85 | k30_owned_rolled 1 | k30_best_rolled 0.6471 | k30_recall_owned_rolled 0.6471 | k30_purity_best_rolled 0.85 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.82 | k30_owned_waist 5 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9468 | k30_purity_best_waist 0.92 | k30_inertia 52287.19 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020023__gc10_e09_05_v2_r1_s1
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r1_s1, compared with gc10_pl01b_r1_s1; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 1 | k5_K 5 | k5_ami 0.3956 | k5_size_cv 0.5015 | k5_purity_mean 0.6924 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5245 | k5_owned_silk 3 | k5_best_silk 0.4190 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6286 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2234 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.8692 | k5_inertia 89757.89 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4378 | k10_size_cv 0.3849 | k10_purity_mean 0.8096 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3534 | k10_owned_silk 5 | k10_best_silk 0.2500 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 1.00 | k10_owned_inclusion 2 | k10_best_inclusion 0.3111 | k10_recall_owned_inclusion 0.6148 | k10_purity_best_inclusion 0.5676 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5938 | k10_recall_owned_crease 0.5938 | k10_purity_best_crease 0.2923 | k10_owned_waist 2 | k10_best_waist 0.5532 | k10_recall_owned_waist 0.9681 | k10_purity_best_waist 0.8525 | k10_inertia 74755.51 | k15_K 15 | k15_ami 0.4716 | k15_size_cv 0.5910 | k15_purity_mean 0.7839 | k15_old_in_novel_owned 19 | k15_bal_acc_hungarian 0.4091 | k15_owned_silk 6 | k15_best_silk 0.2500 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.3037 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.9111 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 2 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.8125 | k15_purity_best_crease 0.4545 | k15_owned_waist 2 | k15_best_waist 0.6702 | k15_recall_owned_waist 0.8404 | k15_purity_best_waist 0.97 | k15_inertia 65860.51 | k20_K 20 | k20_ami 0.4633 | k20_size_cv 0.6762 | k20_purity_mean 0.8416 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3968 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.99 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.6074 | k20_purity_best_inclusion 0.91 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.5000 | k20_owned_crease 2 | k20_best_crease 0.4062 | k20_recall_owned_crease 0.7500 | k20_purity_best_crease 0.72 | k20_owned_waist 4 | k20_best_waist 0.2766 | k20_recall_owned_waist 0.9468 | k20_purity_best_waist 0.90 | k20_inertia 59190.00 | k30_K 30 | k30_ami 0.4426 | k30_size_cv 0.8826 | k30_purity_mean 0.8214 | k30_old_in_novel_owned 21 | k30_bal_acc_hungarian 0.3739 | k30_owned_silk 12 | k30_best_silk 0.2500 | k30_recall_owned_silk 0.99 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.7704 | k30_purity_best_inclusion 0.87 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.75 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.68 | k30_owned_waist 6 | k30_best_waist 0.2553 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.96 | k30_inertia 52723.62 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020033__gc10_e09_05_v2_r1_s2
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r1_s2, compared with gc10_pl01b_r1_s2; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 2 | k5_K 5 | k5_ami 0.3989 | k5_size_cv 0.5253 | k5_purity_mean 0.6972 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.5323 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6146 | k5_owned_inclusion 0 | k5_best_inclusion 0.0000 | k5_recall_owned_inclusion 0.0000 | k5_purity_best_inclusion 0.0000 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 1 | k5_best_crease 0.6562 | k5_recall_owned_crease 0.6562 | k5_purity_best_crease 0.2165 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.8692 | k5_inertia 89581.10 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4373 | k10_size_cv 0.4026 | k10_purity_mean 0.7708 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4121 | k10_owned_silk 5 | k10_best_silk 0.2476 | k10_recall_owned_silk 0.9667 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3704 | k10_recall_owned_inclusion 0.5556 | k10_purity_best_inclusion 0.6667 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.3043 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8413 | k10_inertia 74510.82 | k15_K 15 | k15_ami 0.4764 | k15_size_cv 0.4735 | k15_purity_mean 0.8154 | k15_old_in_novel_owned 20 | k15_bal_acc_hungarian 0.4448 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9405 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.6889 | k15_purity_best_inclusion 0.7018 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.8438 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.5510 | k15_owned_waist 2 | k15_best_waist 0.4468 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.98 | k15_inertia 65262.89 | k20_K 20 | k20_ami 0.4495 | k20_size_cv 0.5609 | k20_purity_mean 0.8319 | k20_old_in_novel_owned 25 | k20_bal_acc_hungarian 0.4551 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.91 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.93 | k20_owned_rolled 1 | k20_best_rolled 0.8235 | k20_recall_owned_rolled 0.8235 | k20_purity_best_rolled 0.3889 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.71 | k20_owned_waist 3 | k20_best_waist 0.4468 | k20_recall_owned_waist 0.9255 | k20_purity_best_waist 0.89 | k20_inertia 59920.51 | k30_K 30 | k30_ami 0.4377 | k30_size_cv 0.6856 | k30_purity_mean 0.8354 | k30_old_in_novel_owned 16 | k30_bal_acc_hungarian 0.2570 | k30_owned_silk 12 | k30_best_silk 0.1738 | k30_recall_owned_silk 0.99 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.7926 | k30_purity_best_inclusion 0.94 | k30_owned_rolled 1 | k30_best_rolled 0.3529 | k30_recall_owned_rolled 0.3529 | k30_purity_best_rolled 0.38 | k30_owned_crease 2 | k30_best_crease 0.3438 | k30_recall_owned_crease 0.5938 | k30_purity_best_crease 0.73 | k30_owned_waist 6 | k30_best_waist 0.1702 | k30_recall_owned_waist 0.8511 | k30_purity_best_waist 0.94 | k30_inertia 52804.92 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020043__gc10_e09_05_v2_r1_s3
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r1_s3, compared with gc10_pl01b_r1_s3; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 3 | k5_K 5 | k5_ami 0.3968 | k5_size_cv 0.5197 | k5_purity_mean 0.7443 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3506 | k5_owned_silk 3 | k5_best_silk 0.4214 | k5_recall_owned_silk 0.96 | k5_purity_best_silk 0.6367 | k5_owned_inclusion 1 | k5_best_inclusion 0.3111 | k5_recall_owned_inclusion 0.3111 | k5_purity_best_inclusion 0.5385 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5813 | k5_inertia 89382.68 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4686 | k10_size_cv 0.5457 | k10_purity_mean 0.7620 | k10_old_in_novel_owned 21 | k10_bal_acc_hungarian 0.3956 | k10_owned_silk 4 | k10_best_silk 0.2905 | k10_recall_owned_silk 0.9595 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.3333 | k10_recall_owned_inclusion 0.6222 | k10_purity_best_inclusion 0.6818 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 0 | k10_best_crease 0.0000 | k10_recall_owned_crease 0.0000 | k10_purity_best_crease 0.0000 | k10_owned_waist 2 | k10_best_waist 0.5638 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8154 | k10_inertia 74249.21 | k15_K 15 | k15_ami 0.4451 | k15_size_cv 0.5069 | k15_purity_mean 0.8067 | k15_old_in_novel_owned 37 | k15_bal_acc_hungarian 0.3700 | k15_owned_silk 7 | k15_best_silk 0.2548 | k15_recall_owned_silk 0.9667 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2963 | k15_recall_owned_inclusion 0.6889 | k15_purity_best_inclusion 0.6897 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4062 | k15_recall_owned_crease 0.4062 | k15_purity_best_crease 0.3714 | k15_owned_waist 3 | k15_best_waist 0.5106 | k15_recall_owned_waist 0.9787 | k15_purity_best_waist 0.83 | k15_inertia 64676.45 | k20_K 20 | k20_ami 0.4640 | k20_size_cv 0.6677 | k20_purity_mean 0.8450 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3750 | k20_owned_silk 9 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.93 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2889 | k20_recall_owned_inclusion 0.8889 | k20_purity_best_inclusion 0.56 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 2 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.7812 | k20_purity_best_crease 0.70 | k20_owned_waist 3 | k20_best_waist 0.4894 | k20_recall_owned_waist 0.9574 | k20_purity_best_waist 0.88 | k20_inertia 58834.02 | k30_K 30 | k30_ami 0.4336 | k30_size_cv 0.5648 | k30_purity_mean 0.8516 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3673 | k30_owned_silk 14 | k30_best_silk 0.1357 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2667 | k30_recall_owned_inclusion 0.6963 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 1 | k30_best_rolled 0.8235 | k30_recall_owned_rolled 0.8235 | k30_purity_best_rolled 0.54 | k30_owned_crease 2 | k30_best_crease 0.3125 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.91 | k30_owned_waist 6 | k30_best_waist 0.2979 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 0.97 | k30_inertia 51927.98 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020053__gc10_e09_05_v2_r1_s4
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r1_s4, compared with gc10_pl01b_r1_s4; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 4 | k5_K 5 | k5_ami 0.3482 | k5_size_cv 0.2987 | k5_purity_mean 0.7415 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4059 | k5_owned_silk 3 | k5_best_silk 0.2976 | k5_recall_owned_silk 0.77 | k5_purity_best_silk 0.9766 | k5_owned_inclusion 1 | k5_best_inclusion 0.7111 | k5_recall_owned_inclusion 0.7111 | k5_purity_best_inclusion 0.4923 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.4745 | k5_inertia 90109.13 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4874 | k10_size_cv 0.4522 | k10_purity_mean 0.7403 | k10_old_in_novel_owned 26 | k10_bal_acc_hungarian 0.5288 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.8548 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.4519 | k10_recall_owned_inclusion 0.9481 | k10_purity_best_inclusion 0.5304 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.5312 | k10_recall_owned_crease 0.5312 | k10_purity_best_crease 0.3469 | k10_owned_waist 1 | k10_best_waist 0.8511 | k10_recall_owned_waist 0.8511 | k10_purity_best_waist 0.9756 | k10_inertia 75282.20 | k15_K 15 | k15_ami 0.4541 | k15_size_cv 0.5991 | k15_purity_mean 0.7795 | k15_old_in_novel_owned 29 | k15_bal_acc_hungarian 0.4601 | k15_owned_silk 7 | k15_best_silk 0.2952 | k15_recall_owned_silk 0.9286 | k15_purity_best_silk 0.99 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.7926 | k15_purity_best_inclusion 0.8864 | k15_owned_rolled 1 | k15_best_rolled 0.8824 | k15_recall_owned_rolled 0.8824 | k15_purity_best_rolled 0.5556 | k15_owned_crease 0 | k15_best_crease 0.0000 | k15_recall_owned_crease 0.0000 | k15_purity_best_crease 0.0000 | k15_owned_waist 2 | k15_best_waist 0.5213 | k15_recall_owned_waist 0.9681 | k15_purity_best_waist 0.88 | k15_inertia 66652.62 | k20_K 20 | k20_ami 0.4647 | k20_size_cv 0.6525 | k20_purity_mean 0.8311 | k20_old_in_novel_owned 19 | k20_bal_acc_hungarian 0.3521 | k20_owned_silk 8 | k20_best_silk 0.2476 | k20_recall_owned_silk 0.94 | k20_purity_best_silk 1.00 | k20_owned_inclusion 4 | k20_best_inclusion 0.2741 | k20_recall_owned_inclusion 0.8593 | k20_purity_best_inclusion 0.90 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.8125 | k20_recall_owned_crease 0.8125 | k20_purity_best_crease 0.45 | k20_owned_waist 4 | k20_best_waist 0.3085 | k20_recall_owned_waist 0.8191 | k20_purity_best_waist 0.97 | k20_inertia 59384.50 | k30_K 30 | k30_ami 0.4542 | k30_size_cv 0.6044 | k30_purity_mean 0.8886 | k30_old_in_novel_owned 25 | k30_bal_acc_hungarian 0.3670 | k30_owned_silk 12 | k30_best_silk 0.1405 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 0.97 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.8296 | k30_purity_best_inclusion 0.87 | k30_owned_rolled 1 | k30_best_rolled 0.8235 | k30_recall_owned_rolled 0.8235 | k30_purity_best_rolled 0.78 | k30_owned_crease 2 | k30_best_crease 0.4062 | k30_recall_owned_crease 0.7500 | k30_purity_best_crease 0.68 | k30_owned_waist 7 | k30_best_waist 0.2128 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 1.00 | k30_inertia 52359.21 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020103__gc10_e09_05_v2_r2_s0
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r2_s0, compared with gc10_pl01b_r2_s0; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 0 | k5_K 5 | k5_ami 0.4403 | k5_size_cv 0.4040 | k5_purity_mean 0.7020 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4580 | k5_owned_silk 2 | k5_best_silk 0.5405 | k5_recall_owned_silk 0.79 | k5_purity_best_silk 0.9660 | k5_owned_inclusion 2 | k5_best_inclusion 0.6074 | k5_recall_owned_inclusion 0.9111 | k5_purity_best_inclusion 0.4881 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5886 | k5_inertia 92754.14 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4394 | k10_size_cv 0.3881 | k10_purity_mean 0.7804 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3854 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5926 | k10_purity_best_inclusion 0.6557 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2857 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7838 | k10_inertia 76462.56 | k15_K 15 | k15_ami 0.4423 | k15_size_cv 0.5495 | k15_purity_mean 0.8142 | k15_old_in_novel_owned 36 | k15_bal_acc_hungarian 0.4096 | k15_owned_silk 7 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.8833 | k15_purity_best_silk 0.98 | k15_owned_inclusion 3 | k15_best_inclusion 0.3852 | k15_recall_owned_inclusion 0.8519 | k15_purity_best_inclusion 0.5000 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.7812 | k15_recall_owned_crease 0.7812 | k15_purity_best_crease 0.4808 | k15_owned_waist 3 | k15_best_waist 0.4574 | k15_recall_owned_waist 0.8830 | k15_purity_best_waist 1.00 | k15_inertia 67977.65 | k20_K 20 | k20_ami 0.4520 | k20_size_cv 0.6528 | k20_purity_mean 0.8107 | k20_old_in_novel_owned 27 | k20_bal_acc_hungarian 0.3714 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.89 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.8667 | k20_purity_best_inclusion 0.55 | k20_owned_rolled 1 | k20_best_rolled 0.6471 | k20_recall_owned_rolled 0.6471 | k20_purity_best_rolled 0.4583 | k20_owned_crease 1 | k20_best_crease 0.4375 | k20_recall_owned_crease 0.4375 | k20_purity_best_crease 0.64 | k20_owned_waist 5 | k20_best_waist 0.2234 | k20_recall_owned_waist 0.9362 | k20_purity_best_waist 0.95 | k20_inertia 62269.23 | k30_K 30 | k30_ami 0.4297 | k30_size_cv 0.6204 | k30_purity_mean 0.8426 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.3824 | k30_owned_silk 14 | k30_best_silk 0.2000 | k30_recall_owned_silk 0.98 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2296 | k30_recall_owned_inclusion 0.7852 | k30_purity_best_inclusion 0.86 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.67 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7500 | k30_purity_best_crease 0.70 | k30_owned_waist 4 | k30_best_waist 0.2766 | k30_recall_owned_waist 0.8936 | k30_purity_best_waist 0.96 | k30_inertia 54717.25 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020114__gc10_e09_05_v2_r2_s1
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r2_s1, compared with gc10_pl01b_r2_s1; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 1 | k5_K 5 | k5_ami 0.4078 | k5_size_cv 0.4997 | k5_purity_mean 0.7390 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3496 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.6473 | k5_owned_inclusion 1 | k5_best_inclusion 0.3037 | k5_recall_owned_inclusion 0.3037 | k5_purity_best_inclusion 0.4767 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5962 | k5_inertia 92578.14 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4434 | k10_size_cv 0.4028 | k10_purity_mean 0.7872 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3914 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.99 | k10_owned_inclusion 2 | k10_best_inclusion 0.3037 | k10_recall_owned_inclusion 0.6000 | k10_purity_best_inclusion 0.5942 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2979 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8056 | k10_inertia 76947.13 | k15_K 15 | k15_ami 0.4464 | k15_size_cv 0.4018 | k15_purity_mean 0.8147 | k15_old_in_novel_owned 36 | k15_bal_acc_hungarian 0.3180 | k15_owned_silk 8 | k15_best_silk 0.2024 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2741 | k15_recall_owned_inclusion 0.7556 | k15_purity_best_inclusion 0.7551 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.4375 | k15_recall_owned_crease 0.4375 | k15_purity_best_crease 0.3500 | k15_owned_waist 2 | k15_best_waist 0.6170 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.79 | k15_inertia 68395.62 | k20_K 20 | k20_ami 0.4443 | k20_size_cv 0.5823 | k20_purity_mean 0.8351 | k20_old_in_novel_owned 22 | k20_bal_acc_hungarian 0.3475 | k20_owned_silk 9 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.95 | k20_purity_best_silk 1.00 | k20_owned_inclusion 3 | k20_best_inclusion 0.2667 | k20_recall_owned_inclusion 0.7407 | k20_purity_best_inclusion 0.82 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.5000 | k20_recall_owned_crease 0.5000 | k20_purity_best_crease 0.46 | k20_owned_waist 5 | k20_best_waist 0.2660 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 0.93 | k20_inertia 62545.11 | k30_K 30 | k30_ami 0.4357 | k30_size_cv 0.7439 | k30_purity_mean 0.8673 | k30_old_in_novel_owned 24 | k30_bal_acc_hungarian 0.3177 | k30_owned_silk 14 | k30_best_silk 0.2024 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 6 | k30_best_inclusion 0.2519 | k30_recall_owned_inclusion 0.8296 | k30_purity_best_inclusion 0.92 | k30_owned_rolled 0 | k30_best_rolled 0.0000 | k30_recall_owned_rolled 0.0000 | k30_purity_best_rolled 0.00 | k30_owned_crease 2 | k30_best_crease 0.5312 | k30_recall_owned_crease 0.6250 | k30_purity_best_crease 0.74 | k30_owned_waist 6 | k30_best_waist 0.1915 | k30_recall_owned_waist 0.9787 | k30_purity_best_waist 1.00 | k30_inertia 55272.96 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020125__gc10_e09_05_v2_r2_s2
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r2_s2, compared with gc10_pl01b_r2_s2; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 2 | k5_K 5 | k5_ami 0.4238 | k5_size_cv 0.3988 | k5_purity_mean 0.7763 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3881 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.7149 | k5_owned_inclusion 1 | k5_best_inclusion 0.4963 | k5_recall_owned_inclusion 0.4963 | k5_purity_best_inclusion 0.6321 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5849 | k5_inertia 92739.22 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4431 | k10_size_cv 0.4191 | k10_purity_mean 0.7863 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.3826 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.9643 | k10_purity_best_silk 0.98 | k10_owned_inclusion 2 | k10_best_inclusion 0.2963 | k10_recall_owned_inclusion 0.5926 | k10_purity_best_inclusion 0.7018 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.3111 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7733 | k10_inertia 76585.35 | k15_K 15 | k15_ami 0.4466 | k15_size_cv 0.3516 | k15_purity_mean 0.8124 | k15_old_in_novel_owned 50 | k15_bal_acc_hungarian 0.4136 | k15_owned_silk 8 | k15_best_silk 0.2024 | k15_recall_owned_silk 0.9619 | k15_purity_best_silk 1.00 | k15_owned_inclusion 3 | k15_best_inclusion 0.2815 | k15_recall_owned_inclusion 0.7111 | k15_purity_best_inclusion 0.8085 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 2 | k15_best_crease 0.4688 | k15_recall_owned_crease 0.8438 | k15_purity_best_crease 0.3261 | k15_owned_waist 2 | k15_best_waist 0.4681 | k15_recall_owned_waist 0.8723 | k15_purity_best_waist 0.98 | k15_inertia 68539.39 | k20_K 20 | k20_ami 0.4354 | k20_size_cv 0.6105 | k20_purity_mean 0.8021 | k20_old_in_novel_owned 28 | k20_bal_acc_hungarian 0.3575 | k20_owned_silk 7 | k20_best_silk 0.2452 | k20_recall_owned_silk 0.90 | k20_purity_best_silk 0.99 | k20_owned_inclusion 5 | k20_best_inclusion 0.2519 | k20_recall_owned_inclusion 0.8593 | k20_purity_best_inclusion 0.79 | k20_owned_rolled 1 | k20_best_rolled 0.4706 | k20_recall_owned_rolled 0.4706 | k20_purity_best_rolled 0.3333 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.58 | k20_owned_waist 4 | k20_best_waist 0.3511 | k20_recall_owned_waist 0.9149 | k20_purity_best_waist 0.97 | k20_inertia 62356.22 | k30_K 30 | k30_ami 0.4611 | k30_size_cv 0.6968 | k30_purity_mean 0.8989 | k30_old_in_novel_owned 22 | k30_bal_acc_hungarian 0.3950 | k30_owned_silk 13 | k30_best_silk 0.2000 | k30_recall_owned_silk 1.00 | k30_purity_best_silk 1.00 | k30_owned_inclusion 5 | k30_best_inclusion 0.2444 | k30_recall_owned_inclusion 0.8222 | k30_purity_best_inclusion 0.82 | k30_owned_rolled 1 | k30_best_rolled 0.7647 | k30_recall_owned_rolled 0.7647 | k30_purity_best_rolled 0.72 | k30_owned_crease 2 | k30_best_crease 0.5000 | k30_recall_owned_crease 0.7188 | k30_purity_best_crease 0.94 | k30_owned_waist 6 | k30_best_waist 0.2660 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.96 | k30_inertia 54609.66 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020135__gc10_e09_05_v2_r2_s3
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r2_s3, compared with gc10_pl01b_r2_s3; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 3 | k5_K 5 | k5_ami 0.3786 | k5_size_cv 0.2436 | k5_purity_mean 0.7347 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.4109 | k5_owned_silk 3 | k5_best_silk 0.3310 | k5_recall_owned_silk 0.82 | k5_purity_best_silk 0.7596 | k5_owned_inclusion 1 | k5_best_inclusion 0.6296 | k5_recall_owned_inclusion 0.6296 | k5_purity_best_inclusion 0.4802 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 1.00 | k5_recall_owned_waist 1.00 | k5_purity_best_waist 0.5629 | k5_inertia 92630.82 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4645 | k10_size_cv 0.4416 | k10_purity_mean 0.7824 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4165 | k10_owned_silk 4 | k10_best_silk 0.2952 | k10_recall_owned_silk 0.8476 | k10_purity_best_silk 0.98 | k10_owned_inclusion 3 | k10_best_inclusion 0.4074 | k10_recall_owned_inclusion 0.9704 | k10_purity_best_inclusion 0.4867 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4688 | k10_recall_owned_crease 0.4688 | k10_purity_best_crease 0.3333 | k10_owned_waist 2 | k10_best_waist 0.6170 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.7838 | k10_inertia 76431.62 | k15_K 15 | k15_ami 0.4664 | k15_size_cv 0.5476 | k15_purity_mean 0.8049 | k15_old_in_novel_owned 23 | k15_bal_acc_hungarian 0.3925 | k15_owned_silk 6 | k15_best_silk 0.2476 | k15_recall_owned_silk 0.9095 | k15_purity_best_silk 1.00 | k15_owned_inclusion 4 | k15_best_inclusion 0.2889 | k15_recall_owned_inclusion 0.8593 | k15_purity_best_inclusion 0.5735 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.7083 | k15_owned_waist 2 | k15_best_waist 0.6596 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.79 | k15_inertia 68098.40 | k20_K 20 | k20_ami 0.4493 | k20_size_cv 0.5345 | k20_purity_mean 0.8121 | k20_old_in_novel_owned 24 | k20_bal_acc_hungarian 0.4288 | k20_owned_silk 9 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.98 | k20_purity_best_silk 0.98 | k20_owned_inclusion 4 | k20_best_inclusion 0.3037 | k20_recall_owned_inclusion 0.7333 | k20_purity_best_inclusion 0.95 | k20_owned_rolled 1 | k20_best_rolled 0.7647 | k20_recall_owned_rolled 0.7647 | k20_purity_best_rolled 0.5000 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.62 | k20_owned_waist 3 | k20_best_waist 0.4043 | k20_recall_owned_waist 0.8936 | k20_purity_best_waist 0.90 | k20_inertia 61994.48 | k30_K 30 | k30_ami 0.4306 | k30_size_cv 0.6533 | k30_purity_mean 0.8301 | k30_old_in_novel_owned 23 | k30_bal_acc_hungarian 0.3979 | k30_owned_silk 13 | k30_best_silk 0.2000 | k30_recall_owned_silk 0.97 | k30_purity_best_silk 1.00 | k30_owned_inclusion 7 | k30_best_inclusion 0.1852 | k30_recall_owned_inclusion 0.7333 | k30_purity_best_inclusion 0.86 | k30_owned_rolled 1 | k30_best_rolled 0.7647 | k30_recall_owned_rolled 0.7647 | k30_purity_best_rolled 0.65 | k30_owned_crease 1 | k30_best_crease 0.5312 | k30_recall_owned_crease 0.5312 | k30_purity_best_crease 0.89 | k30_owned_waist 6 | k30_best_waist 0.3085 | k30_recall_owned_waist 0.9894 | k30_purity_best_waist 0.91 | k30_inertia 54626.82 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261021_020146__gc10_e09_05_v2_r2_s4
why: E09.05 through replay v2 (QD20: lab/score.py replaces lab/pl_cluster.py): tier 1 of step 09 from the snapshots of gc10_v2_ft_r2_s4, compared with gc10_pl01b_r2_s4; pass = every k*_ key equal (inertia within 1e-6 relative) and C1-C3 ok
result: seed 4 | k5_K 5 | k5_ami 0.4275 | k5_size_cv 0.4028 | k5_purity_mean 0.7776 | k5_old_in_novel_owned 50 | k5_bal_acc_hungarian 0.3881 | k5_owned_silk 3 | k5_best_silk 0.4238 | k5_recall_owned_silk 0.97 | k5_purity_best_silk 0.7120 | k5_owned_inclusion 1 | k5_best_inclusion 0.4963 | k5_recall_owned_inclusion 0.4963 | k5_purity_best_inclusion 0.6321 | k5_owned_rolled 0 | k5_best_rolled 0.0000 | k5_recall_owned_rolled 0.0000 | k5_purity_best_rolled 0.0000 | k5_owned_crease 0 | k5_best_crease 0.0000 | k5_recall_owned_crease 0.0000 | k5_purity_best_crease 0.0000 | k5_owned_waist 1 | k5_best_waist 0.99 | k5_recall_owned_waist 0.99 | k5_purity_best_waist 0.5849 | k5_inertia 92764.02 | C1_ok 1 | C3_ok 1 | k10_K 10 | k10_ami 0.4579 | k10_size_cv 0.3704 | k10_purity_mean 0.7877 | k10_old_in_novel_owned 50 | k10_bal_acc_hungarian 0.4747 | k10_owned_silk 5 | k10_best_silk 0.2452 | k10_recall_owned_silk 0.8524 | k10_purity_best_silk 0.99 | k10_owned_inclusion 3 | k10_best_inclusion 0.4074 | k10_recall_owned_inclusion 0.9704 | k10_purity_best_inclusion 0.5046 | k10_owned_rolled 0 | k10_best_rolled 0.0000 | k10_recall_owned_rolled 0.0000 | k10_purity_best_rolled 0.0000 | k10_owned_crease 1 | k10_best_crease 0.4375 | k10_recall_owned_crease 0.4375 | k10_purity_best_crease 0.2917 | k10_owned_waist 1 | k10_best_waist 0.9894 | k10_recall_owned_waist 0.9894 | k10_purity_best_waist 0.8692 | k10_inertia 76921.80 | k15_K 15 | k15_ami 0.4396 | k15_size_cv 0.4845 | k15_purity_mean 0.8256 | k15_old_in_novel_owned 35 | k15_bal_acc_hungarian 0.3430 | k15_owned_silk 6 | k15_best_silk 0.2452 | k15_recall_owned_silk 0.9381 | k15_purity_best_silk 0.99 | k15_owned_inclusion 3 | k15_best_inclusion 0.3630 | k15_recall_owned_inclusion 0.8889 | k15_purity_best_inclusion 0.7424 | k15_owned_rolled 0 | k15_best_rolled 0.0000 | k15_recall_owned_rolled 0.0000 | k15_purity_best_rolled 0.0000 | k15_owned_crease 1 | k15_best_crease 0.5312 | k15_recall_owned_crease 0.5312 | k15_purity_best_crease 0.5000 | k15_owned_waist 4 | k15_best_waist 0.3404 | k15_recall_owned_waist 0.9894 | k15_purity_best_waist 0.86 | k15_inertia 69094.20 | k20_K 20 | k20_ami 0.4508 | k20_size_cv 0.5242 | k20_purity_mean 0.8147 | k20_old_in_novel_owned 19 | k20_bal_acc_hungarian 0.2983 | k20_owned_silk 9 | k20_best_silk 0.2024 | k20_recall_owned_silk 0.98 | k20_purity_best_silk 0.99 | k20_owned_inclusion 4 | k20_best_inclusion 0.2444 | k20_recall_owned_inclusion 0.7481 | k20_purity_best_inclusion 0.94 | k20_owned_rolled 0 | k20_best_rolled 0.0000 | k20_recall_owned_rolled 0.0000 | k20_purity_best_rolled 0.0000 | k20_owned_crease 1 | k20_best_crease 0.4688 | k20_recall_owned_crease 0.4688 | k20_purity_best_crease 0.62 | k20_owned_waist 4 | k20_best_waist 0.3404 | k20_recall_owned_waist 0.9894 | k20_purity_best_waist 1.00 | k20_inertia 63233.57 | k30_K 30 | k30_ami 0.4337 | k30_size_cv 0.7542 | k30_purity_mean 0.8464 | k30_old_in_novel_owned 12 | k30_bal_acc_hungarian 0.3519 | k30_owned_silk 11 | k30_best_silk 0.2024 | k30_recall_owned_silk 0.94 | k30_purity_best_silk 0.99 | k30_owned_inclusion 6 | k30_best_inclusion 0.2222 | k30_recall_owned_inclusion 0.8148 | k30_purity_best_inclusion 0.77 | k30_owned_rolled 1 | k30_best_rolled 0.7059 | k30_recall_owned_rolled 0.7059 | k30_purity_best_rolled 0.71 | k30_owned_crease 2 | k30_best_crease 0.4375 | k30_recall_owned_crease 0.5000 | k30_purity_best_crease 0.82 | k30_owned_waist 6 | k30_best_waist 0.1915 | k30_recall_owned_waist 0.9681 | k30_purity_best_waist 0.78 | k30_inertia 55369.11 | C2_ok 1 | checks_ok 1
note: regression OK

## 20261101_165324__gc10_e03_02
why: E03.02: per-class probes (lr, lr_bal, knn, pca_lr_bal) on the GC10 features fz and ft r0-r2 against the per-class recall of the pipeline (E02.05 runs); pre-registered at afdeb1d, code b89f1d6
result: lr_fz_silk_spot_k 113 | lr_fz_silk_spot_n 124 | lr_fz_silk_spot_lo 0.8481 | lr_fz_silk_spot_hi 0.9497 | lr_fz_inclusion_k 20 | lr_fz_inclusion_n 43 | lr_fz_inclusion_lo 0.3251 | lr_fz_inclusion_hi 0.6108 | lr_fz_rolled_pit_k 4 | lr_fz_rolled_pit_n 6 | lr_fz_rolled_pit_lo 0.3000 | lr_fz_rolled_pit_hi 0.9032 | lr_fz_crease_k 5 | lr_fz_crease_n 11 | lr_fz_crease_lo 0.2127 | lr_fz_crease_hi 0.7199 | lr_fz_waist_folding_k 25 | lr_fz_waist_folding_n 30 | lr_fz_waist_folding_lo 0.6644 | lr_fz_waist_folding_hi 0.9266 | lr_fz_bal_novel 0.6662 | lr_bal_fz_silk_spot_k 111 | lr_bal_fz_silk_spot_n 124 | lr_bal_fz_silk_spot_lo 0.8289 | lr_bal_fz_silk_spot_hi 0.9377 | lr_bal_fz_inclusion_k 22 | lr_bal_fz_inclusion_n 43 | lr_bal_fz_inclusion_lo 0.3675 | lr_bal_fz_inclusion_hi 0.6538 | lr_bal_fz_rolled_pit_k 4 | lr_bal_fz_rolled_pit_n 6 | lr_bal_fz_rolled_pit_lo 0.3000 | lr_bal_fz_rolled_pit_hi 0.9032 | lr_bal_fz_crease_k 5 | lr_bal_fz_crease_n 11 | lr_bal_fz_crease_lo 0.2127 | lr_bal_fz_crease_hi 0.7199 | lr_bal_fz_waist_folding_k 25 | lr_bal_fz_waist_folding_n 30 | lr_bal_fz_waist_folding_lo 0.6644 | lr_bal_fz_waist_folding_hi 0.9266 | lr_bal_fz_bal_novel 0.6723 | knn_fz_silk_spot_k 106 | knn_fz_silk_spot_n 124 | knn_fz_silk_spot_lo 0.7822 | knn_fz_silk_spot_hi 0.9062 | knn_fz_inclusion_k 13 | knn_fz_inclusion_n 43 | knn_fz_inclusion_lo 0.1860 | knn_fz_inclusion_hi 0.4511 | knn_fz_rolled_pit_k 1 | knn_fz_rolled_pit_n 6 | knn_fz_rolled_pit_lo 0.0301 | knn_fz_rolled_pit_hi 0.5635 | knn_fz_crease_k 0 | knn_fz_crease_n 11 | knn_fz_crease_lo 0.0000 | knn_fz_crease_hi 0.2588 | knn_fz_waist_folding_k 27 | knn_fz_waist_folding_n 30 | knn_fz_waist_folding_lo 0.7438 | knn_fz_waist_folding_hi 0.9654 | knn_fz_bal_novel 0.4448 | pca_lr_bal_fz_silk_spot_k 114 | pca_lr_bal_fz_silk_spot_n 124 | pca_lr_bal_fz_silk_spot_lo 0.8579 | pca_lr_bal_fz_silk_spot_hi 0.9556 | pca_lr_bal_fz_inclusion_k 18 | pca_lr_bal_fz_inclusion_n 43 | pca_lr_bal_fz_inclusion_lo 0.2838 | pca_lr_bal_fz_inclusion_hi 0.5667 | pca_lr_bal_fz_rolled_pit_k 3 | pca_lr_bal_fz_rolled_pit_n 6 | pca_lr_bal_fz_rolled_pit_lo 0.1876 | pca_lr_bal_fz_rolled_pit_hi 0.8124 | pca_lr_bal_fz_crease_k 3 | pca_lr_bal_fz_crease_n 11 | pca_lr_bal_fz_crease_lo 0.0975 | pca_lr_bal_fz_crease_hi 0.5657 | pca_lr_bal_fz_waist_folding_k 21 | pca_lr_bal_fz_waist_folding_n 30 | pca_lr_bal_fz_waist_folding_lo 0.5212 | pca_lr_bal_fz_waist_folding_hi 0.8334 | pca_lr_bal_fz_bal_novel 0.5621 | pipe_fz_silk_spot_recall 0.6645 | pipe_fz_silk_spot_lo 0.5743 | pipe_fz_silk_spot_hi 0.7386 | pipe_fz_inclusion_recall 0.1395 | pipe_fz_inclusion_lo 0.0656 | pipe_fz_inclusion_hi 0.2726 | pipe_fz_rolled_pit_recall 0.0000 | pipe_fz_rolled_pit_lo 0.0000 | pipe_fz_rolled_pit_hi 0.3903 | pipe_fz_crease_recall 0.1091 | pipe_fz_crease_lo 0.0162 | pipe_fz_crease_hi 0.3774 | pipe_fz_waist_folding_recall 0.9667 | pipe_fz_waist_folding_lo 0.8333 | pipe_fz_waist_folding_hi 0.9941 | pipe_fz_bal_novel 0.3760 | pipe_fz_runs 5 | type_fz_rolled_pit U | type_fz_crease R | type_fz_inclusion D | type_fz_silk_spot D | type_fz_waist_folding U | C4_fz_lr_all_rows_acc 80.20 | lr_r0_silk_spot_k 91 | lr_r0_silk_spot_n 124 | lr_r0_silk_spot_lo 0.6499 | lr_r0_silk_spot_hi 0.8038 | lr_r0_inclusion_k 24 | lr_r0_inclusion_n 43 | lr_r0_inclusion_lo 0.4111 | lr_r0_inclusion_hi 0.6957 | lr_r0_rolled_pit_k 3 | lr_r0_rolled_pit_n 6 | lr_r0_rolled_pit_lo 0.1876 | lr_r0_rolled_pit_hi 0.8124 | lr_r0_crease_k 5 | lr_r0_crease_n 11 | lr_r0_crease_lo 0.2127 | lr_r0_crease_hi 0.7199 | lr_r0_waist_folding_k 26 | lr_r0_waist_folding_n 30 | lr_r0_waist_folding_lo 0.7032 | lr_r0_waist_folding_hi 0.9469 | lr_r0_bal_novel 0.6226 | lr_bal_r0_silk_spot_k 84 | lr_bal_r0_silk_spot_n 124 | lr_bal_r0_silk_spot_lo 0.5909 | lr_bal_r0_silk_spot_hi 0.7533 | lr_bal_r0_inclusion_k 24 | lr_bal_r0_inclusion_n 43 | lr_bal_r0_inclusion_lo 0.4111 | lr_bal_r0_inclusion_hi 0.6957 | lr_bal_r0_rolled_pit_k 3 | lr_bal_r0_rolled_pit_n 6 | lr_bal_r0_rolled_pit_lo 0.1876 | lr_bal_r0_rolled_pit_hi 0.8124 | lr_bal_r0_crease_k 5 | lr_bal_r0_crease_n 11 | lr_bal_r0_crease_lo 0.2127 | lr_bal_r0_crease_hi 0.7199 | lr_bal_r0_waist_folding_k 26 | lr_bal_r0_waist_folding_n 30 | lr_bal_r0_waist_folding_lo 0.7032 | lr_bal_r0_waist_folding_hi 0.9469 | lr_bal_r0_bal_novel 0.6114 | knn_r0_silk_spot_k 118 | knn_r0_silk_spot_n 124 | knn_r0_silk_spot_lo 0.8984 | knn_r0_silk_spot_hi 0.9776 | knn_r0_inclusion_k 14 | knn_r0_inclusion_n 43 | knn_r0_inclusion_lo 0.2049 | knn_r0_inclusion_hi 0.4748 | knn_r0_rolled_pit_k 2 | knn_r0_rolled_pit_n 6 | knn_r0_rolled_pit_lo 0.0968 | knn_r0_rolled_pit_hi 0.7000 | knn_r0_crease_k 0 | knn_r0_crease_n 11 | knn_r0_crease_lo 0.0000 | knn_r0_crease_hi 0.2588 | knn_r0_waist_folding_k 28 | knn_r0_waist_folding_n 30 | knn_r0_waist_folding_lo 0.7868 | knn_r0_waist_folding_hi 0.9815 | knn_r0_bal_novel 0.5088 | pca_lr_bal_r0_silk_spot_k 85 | pca_lr_bal_r0_silk_spot_n 124 | pca_lr_bal_r0_silk_spot_lo 0.5992 | pca_lr_bal_r0_silk_spot_hi 0.7606 | pca_lr_bal_r0_inclusion_k 26 | pca_lr_bal_r0_inclusion_n 43 | pca_lr_bal_r0_inclusion_lo 0.4558 | pca_lr_bal_r0_inclusion_hi 0.7363 | pca_lr_bal_r0_rolled_pit_k 3 | pca_lr_bal_r0_rolled_pit_n 6 | pca_lr_bal_r0_rolled_pit_lo 0.1876 | pca_lr_bal_r0_rolled_pit_hi 0.8124 | pca_lr_bal_r0_crease_k 6 | pca_lr_bal_r0_crease_n 11 | pca_lr_bal_r0_crease_lo 0.2801 | pca_lr_bal_r0_crease_hi 0.7873 | pca_lr_bal_r0_waist_folding_k 26 | pca_lr_bal_r0_waist_folding_n 30 | pca_lr_bal_r0_waist_folding_lo 0.7032 | pca_lr_bal_r0_waist_folding_hi 0.9469 | pca_lr_bal_r0_bal_novel 0.6405 | pipe_r0_silk_spot_recall 0.6823 | pipe_r0_silk_spot_lo 0.5992 | pipe_r0_silk_spot_hi 0.7606 | pipe_r0_inclusion_recall 0.1628 | pipe_r0_inclusion_lo 0.0812 | pipe_r0_inclusion_hi 0.2997 | pipe_r0_rolled_pit_recall 0.0000 | pipe_r0_rolled_pit_lo 0.0000 | pipe_r0_rolled_pit_hi 0.3903 | pipe_r0_crease_recall 0.3455 | pipe_r0_crease_lo 0.1517 | pipe_r0_crease_hi 0.6462 | pipe_r0_waist_folding_recall 0.7800 | pipe_r0_waist_folding_lo 0.5907 | pipe_r0_waist_folding_hi 0.8821 | pipe_r0_bal_novel 0.3941 | pipe_r0_runs 5 | type_r0_rolled_pit U | type_r0_crease R | type_r0_inclusion D | type_r0_silk_spot U | type_r0_waist_folding U | lr_r1_silk_spot_k 101 | lr_r1_silk_spot_n 124 | lr_r1_silk_spot_lo 0.7370 | lr_r1_silk_spot_hi 0.8731 | lr_r1_inclusion_k 25 | lr_r1_inclusion_n 43 | lr_r1_inclusion_lo 0.4333 | lr_r1_inclusion_hi 0.7162 | lr_r1_rolled_pit_k 3 | lr_r1_rolled_pit_n 6 | lr_r1_rolled_pit_lo 0.1876 | lr_r1_rolled_pit_hi 0.8124 | lr_r1_crease_k 5 | lr_r1_crease_n 11 | lr_r1_crease_lo 0.2127 | lr_r1_crease_hi 0.7199 | lr_r1_waist_folding_k 28 | lr_r1_waist_folding_n 30 | lr_r1_waist_folding_lo 0.7868 | lr_r1_waist_folding_hi 0.9815 | lr_r1_bal_novel 0.6568 | lr_bal_r1_silk_spot_k 96 | lr_bal_r1_silk_spot_n 124 | lr_bal_r1_silk_spot_lo 0.6930 | lr_bal_r1_silk_spot_hi 0.8389 | lr_bal_r1_inclusion_k 27 | lr_bal_r1_inclusion_n 43 | lr_bal_r1_inclusion_lo 0.4786 | lr_bal_r1_inclusion_hi 0.7562 | lr_bal_r1_rolled_pit_k 3 | lr_bal_r1_rolled_pit_n 6 | lr_bal_r1_rolled_pit_lo 0.1876 | lr_bal_r1_rolled_pit_hi 0.8124 | lr_bal_r1_crease_k 6 | lr_bal_r1_crease_n 11 | lr_bal_r1_crease_lo 0.2801 | lr_bal_r1_crease_hi 0.7873 | lr_bal_r1_waist_folding_k 28 | lr_bal_r1_waist_folding_n 30 | lr_bal_r1_waist_folding_lo 0.7868 | lr_bal_r1_waist_folding_hi 0.9815 | lr_bal_r1_bal_novel 0.6762 | knn_r1_silk_spot_k 119 | knn_r1_silk_spot_n 124 | knn_r1_silk_spot_lo 0.9091 | knn_r1_silk_spot_hi 0.9827 | knn_r1_inclusion_k 14 | knn_r1_inclusion_n 43 | knn_r1_inclusion_lo 0.2049 | knn_r1_inclusion_hi 0.4748 | knn_r1_rolled_pit_k 1 | knn_r1_rolled_pit_n 6 | knn_r1_rolled_pit_lo 0.0301 | knn_r1_rolled_pit_hi 0.5635 | knn_r1_crease_k 0 | knn_r1_crease_n 11 | knn_r1_crease_lo 0.0000 | knn_r1_crease_hi 0.2588 | knn_r1_waist_folding_k 29 | knn_r1_waist_folding_n 30 | knn_r1_waist_folding_lo 0.8333 | knn_r1_waist_folding_hi 0.9941 | knn_r1_bal_novel 0.4837 | pca_lr_bal_r1_silk_spot_k 72 | pca_lr_bal_r1_silk_spot_n 124 | pca_lr_bal_r1_silk_spot_lo 0.4926 | pca_lr_bal_r1_silk_spot_hi 0.6638 | pca_lr_bal_r1_inclusion_k 21 | pca_lr_bal_r1_inclusion_n 43 | pca_lr_bal_r1_inclusion_lo 0.3462 | pca_lr_bal_r1_inclusion_hi 0.6325 | pca_lr_bal_r1_rolled_pit_k 3 | pca_lr_bal_r1_rolled_pit_n 6 | pca_lr_bal_r1_rolled_pit_lo 0.1876 | pca_lr_bal_r1_rolled_pit_hi 0.8124 | pca_lr_bal_r1_crease_k 6 | pca_lr_bal_r1_crease_n 11 | pca_lr_bal_r1_crease_lo 0.2801 | pca_lr_bal_r1_crease_hi 0.7873 | pca_lr_bal_r1_waist_folding_k 22 | pca_lr_bal_r1_waist_folding_n 30 | pca_lr_bal_r1_waist_folding_lo 0.5555 | pca_lr_bal_r1_waist_folding_hi 0.8582 | pca_lr_bal_r1_bal_novel 0.5696 | pipe_r1_silk_spot_recall 0.4210 | pipe_r1_silk_spot_lo 0.3362 | pipe_r1_silk_spot_hi 0.5074 | pipe_r1_inclusion_recall 0.0744 | pipe_r1_inclusion_lo 0.0240 | pipe_r1_inclusion_hi 0.1861 | pipe_r1_rolled_pit_recall 0.0000 | pipe_r1_rolled_pit_lo 0.0000 | pipe_r1_rolled_pit_hi 0.3903 | pipe_r1_crease_recall 0.0000 | pipe_r1_crease_lo 0.0000 | pipe_r1_crease_hi 0.2588 | pipe_r1_waist_folding_recall 0.9667 | pipe_r1_waist_folding_lo 0.8333 | pipe_r1_waist_folding_hi 0.9941 | pipe_r1_bal_novel 0.2924 | pipe_r1_runs 5 | type_r1_rolled_pit U | type_r1_crease D | type_r1_inclusion D | type_r1_silk_spot D | type_r1_waist_folding U | lr_r2_silk_spot_k 106 | lr_r2_silk_spot_n 124 | lr_r2_silk_spot_lo 0.7822 | lr_r2_silk_spot_hi 0.9062 | lr_r2_inclusion_k 23 | lr_r2_inclusion_n 43 | lr_r2_inclusion_lo 0.3892 | lr_r2_inclusion_hi 0.6749 | lr_r2_rolled_pit_k 3 | lr_r2_rolled_pit_n 6 | lr_r2_rolled_pit_lo 0.1876 | lr_r2_rolled_pit_hi 0.8124 | lr_r2_crease_k 7 | lr_r2_crease_n 11 | lr_r2_crease_lo 0.3538 | lr_r2_crease_hi 0.8483 | lr_r2_waist_folding_k 27 | lr_r2_waist_folding_n 30 | lr_r2_waist_folding_lo 0.7438 | lr_r2_waist_folding_hi 0.9654 | lr_r2_bal_novel 0.6852 | lr_bal_r2_silk_spot_k 105 | lr_bal_r2_silk_spot_n 124 | lr_bal_r2_silk_spot_lo 0.7730 | lr_bal_r2_silk_spot_hi 0.8997 | lr_bal_r2_inclusion_k 25 | lr_bal_r2_inclusion_n 43 | lr_bal_r2_inclusion_lo 0.4333 | lr_bal_r2_inclusion_hi 0.7162 | lr_bal_r2_rolled_pit_k 3 | lr_bal_r2_rolled_pit_n 6 | lr_bal_r2_rolled_pit_lo 0.1876 | lr_bal_r2_rolled_pit_hi 0.8124 | lr_bal_r2_crease_k 8 | lr_bal_r2_crease_n 11 | lr_bal_r2_crease_lo 0.4343 | lr_bal_r2_crease_hi 0.9025 | lr_bal_r2_waist_folding_k 27 | lr_bal_r2_waist_folding_n 30 | lr_bal_r2_waist_folding_lo 0.7438 | lr_bal_r2_waist_folding_hi 0.9654 | lr_bal_r2_bal_novel 0.7111 | knn_r2_silk_spot_k 118 | knn_r2_silk_spot_n 124 | knn_r2_silk_spot_lo 0.8984 | knn_r2_silk_spot_hi 0.9776 | knn_r2_inclusion_k 19 | knn_r2_inclusion_n 43 | knn_r2_inclusion_lo 0.3043 | knn_r2_inclusion_hi 0.5889 | knn_r2_rolled_pit_k 1 | knn_r2_rolled_pit_n 6 | knn_r2_rolled_pit_lo 0.0301 | knn_r2_rolled_pit_hi 0.5635 | knn_r2_crease_k 0 | knn_r2_crease_n 11 | knn_r2_crease_lo 0.0000 | knn_r2_crease_hi 0.2588 | knn_r2_waist_folding_k 28 | knn_r2_waist_folding_n 30 | knn_r2_waist_folding_lo 0.7868 | knn_r2_waist_folding_hi 0.9815 | knn_r2_bal_novel 0.4987 | pca_lr_bal_r2_silk_spot_k 83 | pca_lr_bal_r2_silk_spot_n 124 | pca_lr_bal_r2_silk_spot_lo 0.5826 | pca_lr_bal_r2_silk_spot_hi 0.7460 | pca_lr_bal_r2_inclusion_k 22 | pca_lr_bal_r2_inclusion_n 43 | pca_lr_bal_r2_inclusion_lo 0.3675 | pca_lr_bal_r2_inclusion_hi 0.6538 | pca_lr_bal_r2_rolled_pit_k 3 | pca_lr_bal_r2_rolled_pit_n 6 | pca_lr_bal_r2_rolled_pit_lo 0.1876 | pca_lr_bal_r2_rolled_pit_hi 0.8124 | pca_lr_bal_r2_crease_k 6 | pca_lr_bal_r2_crease_n 11 | pca_lr_bal_r2_crease_lo 0.2801 | pca_lr_bal_r2_crease_hi 0.7873 | pca_lr_bal_r2_waist_folding_k 25 | pca_lr_bal_r2_waist_folding_n 30 | pca_lr_bal_r2_waist_folding_lo 0.6644 | pca_lr_bal_r2_waist_folding_hi 0.9266 | pca_lr_bal_r2_bal_novel 0.6120 | pipe_r2_silk_spot_recall 0.6387 | pipe_r2_silk_spot_lo 0.5495 | pipe_r2_silk_spot_hi 0.7164 | pipe_r2_inclusion_recall 0.1814 | pipe_r2_inclusion_lo 0.0974 | pipe_r2_inclusion_hi 0.3262 | pipe_r2_rolled_pit_recall 0.0000 | pipe_r2_rolled_pit_lo 0.0000 | pipe_r2_rolled_pit_hi 0.3903 | pipe_r2_crease_recall 0.0545 | pipe_r2_crease_lo 0.0162 | pipe_r2_crease_hi 0.3774 | pipe_r2_waist_folding_recall 0.9333 | pipe_r2_waist_folding_lo 0.7868 | pipe_r2_waist_folding_hi 0.9815 | pipe_r2_bal_novel 0.3616 | pipe_r2_runs 5 | type_r2_rolled_pit U | type_r2_crease D | type_r2_inclusion D | type_r2_silk_spot D | type_r2_waist_folding U | C1_fz 1 | C3_fz 1 | C4 1 | C1_r0 1 | C3_r0 1 | C1_r1 1 | C3_r1 1 | C1_r2 1 | C3_r2 1 | checks_ok 1
note: checks_ok 1 (C1, C3 for all 4 sets; C4 fz lr on all rows = 80.2). Decision rule (lr_bal, ft r0-r2): crease D in r1 and r2 (R in r0) -> D; rolled_pit U in all three (probe 3/6 vs pipeline 0/6, Wilson intervals overlap) -> E09.06 stays the next PL experiment; no class R in 2/3 -> Q03.02/Q05.02 not raised. Prediction 'rolled_pit D' failed. Balanced novel, lr_bal vs pipeline: ft 0.61/0.68/0.71 vs 0.39/0.29/0.36; fz 0.67 vs 0.38. Details: notes/pl/experiments.vi.md E03.02
