# Paper vs code discrepancies — VB-CGCD
Line numbers refer to this fork. For files changed by patches, a search string is given instead.
D1 and D2 are not used; numbers are never reused.

## D3 (CONFIRMED). Eq. 2 and Eq. 3 are NOT implemented
Code, feature_extractor/sl_finetuned_model.py:103-105, the entire training step:
    cls_loss = F.cross_entropy(output, batch["labels"].to(device))
    loss = cls_loss / accum        (accum added by PATCH 2)
No contrastive loss, no lambda, no temperature, no augmented view pairs.
Paper also gives no values: lambda appears only in Eq. 3 and Algorithm 1, tau only in Eq. 2.
Any implementation must choose them, which is itself a deviation.

Training transform is deterministic (Resize -> CenterCrop -> ToTensor -> Normalize):
no RandomResizedCrop, no flip, no color jitter. Consistent with the absence of
contrastive learning, which requires two augmented views per image.

Contradicts:
- Eq. 2 (L_con, InfoNCE) — not present.
- Eq. 3 (L_ft = (1-lambda)L_ce + lambda*L_con) — only L_ce present.
- Sec. 3.3 core component (1): "Self-supervised learning based offline fine-tuning"
  — fine-tuning is purely supervised.
- Sec. 3.2.1: "learn label-independent features ... through techniques such as data
  augmentation" — no augmentation applied during fine-tuning.
- Table 1 credits "Offline Fine-Tuning" with addressing Label Bias; with plain CE the
  claimed mechanism (label-independent features) is absent.

Impact: one of four claimed core components does not exist as described. The Table 5
"w/o fine-tuning" ablation removes fine-tuning wholesale and therefore cannot isolate
the contribution of the contrastive term (which was never there).

## D4. Random seed: phase 1 is always 42, paper claims 0
paper:   Appendix C.3: "All random seeds are set to 0."
phase 1: sl_finetuned_model.py:32 torch.manual_seed(42) runs at import time.
         dino-cifar100.py:83 skips seeding when --seed is 0 -> seed 0 is impossible via CLI.
phase 2: main.py:87 --seed default 0; main.py:103-105 seeds torch, numpy, jax key.
         BUT mngmm.py:110 SVI init uses random.PRNGKey(0) hard-coded -> --seed never reaches SVI init.
impact:  phase-1 seed cannot match the paper; phase-2 seed only changes data sampling and GMM init.

## D5. Clustering: paper k-means, code GMM
paper:  Appendix C.3: "We employed the standard k-means algorithm for clustering"
code:   main.py:89 --clustering_alg default 'gmm'; main.py:32-36 only 'gmm' is supported.
        clustering/gmm.py:14 sklearn GaussianMixture (default full covariance, random_state=None).
impact: pseudo-label quality in S1-S5 -> New acc and M_d. New acc varies a lot across seeds
        (old 3-seed run: S1_New 67.9-80.7).

## D6. Classifier learning rate: paper fixed 1e-5, code warmup + decay from 4e-6
paper:  Appendix C.3: "a fixed learning rate of 1e-5 was applied across all stages, with 1000 update steps"
code:   main.py:96 --lr default 4e-6.
        mngmm.py:96-97 linear 4e-6 -> 4e-5 over 100 steps, then exponential decay x0.85 every 500 steps.
impact: how far class means / covariances move from init (see open question: Sigma close to I?).

## D7. Covariance storage: paper lower triangle, code full matrix
paper:  Appendix C.6: "only the lower triangular part needs to be stored"
code:   mngmm.py:298-299 saves full class_means and class_covs (100 x 384 x 384 float32, ~59 MB per stage).
impact: storage only, not accuracy. Weak discrepancy: C.6 describes what is possible.

## D8. Backbone weights: paper TorchHub, code Hugging Face on the fine-tuned path
paper:  Appendix C.3: "DINO pre-trained ViT-B/16 model with weights from TorchHub"
code:   with --finetuned: sl_finetuned_model.py, search 'AutoModel.from_pretrained' (facebook/dino-vitb16).
        TorchHub is used only WITHOUT --finetuned: dino-cifar100.py, search "torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')".
        Features are taken from the HF pooler output (dense + tanh); the pooler is not part of DINO,
        is randomly initialised and stays frozen (confirmed in an earlier session).
impact: extracted features differ from the TorchHub DINO CLS token -> all numbers.

## D9. Fine-tuning data: 500 images/class, CL labeled set only 400/class
paper:  Sec 3.2 and Fig. 1: fine-tune the backbone "on initial labeled data" (D_l).
code:   dino-cifar100.py, search 'x["labels"] < 50': all 500 train images of each base class (25,000).
        CL loader: dataloaders/cifar100.py:12 num_labeled=20000 -> utils/dataloader.py:138,157 keep 400/class;
        the other 100/class go to the unlabeled pool of later sessions.
impact: the backbone has seen labels of samples that phase 2 treats as unlabeled (train split only; test not involved).
status: NOT FIXED. The fix exists (FT_LABELED_IDX=splits/c100_labeled_hf_idx.npy) but has not been
        run yet (planned tag c100_extract_base400). As of 2026-09-29.

## D10. PCA fit data: not stated in paper, code fits once on S0 labeled train features
paper:  Appendix C.3: PCA reduces output features to 384 dims. Does not say which data PCA is fit on.
code:   mngmm.py:170-176 pre_processing: first call (pca is None) -> fit_transform; later calls -> transform.
        Call order at stage 0: main.py:128 run() -> mngmm.py:275 train() -> mngmm.py:180 pre_processing(train);
        all test() calls come later (mngmm.py:279-291). So PCA is fit on S0 labeled train features, then frozen.
        sklearn 1.5.2 auto solver for 20000x768, 384 comps = covariance_eigh (checked 2026-09-17) -> deterministic.
impact: PCA is not a source of seed variance. The 384-d space is built from base classes only;
        new classes may lose variance. Matters more for datasets with few base classes (NEU-DET, GC10-DET).

## D11. No variational inference: mu and Sigma are plain parameters, the guide is empty
paper:  Eq. 7 is an ELBO with a N(0, I) prior and a KL term; Sec 3.2.2 argues the KL penalty
        down-weights outliers. Sec 3.2.3 and the method name (VB = Variational Bayes) rest on this.
code:   classifier/mngmm.py model(): class_means and class_covs are numpyro.param, NOT numpyro.sample.
        The guide is empty: search 'self.guide = lambda'. With no latent sample sites,
        Trace_ELBO reduces to the negative log-likelihood -> plain stochastic MLE by Adam.
        No prior, no KL. N(0, I) survives only as the INITIAL value.
        AutoMultivariateNormal is imported but never used.
impact: the claimed mechanism (KL regularisation) does not exist. Table 5's "w/o SVI" ablation
        therefore compares two point estimates (closed form Eq. 5 vs gradient descent with early
        stopping), not Bayesian vs non-Bayesian. Sigma is also an unconstrained parameter, so
        positive-definiteness is not enforced.
measured: on c100_base_s4 checkpoints, learned Sigma stays close to its I initialisation for every
        probed class: trace/d ~0.98, ||S - I||_F ~1.8-2.4 (||I||_F = 19.6), eigenvalues in
        [0.87, 1.79], condition number ~1.8. Consistent with D6 (lr 4e-6 + decay) and with MLE
        from a near-I start. [probe.py ckpt]

## D12. "Re-labeling" in the code is a novelty filter, not re-labeling
paper:  Algorithm 1 lines 34-39: after the first SVI pass, the classifier RE-LABELS the unlabeled
        data (y-bar) and SVI is run again on those new labels, over all samples.
code:   mngmm.py train(): F-pass runs SVI on the GMM pseudo-labels; then
        novel_idx = pred_labels >= self.label_offset selects the samples PREDICTED as novel; then
        labels = labels[novel_idx] keeps the ORIGINAL GMM pseudo-labels of those samples; the
        S-pass trains on that filtered set.
        So (a) the classifier's new predictions are used only as a mask, never as labels, and
        (b) samples predicted as old classes are DROPPED instead of updating the old classes.
impact: old-class parameters receive no gradient in the S-pass. Measured: for probed classes
        0, 1, 49 the stored mu and Sigma are byte-identical across S0-S5, and for 50, 59 across
        S1-S5 (4-decimal stats, probe.py ckpt on c100_base_s4). Novelty counts stay near 4000
        of 5250-6250 per session, so roughly a quarter of each session's samples are discarded.

## D13. Hungarian matching uses the TRUE labels of the unlabeled data
paper:  never mentions it. In the C-GCD protocol, sessions S1-S5 are unlabeled by definition.
code:   main.py: clustering.predict(train_data._x, train_data._y, with_known=True) passes the
        ground-truth labels of the unlabeled session data.
        clustering/clustering_base.py:44 novel_mask = (label >= self.label_offset);
        :47 scaled_label = label - self.label_offset; :54-56 confusion_matrix(y_true=scaled_label...);
        :59 linear_sum_assignment(-conf_matrix). The resulting permutation names the clusters,
        and those names become the pseudo-labels used for training.
scope:  this is the ONLY place where true labels of unlabeled data enter training (verified by
        grep over clustering_base.py and mngmm.py). Early stopping does NOT use test labels:
        mngmm.py:116-118 stops on covariance determinants and step count only; the test accuracy at
        :124 and :147 is computed after the decision and only logged.
impact: the mapping is one-to-one over the 10 new ids, so it cannot make a cluster purer, but the
        standard GCD protocol computes this alignment at EVALUATION time on the test set, not
        during training. Planned fix: --hungarian_on train|eval, assign ids by cluster order during
        training, compute the mapping once per session at evaluation and freeze it.
status: NOT FIXED, not implemented. Next step, before NEU-DET. As of 2026-09-29.

## D14. Early-stopping rule matches neither Eq. 11 nor Algorithm 1; --early_stop_ratio is dead
paper:  Eq. 11 stops when R < epsilon = 0.01; Algorithm 1 uses det(Sigma_i) >= avg(det(Sigma));
        Appendix C.4 treats R as a hyper-parameter and sweeps it (Tables 11-13).
code:   mngmm.py calculate_metrics_on_covariances compares the MEAN det of the new classes with
        the MEAN det of the old ones, scaled by sqrt(n/(total+n)) in the S-pass or
        (label_offset+increment)/num_base in the F-pass, with the comparison direction depending on
        whether dets[0] > 1. It is active only after step > num_steps/3 (the if at mngmm.py:118, dets from :116) and rolls
        back one step. --early_stop_ratio (main.py) is parsed and never read.
impact: Tables 11-13 (the R sweep) cannot be reproduced with this code. Measured: S-pass stop step
        falls across sessions in every seed (752 -> 707 -> 704 -> 651 -> 597 at lambda 0, GMM),
        while the F-pass stays flat near 800. Explanation that fits the data: old mean det < 1
        (logdet ~ -10), so the active branch is det_new < det_old / sf, and sf shrinks with
        label_offset, raising the threshold. No underflow. [INFERENCE, per-step dets not logged]

## D15. M_f is computed on a proxy set, not on D_l as defined
paper:  Appendix C.2 defines M_f (forgetting) over the initially labeled dataset D_l.
code:   scripts/extract_vbcgcd.py takes M_f = accuracy on the S0 TEST set at stage 0 minus the same
        accuracy at the final stage (the "S0" row of the log, known_test), not accuracy on D_l.
impact: all M_f numbers in this repo are that proxy. Weak evidence that the paper does the same:
        our 5-seed S5_S0test mean is 84.74 and the paper's 91.68 - 6.94 = 84.74 exactly. One
        coincidence on one dataset; not proof.

## D16. The old-class sample pool is never consumed: samples repeat across sessions
paper:  Table 8 specifies 25 samples per known class per online session.
code:   utils/dataloader.py:261-278 _sample_known_instance computes remained_x / remained_y but
        RETURNS x, y (the whole shuffled pool). The pool is therefore never reduced, and each
        session draws its 25 samples per class from the full 100.
measured: base classes need 25 x 5 = 125 samples but only 100 exist, so repetition is unavoidable.
        Reused samples across S1-S5: 2441 (seed 0) and 2458 (seed 4) of 6250; the with-replacement
        expectation is 6250 - 50 x 100 x (1 - 0.75^5) = 2437. Fixing the return value would make
        S5 run out of samples, so the current behaviour is required by Table 8's numbers.
impact: undocumented sample reuse in every online session.

## D17. The labeled split depends on the extraction batch size
code:   the CL loader takes the first 400 ROWS of each base class in the merged .npy
        (utils/dataloader.py:309), and merge_npy (feature_extractor/dino-cifar100.py:44-45) sorts
        shard files as STRINGS: labels_0, labels_1, labels_10, labels_100, ... So the merged row
        order is a permutation P of the HF dataset order, determined by the extraction batch size
        (128 here) and by DataLoader(shuffle=False).
verified: 391 shards; every shard equals hf[128*b : 128*b+128]; P is a permutation; hf[P] equals
        all 50000 merged labels. The resulting labeled set (20000 HF indices) overlaps the naive
        "first 400 per class in HF order" in only 15752 of 20000 cases (scripts/make_labeled_idx.py).
impact: changing --batch-size at extraction silently changes which images phase 2 treats as
        labeled. Keep 128. splits/c100_labeled_hf_idx.npy stores the 20000 HF indices so that
        phase 1 can be restricted to exactly that set (the D9 fix), and the folds in
        feature_extractor/ft_data.py are defined on the same index order.

## D18. Clustering runs once (n_init=1); restarts matter more than the algorithm
paper:  Appendix C.3 says standard k-means. Sec 3.2.1 / Alg. 1 only say "clustering".
code:   clustering/gmm.py GaussianMixture(n_components, random_state), all defaults:
        covariance_type='full', n_init=1, reg_covar=1e-6. No k-means in the repo.

2x2 design, 5 seeds per cell, SAME features (c100_extract), only the clustering changed:

                    n_init=1                      n_init=10
  GMM         final 79.86 +- 1.68  (c100_base)  final 81.35 +- 0.77  (c100_gmm10)
              M_d   76.51 +- 3.00               M_d   79.11 +- 1.11
  k-means     final 79.86 +- 1.68  (c100_km1)   final 81.76 +- 0.10  (c100_km)
              M_d   76.51 +- 3.00               M_d   79.88 +- 0.23
  paper: final 81.23, M_d 79.14, M_f 6.94, S1_New 83.10

main effects:
  restarts    +1.49 (GMM) / +1.90 (k-means) on final_All; +2.60 / +3.37 on M_d. Consistent.
  algorithm   +0.00 at n_init=1, +0.41 at n_init=10. About 4x smaller. No real interaction.
  stability   here the algorithm dominates: std of final_All 0.10 (k-means,10) vs 0.77 (GMM,10)
              vs 1.68 (either, n_init=1). Likely because GMM picks by likelihood while k-means
              picks by inertia, which is tied to cluster geometry. [INFERENCE]
  control     S0_All is 91.90 +- 0.07 in all four cells (S0 uses no clustering), as expected.

c100_km1 reproduces c100_base EXACTLY on all 43 metrics. On the real S1 data (5250 imgs,
384 dims after PCA) GMM and k-means partitions differ by only 1-2 of 5250 samples, so the
two are effectively the same clustering when each runs once.

WITHDRAWN: an earlier version of this entry claimed GMM(full) is degenerate in 384 dims
(bimodal 81 / 94.8 accuracy, ELBO vs accuracy r = -0.999). That was measured on validation
fold F5 only (1-epoch features, 40 classes, 4000 novel-only images) and does NOT transfer to
the real S1 mix. Kept here as a warning: do not generalise from the fold-F5 setup.

impact: best config is k-means with n_init=10, i.e. what C.3 describes plus 10 restarts:
        final_All 81.76 vs paper 81.23, M_d 79.88 vs 79.14, M_f 6.61 vs 6.94 (lower is better),
        and the seed spread nearly vanishes. Remaining gap: S1_New 80.28 vs paper 83.10.
code added: clustering/kmeans.py (KMeansCluster), main.py Clustering_alg('kmeans'),
        --kmeans_n_init (default 10) and --gmm_n_init (default 1 = upstream).
        Regression check: --gmm_n_init default reproduces c100_base_s0 exactly (c100_regress_s0).
open:   c100_gmm10 has S4_New 87.02 (4 of 5 seeds 87.7-88.7) but the lowest S5_New 71.82.
        Possible cross-session trade-off; r(S4_New, S5_New) = -0.55, n = 5, not significant.

## D19. Eq. 2 / Eq. 3 (contrastive fine-tuning) is described but not implemented, and adding it does not help
paper:  Eq. 2 defines an InfoNCE-style L_con; Eq. 3 combines it with cross-entropy,
        L_ft = (1-lambda)*L_ce + lambda*L_con. Algorithm 1 line 15 uses the same formula.
        NOT given anywhere in the paper: lambda, tau, the augmentation, whether a projection
        head is used, the anchor set, or whether the positive is in the denominator of Eq. 2.
code:   feature_extractor/sl_finetuned_model.py has only F.cross_entropy. No L_con, no lambda,
        no tau, no augmentation (the fine-tuning transform is Resize + CenterCrop, deterministic).
        This is D3; D19 is what happens when Eq. 3 IS implemented.

our implementation (all choices are ours, the paper underspecifies Eq. 2):
        NT-Xent (SimCLR Eq. 1): 2N anchors, symmetric, denominator = all other 2N-1 views.
        Taken literally, Eq. 2 gives a NEGATIVE loss (positive missing from the denominator)
        and only N-1 negatives per anchor.
        L_con on the L2-normalised pooler_output, NO projection head (Eq. 2 uses the same h(.)
        as Eq. 4 and Algorithm 1). Augmentation: RandomResizedCrop(224, scale 0.5-1.0) +
        HorizontalFlip + ColorJitter(0.2,0.2,0.2,0.05) p=0.8; CE on view 1.
        Batch 16 x 2 views x accum 8 (32 x 2 views = 7.68 GB > 6.93 GB available). Effective
        batch 128, same as upstream, but only 30 NEGATIVES PER ANCHOR.
        Control: c100_ft16 (lambda 0, batch 16x8) = 81.73 vs c100_km (32x4) = 81.76, so the
        batch change alone is not the cause of anything below.

statistical unit: ONE phase-1 run (FT_SEED). Its value = mean of 5 phase-2 seeds (k-means,
        n_init 10). The 5 phase-2 seeds share one feature set and are NOT independent units.
        Phase-1 noise at lambda=0, n=5 (FT_SEED 42-46): sigma = 0.09 (S0_All), 0.11 (final_All),
        0.23 (M_d), 0.73 (S1_New). Every comparison below uses these as the error bars.

lambda axis (tau = 0.2, 500 imgs/class, 10 epochs):
          lambda 0     n=5  final 81.84+-0.11   M_d 80.04+-0.23
          lambda 0.1   n=3  final 81.94+-0.36   M_d 79.91+-0.25
          lambda 0.25  n=3  final 81.71+-0.22   M_d 79.76+-0.29
          lambda 0.5   n=1  final 81.15         M_d 79.31
        slope final_All -1.17 per unit lambda (p = 0.027), M_d -1.35 (p = 0.011), 12 phase-1 runs.

tau axis (lambda = 0.5, n=1 unless noted), with the final CON value from the log:
          tau 0.03        CON 0.0008   S0 92.13         final 82.06
          tau 0.07  n=3   CON 0.005    S0 92.12+-0.05   final 82.02+-0.22
          tau 0.12        CON  -       S0 91.88         final 81.91
          tau 0.2         CON 0.34     S0 91.29         final 81.15
          tau 0.5         CON  -       S0 91.18         final 80.26
          tau 5.0         CON 3.27     S0 92.13         final 81.92
        Range on the tau axis is 1.87 points of final_All vs 0.79 on the lambda axis, and the two
        interact: the whole lambda sweep above was run at tau = 0.2, which is a harmful setting.

why tau matters: it is a SWITCH, not a tunable weight. Chance level for 30 negatives is
        log(31) = 3.43. At tau <= 0.07 the contrastive task is solved (CON ~ 0, no gradient left);
        at tau = 5.0 it is flat (CON 3.27, no gradient either). Both extremes disable L_con and
        both give S0 ~ 92.1. In between (tau 0.12-0.5) L_con is actually active, and every such
        configuration is WORSE than the lambda = 0 baseline. So the contrastive term helps in
        exactly the regimes where it does nothing.

isolating augmentation (FT_AUG=1 with lambda=0: augmentation, 2 forwards, no L_con, CE weight 1.0):
                              n   S0_All          p       final_All       p
          lambda 0 (no aug)   5   91.84+-0.09     -       81.84+-0.11     -
          aug only            3   92.09+-0.26   0.229     81.90+-0.40   0.827
          lambda 0.5 tau 0.07 3   92.12+-0.05   0.001     82.02+-0.22   0.298
        Welch t-test vs the lambda=0 arm.

conclusions:
        1. No configuration with an ACTIVE contrastive term beats the baseline. Larger lambda at
           tau = 0.2 hurts systematically (slope p = 0.027).
        2. The gain that does appear (~ +0.25 on S0_All) comes from the augmentation, not from
           L_con: it is reproduced with L_con disabled at both tau extremes and with FT_AUG=1.
        3. Augmentation alone is UNSTABLE across phase-1 seeds: sigma 0.26 vs 0.05 for the same
           augmentation plus a near-zero L_con at tau = 0.07, on 3 seeds each (92.20 / 92.28 /
           91.79 vs 92.10 / 92.18 / 92.09). Only the latter reaches p < 0.05. A near-zero L_con
           may act as a mild stabiliser; with n = 3 this is a hypothesis, not a result. [INFERENCE]
        4. No configuration improves the continual metrics (final_All, M_d, S1_New) significantly.
           Reaching 80% power on final_All at the observed effect sizes needs ~14 phase-1 runs per
           arm (> 100 h on this machine).
        5. Taken together: the paper's reported numbers are very unlikely to depend on Eq. 3.
           The released code reaches them with plain cross-entropy (S0 91.90 vs paper 91.68).

limits: one dataset; tau swept at full scale only at lambda = 0.5; 30 negatives per anchor, so a
        null result cannot separate "Eq. 3 does not help" from "too few negatives"; no projection
        head (deliberate, to follow Eq. 2 literally) - the SimCLR-style variant with a head is
        untested; the validation grid on fold F5 (3 epochs, 400 imgs/class) did NOT predict these
        full-scale results and should not be used as a proxy again.

runs:   c100_extract_con*, c100_con*, c100_ft16*, c100_augonly*, f5_* (validation grid),
        c100_con0_probe / c100_consmoke (1-epoch probe), c100_augcheck + c100_seedcheck
        (regression checks that FT_AUG and FT_SEED defaults reproduce upstream).
knobs:  FT_LAMBDA (0), FT_TAU (0.2), FT_AUG ("" = follow FT_LAMBDA), FT_SEED (42), all default
        to upstream behaviour.
## D20. Paper-internal inconsistencies (paper vs itself, not vs code)
Checked against the PMLR PDF, 2026-09-29.
1. Tables 11-13 column headers say S0-S4, but the R=0 rows equal Table 2 S1-S5 for C100:
   All 88.40/86.0/83.61/82.72/81.23, Old 89.46/87.06/84.75/82.85/81.84, New 83.10/79.60/75.60/81.70/75.70.
   The headers are shifted by one session.
2. Table 6 column b50-All(S5) disagrees with Table 2 in two cells: C100 VB-CGCD 81.32 vs 81.23
   (also 81.32 in Table 20), IN100 Happy 75.58 vs 78.58. Tiny and CUB match. The "9.56% vs 22.87%
   drop" claim in Sec 4.2 depends on these cells.
3. Sec 3.2.2 calls N(0, I) a "conjugate prior, ensuring closed-form updates". For the mean and
   covariance of an MVN the conjugate prior is Normal-Inverse-Wishart (e.g. Murphy 2012, Sec 4.6.3).
   Moot for the code: there is no prior at all (D11).
4. Not an inconsistency, evidence for D11: Fig. 5 shows VI covariances in [1.0075, 1.0175] vs
   PE covariances in [3, 6], i.e. the paper's own "VI" estimate barely leaves its I initialisation.
impact: Tables 11-13 must be read as S1-S5; quote Table 2, not Table 6, for b50 numbers.

## Additional notes on the fine-tuning implementation
- BOFT config declared: boft_block_size=4, boft_n_butterfly_factor=2, boft_dropout=0.1,
  bias="boft_only", target_modules=["output.dense","mlp.fc1","mlp.fc2"].
- Only PARTLY matches "updating only the MLP and output dense layers":
  "mlp.fc1" / "mlp.fc2" are timm names; they do not exist in the Hugging Face ViT model.
  "output.dense" matches attention.output.dense (12) and ViTOutput.dense (12, the 2nd MLP layer).
  intermediate.dense (the 1st MLP layer) is NOT adapted.
  Verified: trainable params 221,184 decompose exactly into these 24 modules
  (reproducibility-issues.md A5); adapting intermediate.dense would give a larger count.
- Butterfly factor actually runs as 1, not 2: see reproducibility-issues.md A5 (environment issue).
- optimizer: AdamW, lr=1e-3, 10 epochs — matches Appendix C.3.
- MultiCropWrapper docstring describes multi-resolution forward passes, but forward()
  performs a single pass. Vestigial code copied from the DINO repository.

