# Paper vs code discrepancies — VB-CGCD
Line numbers refer to this fork. For files changed by patches, a search string is given instead.

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

## D10. PCA fit data: not stated in paper, code fits once on S0 labeled train features
paper:  Appendix C.3: PCA reduces output features to 384 dims. Does not say which data PCA is fit on.
code:   mngmm.py:170-176 pre_processing: first call (pca is None) -> fit_transform; later calls -> transform.
        Call order at stage 0: main.py:128 run() -> mngmm.py:275 train() -> mngmm.py:180 pre_processing(train);
        all test() calls come later (mngmm.py:279-291). So PCA is fit on S0 labeled train features, then frozen.
        sklearn 1.5.2 auto solver for 20000x768, 384 comps = covariance_eigh (checked 2026-09-17) -> deterministic.
impact: PCA is not a source of seed variance. The 384-d space is built from base classes only;
        new classes may lose variance. Matters more for datasets with few base classes (NEU-DET, GC10-DET).