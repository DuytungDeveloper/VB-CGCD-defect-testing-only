# LOG

One entry per run, appended by exp run. Fill in "why" and "note".
Before the clean rerun: archive/pre-rerun-20260915/ (not in git) and git tag pre-rerun-20260915.

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
- Are paper numbers inside the 5-seed spread?
- Are re-extracted features identical to the archived ones? (GPU determinism)
- Does adding the contrastive term (paper Eq. 3) change S0 / M_d?
- How well does a frozen DINO backbone separate novel defect classes on NEU-DET / GC10-DET?

---
