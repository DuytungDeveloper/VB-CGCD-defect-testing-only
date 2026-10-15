# VB-CGCD reproduction (fork)

Reproduction of VB-CGCD on CIFAR-100, with notes on where the code differs from the paper.
Then industrial surface defect data: GC10-DET (main benchmark) and NEU-DET (sanity checks).
Current work: pseudo-label quality with minimal real labels (notes/pl/, lab/).

Paper:     Dai & Chauhan, "Continual Generalized Category Discovery: Learning and Forgetting
           from a Bayesian Perspective", ICML 2025.
           https://proceedings.mlr.press/v267/dai25a.html
           arXiv: https://arxiv.org/abs/2507.17382
Upstream:  https://github.com/daihao42/VB-CGCD (its README declares MIT; the upstream repo has no LICENSE file).

## 1. What is where

    scripts/exp.py                     runs any command and records it in runs/<run_id>/
    scripts/run_phase1_extract.sh      phase 1: fine-tune the backbone, extract features
    scripts/check_phase1.py            checks the phase 1 output
    scripts/run_phase2_cl.sh           phase 2: continual learning (main.py)
    scripts/extract_vbcgcd.py          reads the phase 2 log, writes metrics.json

    features/<run_id>/                 phase 1 output: features + fine-tune checkpoint   (not in git)
    logs/<run_id>_<minute>/            phase 2 output: checkpoints + TensorBoard          (not in git)
    runs/<run_id>/                     config.json, metrics.json, console.log (log not in git)
    runs/INDEX.txt                     one line per run, written automatically
    runs/inputs.txt                    feature folder that phase 2 reads
    runs/columns.txt                   metrics shown in INDEX.txt and in the LOG
    runs/paper/                        paper numbers, shown as a reference row
    notes/LOG.md                       one entry per run (automatic)
    notes/reproducibility-issues.md    environment problems and local patches
    notes/paper-code-discrepancies.md  where the code differs from the paper
    notes/code-changes.vi.md           what the fork changed vs upstream (Vietnamese)
    notes/plan.vi.md                   repo rules; plan, decisions, questions up to 2026-10-03 (Vietnamese)
    notes/pl/README.vi.md              current work (pseudo-labels, PL): scope, codes, decisions (Vietnamese)
    notes/pl/questions.vi.md           open questions of the PL work (Vietnamese)
    notes/pl/experiments.vi.md         PL experiments: preregistration, results (Vietnamese)
    lab/README.vi.md                   index of lab/: notebooks per pipeline step, helper .py (Vietnamese)
    archive/pre-rerun-20260915/        runs before the clean rerun; removed 2026-10-15, packed in
                                       /mnt/e/AI_Storage/backups/vbcgcd-archive-pre-rerun-20260915.tar

## 2. Environment

Tested on WSL2 Ubuntu 22.04, RTX 4060 Laptop 8 GB, Python 3.12.
Do not install from requirements.txt: it does not work (see notes, A1 and A2).

    conda create -n vbcgcd python=3.12
    conda activate vbcgcd
    pip install -r requirements-lock.txt

For the notebooks in lab/ only (Jupyter kernel; runs do not need it):

    pip install ipykernel==7.4.0

Check that JAX uses the GPU. It must print [CudaDevice(id=0)]:

    python -c "import jax; print(jax.devices())"

CIFAR-100 and the DINO weights are downloaded by Hugging Face on first use (~/.cache/huggingface).

## 3. Rules

- Run everything through scripts/exp.py, inside tmux.
- Tags: <dataset>_<purpose>[_<variant>][_s<N>], only [a-z0-9_], never "__".
  Examples: c100_extract, c100_base_s0, c100_base_lr1e5_s0. exp agg groups <name>_s<N> exactly.
- Commit code before a run. exp.py records the commit; uncommitted code shows dirty = yes.
- Phase 1 takes about 2 h 50 min on the RTX 4060 (about 2 h 35 min with FT_LABELED_IDX, about 5 h 30 min with 2-view augmentation). Before it: pause Windows Update,
  disable sleep, close apps that use the GPU on Windows.

## 4. Phase 1: features (once per feature set)

    tmux new -s extract
    conda activate vbcgcd
    cd ~/projects/VB-CGCD-defect
    python scripts/exp.py run c100_extract --inputs - -- bash scripts/run_phase1_extract.sh

Clean protocol, fine-tune only on the labeled set (D9):

    FT_LABELED_IDX=splits/c100_labeled_hf_idx.npy python scripts/exp.py run c100_extract_base400 \
        --inputs splits/c100_labeled_hf_idx.npy -- bash scripts/run_phase1_extract.sh

Output: features/<run_id>/dino-vitb16-sl/*.npy and features/<run_id>/ft_ckpt.pt.
The run ends with check_phase1.py, which must print "OK: 0 failed check(s)".

Then tell phase 2 which features to use (one line), and commit it:

    echo "features/<run_id>/dino-vitb16-sl" > runs/inputs.txt

## 5. Phase 2: continual learning

Smoke test, 5 steps per stage, about 2 min:

    python scripts/exp.py run c100_smoke -- bash scripts/run_phase2_cl.sh --clustering_alg kmeans --hungarian_on eval --n_epochs 5

Five seeds:

    tmux new -s cl
    conda activate vbcgcd
    cd ~/projects/VB-CGCD-defect
    for s in 0 1 2 3 4; do
      python scripts/exp.py run c100_km_evhung_s$s -- bash scripts/run_phase2_cl.sh --clustering_alg kmeans --hungarian_on eval --seed $s
    done

Arguments after the script name override its defaults, for example --seed or --n_epochs.
The wrapper default is upstream (GMM, n_init 1). The best configuration is k-means with
--kmeans_n_init 10 (the default when --clustering_alg kmeans), see D18.
Add --hungarian_on eval so that no true label of the unlabeled sessions enters training (D13).

NEU-DET (b3t1: 3 base + 1 session of 3 novel classes; protocol in splits/neu_split.json):

    python scripts/make_neu_split.py      # once; a second run must print "unchanged"
    python scripts/exp.py run neu_extract --inputs splits/neu_split.json -- bash scripts/run_phase1_neu.sh
    echo "features/<neu_extract run_id>/dino-vitb16-sl" > runs/inputs.txt
    for s in 0 1 2 3 4; do
      python scripts/exp.py run neu_km_evhung_d128_s$s -- bash scripts/run_phase2_cl.sh \
        --dataset neudet --base 3 --increment 3 --num_classes 6 --load_mode t1 \
        --clustering_alg kmeans --hungarian_on eval --num_dim 128 --seed $s
    done

--num_dim must stay below the 192 labeled images per class, else Sigma collapses (D21).
Data: datasets/NEU-DET (train / validation). Phase 1 ~5 min, each phase-2 run ~2.5 min.

GC10-DET (b5t1: 5 base + 1 session of 5 novel classes, imbalanced; protocol in splits/gc10_split.json):

    python scripts/make_gc10_split.py      # once; coil-grouped test split, 24 duplicate files removed
    python scripts/exp.py run gc10_extract_frozen --inputs splits/gc10_split.json -- \
      bash scripts/run_phase1_neu.sh --split splits/gc10_split.json --resize full --no_finetune
    python scripts/exp.py run gc10_extract --inputs splits/gc10_split.json -- \
      bash scripts/run_phase1_neu.sh --split splits/gc10_split.json --resize full    # with fine-tuning
    echo "features/<run_id>/<model>" > runs/inputs.txt   # model: dino-vitb16 (frozen) or dino-vitb16-sl
    for s in 0 1 2 3 4; do
      python scripts/exp.py run <tag>_s$s -- bash scripts/run_phase2_cl.sh \
        --dataset gc10 --base 5 --increment 5 --num_classes 10 --load_mode t1 \
        --clustering_alg kmeans --hungarian_on eval --num_dim 64 --seed $s --print_confusion
    done                                   # frozen features: add --pretrained_model_name dino-vitb16
    python scripts/cm_summary.py <tag>     # balanced accuracy, per-class recall, largest confusions
    python scripts/feat_probe.py features/<run_id>/<model>   # diagnostic probes of a feature set

--num_dim must stay below the smallest labeled base class (130 images), else Sigma collapses (D21).

## 6. Results

    python scripts/exp.py table            # prints and writes runs/INDEX.txt
    python scripts/exp.py agg c100_km_evhung   # mean/std over c100_km_evhung_s* -> runs/agg_c100_km_evhung.txt

After each run, fill in "why" and "note" in notes/LOG.md, then:

    git add runs notes
    git commit -m "results: <tag>"
    git push

## 7. tmux and stopping a run

    Ctrl+B then D            detach, the run keeps going
    tmux attach -t cl        come back
    Ctrl+C                   stop the run; it is kept with exit_code "interrupted"