# VB-CGCD reproduction (fork)

Reproduction of VB-CGCD on CIFAR-100, with notes on where the code differs from the paper.
Planned next: industrial surface defect data (NEU-DET, GC10-DET).

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
    notes/LOG.md                       one entry per run (automatic) + open questions
    notes/reproducibility-issues.md    environment problems and local patches
    notes/paper-code-discrepancies.md  where the code differs from the paper
    archive/                           old runs                                           (not in git)

## 2. Environment

Tested on WSL2 Ubuntu 22.04, RTX 4060 Laptop 8 GB, Python 3.12.
Do not install from requirements.txt: it does not work (see notes, A1 and A2).

    conda create -n vbcgcd python=3.12
    conda activate vbcgcd
    pip install -r requirements-lock.txt

Check that JAX uses the GPU. It must print [CudaDevice(id=0)]:

    python -c "import jax; print(jax.devices())"

CIFAR-100 and the DINO weights are downloaded by Hugging Face on first use (~/.cache/huggingface).

## 3. Rules

- Run everything through scripts/exp.py, inside tmux.
- Tags: <dataset>_<purpose>[_<variant>][_s<N>], only [a-z0-9_], never "__".
  Examples: c100_extract, c100_base_s0, c100_base_lr1e5_s0. exp agg groups <name>_s<N> exactly.
- Commit code before a run. exp.py records the commit; uncommitted code shows dirty = yes.
- Phase 1 takes about 3 h 15 min on the RTX 4060. Before it: pause Windows Update,
  disable sleep, close apps that use the GPU on Windows.

## 4. Phase 1: features (once per feature set)

    tmux new -s extract
    conda activate vbcgcd
    cd ~/projects/VB-CGCD-defect
    python scripts/exp.py run c100_extract --inputs - -- bash scripts/run_phase1_extract.sh

Output: features/<run_id>/dino-vitb16-sl/*.npy and features/<run_id>/ft_ckpt.pt.
The run ends with check_phase1.py, which must print "OK: 0 failed check(s)".

Then tell phase 2 which features to use (one line), and commit it:

    echo "features/<run_id>/dino-vitb16-sl" > runs/inputs.txt

## 5. Phase 2: continual learning

Smoke test, 5 steps per stage, about 2 min:

    python scripts/exp.py run c100_smoke -- bash scripts/run_phase2_cl.sh --n_epochs 5

Five seeds:

    tmux new -s cl
    conda activate vbcgcd
    cd ~/projects/VB-CGCD-defect
    for s in 0 1 2 3 4; do
      python scripts/exp.py run c100_base_s$s -- bash scripts/run_phase2_cl.sh --seed $s
    done

Arguments after the script name override its defaults, for example --seed or --n_epochs.

## 6. Results

    python scripts/exp.py table            # prints and writes runs/INDEX.txt
    python scripts/exp.py agg c100_base    # mean/std over c100_base_s* -> runs/agg_c100_base.txt
Tags: <dataset>_<purpose>[_<variant>][_s<N>], only [a-z0-9_], never "__". Example: c100_base_lr1e5_s0.

After each run, fill in "why" and "note" in notes/LOG.md, then:

    git add runs notes
    git commit -m "results: <tag>"
    git push

## 7. tmux and stopping a run

    Ctrl+B then D            detach, the run keeps going
    tmux attach -t cl        come back
    Ctrl+C                   stop the run; it is kept with exit_code "interrupted"