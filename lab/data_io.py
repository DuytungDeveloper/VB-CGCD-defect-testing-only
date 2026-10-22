"""Raw data for the lab/ notebooks 00-data-* (QD23; lab/README.vi.md section 5): images() gives one table of the
images of a split file. Reads the split file only: no torch, no jax, no file written. Paths are relative to the repo
root (ROOT), as in the split files."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                  # repo root
SPLITS = {"gc10": "splits/gc10_split.json",                 # = SPLIT in dataloaders/gc10.py
          "neudet": "splits/neu_split.json"}                # read by phase 1 only (feature_extractor/dino-neudet.py)
NUM_KNOWN = {"gc10": 10, "neudet": 12}                      # old images per base class drawn into session S1:
                                                            # num_known_inc in dataloaders/gc10.py and neudet.py


def images(dataset):
    """One row per image of the split file, in feature-file order (build() in feature_extractor/dino-neudet.py:
    classes in id order, files in split order, train and test apart). Columns:
      path              image file, from the repo root
      cls, name         class id and class name
      part              train or test
      rank              position in the file list of its class and part
      row               row in the feature files of its part (features-* for train, test_features-* for test)
      role              S0, reserve, S1, unused or test (below)
      cam, coil, frame  fields of img_<cam>_<coil>_<frame>.jpg; the coil decides train or test (make_gc10_split.py)
    Roles, from n_use of the split file as _take in dataloaders/gc10.py uses it; S0 and S1 name the session the
    split file puts the image in:
      S0       base class, rank < n_use: labeled data of session S0
      reserve  base class, the rest (pool_x of dataloaders/gc10.py at the draw of session S1): each seed draws
               NUM_KNOWN images per class into session S1 and never removes them (D16)
      S1       novel class, rank < n_use: novel-class data of session S1
      unused   novel class, the rest: never used (one online session)
      test     test images: S0 test set (base classes), novel test set (novel classes)
    The pool of session S1 is pool_x of lab/replay.py (step 04), not the role S1: S1 + NUM_KNOWN images per base
    class drawn from reserve (GC10: 698 + 50 = 748)."""
    if dataset != "gc10":
        raise NotImplementedError(f"{dataset}: added with the notebook 00-data-{dataset}")
    split = json.loads((ROOT / SPLITS[dataset]).read_text())
    out = []
    for part in ("train", "test"):
        row = 0
        for cls, name in enumerate(split["classes"]):
            base, n_use = cls < split["n_base"], split["n_use"][name]
            for rank, f in enumerate(split[part][name]):
                if part == "test":
                    role = "test"
                elif base:
                    role = "S0" if rank < n_use else "reserve"
                else:
                    role = "S1" if rank < n_use else "unused"
                cam, coil, frame = Path(f).stem.split("_")[1:]
                out.append(dict(path=f"{split['root']}/{f}", cls=cls, name=name, part=part, rank=rank, row=row,
                                role=role, cam=cam, coil=coil, frame=frame))
                row += 1
    return pd.DataFrame(out)
