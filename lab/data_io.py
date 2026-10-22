"""Raw data for the lab/ notebooks 00-data-* (QD23; lab/README.vi.md section 5): images() gives one table of the
images of a split file, objects() the defect boxes of one image from its XML file, diagnostics() the checks of the
split file computed again. Reads the split file and the XML files only: no torch, no jax, no file written. Paths are
relative to the repo root (ROOT), as in the split files."""
import functools, json
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]                  # repo root
SPLITS = {"gc10": "splits/gc10_split.json",                 # = SPLIT in dataloaders/gc10.py
          "neudet": "splits/neu_split.json"}                # read by phase 1 only (feature_extractor/dino-neudet.py)
NUM_KNOWN = {"gc10": 10, "neudet": 12}                      # old images per base class drawn into session S1:
                                                            # num_known_inc in dataloaders/gc10.py and neudet.py
NORM = {"10_yaozhed": "10_yaozhe"}                          # misspelled XML name: NORM in scripts/make_gc10_split.py


@functools.lru_cache(maxsize=None)
def _split(dataset):
    """The split file, read once per process; callers must not change it."""
    return json.loads((ROOT / SPLITS[dataset]).read_text())


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
    split = _split(dataset)
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


def _num(node, tag):
    """Float of a child element of an XML node; nan when the node or its text is missing."""
    text = node.findtext(tag) if node is not None else None
    return float(text) if text and text.strip() else float("nan")


def objects(dataset, path):
    """Defect boxes of one image, read from its XML file as scripts/make_gc10_split.py reads it: the name 10_yaozhed
    becomes 10_yaozhe (NORM), objects named d are dropped. path is a path of images(dataset). None when the image has
    no XML file. One dict per box:
      name                    defect name in the XML, after NORM (e.g. 1_chongkong)
      cls                     class id of that name (folder number before "_"), None if it names no class
      xmin, ymin, xmax, ymax  box in pixels of the original image
      W, H                    image size written in the XML
    Missing numbers are nan."""
    if dataset != "gc10":
        raise NotImplementedError(f"{dataset}: added with the notebook 00-data-{dataset}")
    p = Path(path)
    xml = ROOT / p.parent.parent / "label" / f"{p.stem}.xml"         # <root of the dataset>/label/<stem>.xml
    if not xml.parent.is_dir():
        raise FileNotFoundError(f"{xml.parent}: no XML folder (GC10-DET must be under datasets/)")
    if not xml.exists():
        return None
    root = ET.parse(xml).getroot()
    W, H = _num(root, "size/width"), _num(root, "size/height")
    cls = {folder: i for i, folder in enumerate(_split(dataset)["folders"])}
    out = []
    for o in root.iter("object"):
        name = NORM.get(o.findtext("name"), o.findtext("name"))
        if name == "d":
            continue
        b = o.find("bndbox")
        out.append(dict(name=name, cls=cls.get(name.split("_")[0]), xmin=_num(b, "xmin"), ymin=_num(b, "ymin"),
                        xmax=_num(b, "xmax"), ymax=_num(b, "ymax"), W=W, H=H))
    return out


def diagnostics(dataset):
    """The diagnostics of the split file computed again with objects(), with the rules and the order of
    scripts/make_gc10_split.py: lists of "folder/name", by folder, then by name.
      multi_defect       more than one defect name in the XML file
      folder_not_in_xml  no defect of the image's own folder in the XML file
      no_xml             no XML file"""
    if dataset != "gc10":
        raise NotImplementedError(f"{dataset}: added with the notebook 00-data-{dataset}")
    split = _split(dataset)
    out = {"multi_defect": [], "folder_not_in_xml": [], "no_xml": []}
    for folder, name in zip(split["folders"], split["classes"]):
        for f in sorted(split["train"][name] + split["test"][name]):
            objs = objects(dataset, f"{split['root']}/{f}")
            if objs is None:
                out["no_xml"].append(f)
                continue
            names = {o["name"] for o in objs}
            if len(names) > 1:
                out["multi_defect"].append(f)
            if not any(n.split("_")[0] == folder for n in names):
                out["folder_not_in_xml"].append(f)
    return out
