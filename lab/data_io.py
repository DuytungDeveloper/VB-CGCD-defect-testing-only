"""Raw data for the lab/ notebooks 00-data-* (QD23; lab/README.vi.md section 5): images() gives one table of the
images of a split file, objects() the defect boxes of one image from its XML file, diagnostics() the label checks of
scripts/make_gc10_split.py (GC10: equal to its split file). Datasets: gc10, neudet. Reads the split file and the XML
files only: no torch, no jax, no file written. Paths are relative to the repo root (ROOT), as in the split files."""
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
NEU = {"n_base": 3, "n_use": 192}                           # not in splits/neu_split.json: base and num_labeled // base
                                                            # = num_novel_inc in dataloaders/neudet.py (192 = 0.8 * 240)


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
      cam, coil, frame  GC10 only: fields of img_<cam>_<coil>_<frame>.jpg; the coil decides train or test
                        (make_gc10_split.py)
    Roles, from n_base and n_use (GC10: its split file, as _take in dataloaders/gc10.py uses it; NEU: NEU above, as
    dataloaders/neudet.py); S0 and S1 name the session the split file puts the image in:
      S0       base class, rank < n_use: labeled data of session S0
      reserve  base class, the rest: the loader's pool at the draw of session S1 (pool_x of dataloaders/gc10.py;
               unlabeled_x_pool of utils/dataloader.py for NEU): each seed draws NUM_KNOWN images per class into
               session S1 and never removes them (D16)
      S1       novel class, rank < n_use: novel-class data of session S1
      unused   novel class, the rest: never used (one online session)
      test     test images: S0 test set (base classes), novel test set (novel classes)
    The pool of session S1 is pool_x of lab/replay.py (step 04), not the role S1: S1 + NUM_KNOWN images per base
    class drawn from reserve (GC10: 698 + 50 = 748; NEU: 576 + 36 = 612)."""
    split, gc10 = _split(dataset), dataset == "gc10"
    n_base = split["n_base"] if gc10 else NEU["n_base"]
    out = []
    for part in ("train", "test"):
        row = 0
        sub = "train" if part == "train" else "validation"         # NEU folders, as build() in dino-neudet.py
        for cls, name in enumerate(split["classes"]):
            base, n_use = cls < n_base, split["n_use"][name] if gc10 else NEU["n_use"]
            for rank, f in enumerate(split[part][name]):
                if part == "test":
                    role = "test"
                elif base:
                    role = "S0" if rank < n_use else "reserve"
                else:
                    role = "S1" if rank < n_use else "unused"
                path = f"{split['root']}/{f}" if gc10 else f"{split['root']}/{sub}/images/{name}/{f}"
                rec = dict(path=path, cls=cls, name=name, part=part, rank=rank, row=row, role=role)
                if gc10:
                    cam, coil, frame = Path(f).stem.split("_")[1:]
                    rec.update(cam=cam, coil=coil, frame=frame)
                out.append(rec)
                row += 1
    return pd.DataFrame(out)


def _num(node, tag):
    """Float of a child element of an XML node; nan when the node or its text is missing."""
    text = node.findtext(tag) if node is not None else None
    return float(text) if text and text.strip() else float("nan")


@functools.lru_cache(maxsize=None)
def _neu_xml():
    """NEU-DET XML files by file stem, from both annotation folders: the XML of the train image crazing_240 sits in
    validation/annotations. Stems are unique across train and validation (scripts/make_neu_split.py)."""
    root = _split("neudet")["root"]
    out = {}
    for x in sorted((ROOT / root).glob("*/annotations/*.xml")):
        assert x.stem not in out, f"{x.stem}: two XML files"
        out[x.stem] = x
    if not out:
        raise FileNotFoundError(f"{root}: no XML files (NEU-DET must be under datasets/)")
    return out


def objects(dataset, path):
    """Defect boxes of one image, read from its XML file. GC10 as scripts/make_gc10_split.py reads it: the name
    10_yaozhed becomes 10_yaozhe (NORM), objects named d are dropped. NEU: the XML is found by file stem (_neu_xml).
    path is a path of images(dataset). None when the image has no XML file. One dict per box:
      name                    defect name in the XML, after NORM (e.g. 1_chongkong; NEU: a class name)
      cls                     class id of that name (GC10: folder number before "_"; NEU: the class name itself),
                              None if it names no class
      xmin, ymin, xmax, ymax  box in pixels of the original image
      W, H                    image size written in the XML
    Missing numbers are nan."""
    split, gc10, p = _split(dataset), dataset == "gc10", Path(path)
    if gc10:
        xml = ROOT / p.parent.parent / "label" / f"{p.stem}.xml"     # <root of the dataset>/label/<stem>.xml
        if not xml.parent.is_dir():
            raise FileNotFoundError(f"{xml.parent}: no XML folder (GC10-DET must be under datasets/)")
        xml = xml if xml.exists() else None
    else:
        xml = _neu_xml().get(p.stem)
    if xml is None:
        return None
    root = ET.parse(xml).getroot()
    W, H = _num(root, "size/width"), _num(root, "size/height")
    ids = {k: i for i, k in enumerate(split["folders"] if gc10 else split["classes"])}
    out = []
    for o in root.iter("object"):
        name = o.findtext("name")
        if gc10:
            name = NORM.get(name, name)
            if name == "d":
                continue
        b = o.find("bndbox")
        out.append(dict(name=name, cls=ids.get(name.split("_")[0] if gc10 else name), xmin=_num(b, "xmin"),
                        ymin=_num(b, "ymin"), xmax=_num(b, "xmax"), ymax=_num(b, "ymax"), W=W, H=H))
    return out


def diagnostics(dataset):
    """Label checks of scripts/make_gc10_split.py, computed with objects(): lists of keys "folder/name" (GC10, as in its
    split file, whose diagnostics this equals) or "class/name" (NEU), by class, then by key.
      multi_defect       more than one defect name in the XML file
      folder_not_in_xml  no defect of the image's own class (folder) in the XML file
      no_xml             no XML file"""
    df = images(dataset)
    keys = ["/".join(Path(p).parts[-2:]) for p in df["path"]]       # GC10: the file name in its split file
    out = {"multi_defect": [], "folder_not_in_xml": [], "no_xml": []}
    for cls, key, path in sorted(zip(df["cls"], keys, df["path"])):
        objs = objects(dataset, path)
        if objs is None:
            out["no_xml"].append(key)
            continue
        if len({o["name"] for o in objs}) > 1:
            out["multi_defect"].append(key)
        if not any(o["cls"] == cls for o in objs):
            out["folder_not_in_xml"].append(key)
    return out
