import json
import numpy as np
from utils import dataloader

SPLIT = "splits/gc10_split.json"


class SplitCountIncrementalLoader(dataloader.StrictPerClassIncrementalLoader):
    """StrictPerClassIncrementalLoader with PER-CLASS counts, for imbalanced data (GC10).

    n_use[c] rows of class c (in feature-file order) are labeled (base) or session data (novel);
    the rest go to the pool, from which num_known_inc old samples per class are drawn per session.
    Differences from the parent: counts per class, and labels built from the rows actually taken
    (the parent uses np.ones(fixed_count), which misaligns x and y when a class is shorter: D22).
    With equal counts the output is identical to the parent (same RNG call sequence)."""

    def __init__(self, data_dir, pretrained_model_name, base, increment, n_use, num_known_inc):
        super().__init__(data_dir=data_dir, pretrained_model_name=pretrained_model_name, base=base,
                         increment=increment, num_labeled=0, num_novel_inc=0, num_known_inc=num_known_inc)
        self.n_use = list(n_use)

    def _take(self, train_data, idx):
        rows = train_data._x[train_data._y == idx]
        n = self.n_use[int(idx)]
        return rows[:n], rows[n:]

    def _train_data_mix(self):
        pool_x = pool_y = None
        dataset_per_stage, seen_classes = [], None
        for stage_i, train_data in enumerate(self.cl.train_dataloader()):
            idx_classes = np.unique(train_data._y)
            if stage_i == 0:
                lab_x = lab_y = None
                seen_classes = idx_classes
                for idx in idx_classes:
                    use, rest = self._take(train_data, idx)
                    lab_x = self._concatenate_data(lab_x, use)
                    lab_y = self._concatenate_data(lab_y, np.ones(len(use)) * idx)
                    pool_x = self._concatenate_data(pool_x, rest)
                    pool_y = self._concatenate_data(pool_y, np.ones(len(rest)) * idx)
                lab_x, lab_y = self._resort_data(lab_x, lab_y)
                pool_x, pool_y = self._resort_data(pool_x, pool_y)
                dataset_per_stage.append((lab_x, lab_y))
            else:
                pool_x, pool_y, known_x, known_y = self._sample_known_instance(
                    pool_x, pool_y, seen_classes, self.num_known_inc)
                nov_x = nov_y = None
                seen_classes = self._concatenate_data(seen_classes, idx_classes)
                for idx in idx_classes:
                    use, rest = self._take(train_data, idx)
                    nov_x = self._concatenate_data(nov_x, use)
                    nov_y = self._concatenate_data(nov_y, np.ones(len(use)) * idx)
                    pool_x = self._concatenate_data(pool_x, rest)
                    pool_y = self._concatenate_data(pool_y, np.ones(len(rest)) * idx)
                stage_x = self._concatenate_data(known_x, nov_x)
                stage_y = self._concatenate_data(known_y, nov_y)
                stage_x, stage_y = self._resort_data(stage_x, stage_y)
                dataset_per_stage.append((stage_x, stage_y))
                pool_x, pool_y = self._resort_data(pool_x, pool_y)
        return dataset_per_stage


class GC10Loader():
    """GC10-DET, protocol in splits/gc10_split.json (scripts/make_gc10_split.py): 5 base classes +
    ONE online session of 5 novel classes (k = 5); imbalanced, per-class counts from the split."""

    def __init__(self, args):
        self.args = args

    def makeT1Loader(self):
        base, increment, num_known_inc = 5, 5, 10        # 10 old samples/class/session ~ 5% of a base class
        split = json.load(open(SPLIT))
        n_use = [split["n_use"][c] for c in split["classes"]]
        assert (self.args.base, self.args.increment, self.args.num_classes) == (base, increment, 10), \
            "GC10 t1: run with --base 5 --increment 5 --num_classes 10"
        assert self.args.num_dim < min(n_use[:base]) or getattr(self.args, 'cov_prior_kappa', 0) > 0, \
            f"GC10: --num_dim must be < {min(n_use[:base])} (smallest labeled base class), else Sigma collapses (D21)"
        m = self.args.pretrained_model_name
        y = np.load(f"{self.args.data_dir}/{m}/labels-{m}.npy").astype(int)
        assert np.bincount(y).tolist() == [len(split["train"][c]) for c in split["classes"]], \
            "feature file does not match splits/gc10_split.json"
        loader = SplitCountIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=m, base=base,
                                             increment=increment, n_use=n_use, num_known_inc=num_known_inc)
        return (loader.train_dataloader(), loader.test_dataloader(mode='novel'),
                loader.test_dataloader(mode='old'), loader.test_dataloader(mode='all'))
