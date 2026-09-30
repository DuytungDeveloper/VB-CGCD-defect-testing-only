
from utils import dataloader


class NEUDETLoader():
    """NEU-DET, protocol in splits/neu_split.json (scripts/make_neu_split.py): b3t1 =
    3 base classes + ONE online session of 3 novel classes. Per class, the 240 train rows are in
    split order: first 192 = labeled (base) or session data (novel), last 48 = pool.
    12 old-class samples per class per session = 5% of 240, same ratio as 25/500 on CIFAR-100."""

    def __init__(self, args):
        self.args = args

    def makeT1Loader(self):
        base, increment = 3, 3
        num_labeled, num_novel_inc, num_known_inc = 3 * 192, 192, 12
        # main.py derives label_offset from args.base / args.increment: they must match
        assert (self.args.base, self.args.increment, self.args.num_classes) == (base, increment, 6), \
            "NEU t1: run with --base 3 --increment 3 --num_classes 6"
        assert self.args.num_dim < 192, \
            "NEU t1: --num_dim must be < 192 labeled samples per class, else the MLE of Sigma collapses (D21)"

        loader = dataloader.StrictPerClassIncrementalLoader(data_dir=self.args.data_dir, pretrained_model_name=self.args.pretrained_model_name, base=base, increment=increment, num_labeled=num_labeled, num_novel_inc=num_novel_inc, num_known_inc=num_known_inc)

        train_loader = loader.train_dataloader()
        test_all_loader = loader.test_dataloader(mode='all')
        test_novel_loader = loader.test_dataloader(mode='novel')
        test_old_loader = loader.test_dataloader(mode='old')

        return train_loader, test_novel_loader, test_old_loader, test_all_loader
