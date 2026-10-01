#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/10/20
# @Author  : Hao Dai

import os
import sys
import time
import argparse
import functools

import numpy as np

import torch

import jax

from dataloaders.cifar100 import CIFAR100Loader
from dataloaders.tinyimagenet import TinyImageNetLoader
from dataloaders.imagenet100 import ImageNet100Loader
from dataloaders.cub200 import CUB200Loader
from dataloaders.neudet import NEUDETLoader
from dataloaders.gc10 import GC10Loader

from classifier.mngmm import MNGMMClassifier

from clustering.gmm import GMMCluster
from clustering.kmeans import KMeansCluster
from clustering.clustering_base import hungarian_map
from types import SimpleNamespace

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# get the current time with a format yyyyMMdd-HHmm
def get_current_time():
    return time.strftime("%Y%m%d-%H%M", time.localtime())

def Clustering_alg(alg):
    if alg == 'gmm':
        return GMMCluster
    elif alg == 'kmeans':
        return KMeansCluster
    else:
        raise ValueError('Clustering algorithm not supported')

def Classifier_alg(alg):
    if alg == 'mngmm':
        return MNGMMClassifier
    else:
        raise ValueError('Classifier algorithm not supported')

def load_mode(args, loader):
    if args.load_mode == 't5':
        return loader.makeT5Loader()
    elif args.load_mode == 'vin':
        return loader.makeVinLoader()
    elif args.load_mode == 't10':
        return loader.makeT10Loader()
    elif args.load_mode == 't1':
        return loader.makeT1Loader()
    else:
        raise ValueError('Load mode not supported')

def Dataloader(args):
    # make data loader
    if args.dataset == 'cifar100':
        cifar100loader = CIFAR100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args,cifar100loader)

    elif args.dataset == 'tinyimagenet':
        tinyimagenetloader = TinyImageNetLoader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, tinyimagenetloader)

    elif args.dataset == 'imagenet100':
        imageNet100Loader = ImageNet100Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, imageNet100Loader)

    elif args.dataset == 'cub200':
        cub200Loader = CUB200Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, cub200Loader)

    elif args.dataset == 'neudet':
        neudetLoader = NEUDETLoader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, neudetLoader)

    elif args.dataset == 'gc10':
        gc10Loader = GC10Loader(args = args)
        train_loader, test_loader, test_old_loader, test_all_loader = load_mode(args, gc10Loader)

    else:
        raise ValueError('Dataset not supported')

    return train_loader, test_loader, test_old_loader, test_all_loader


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Generalized Class Incremental Learning')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to learn')
    parser.add_argument('--data_dir', type=str, default='datasets/cifar100', help='Directory to the data')
    parser.add_argument('--load_mode', type=str, default='t5', help='Dataset Loader Mode (t5 / t10 / vin / t1)')
    parser.add_argument('--pretrained_model_name', type=str, default='dino-vitb16', help='Name of the model')
    parser.add_argument('--base', type=int, default=50, help='Number of base classes')
    parser.add_argument('--increment', type=int, default=10, help='Number of incremental classes')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--trail_name', type=str, default=f'', help='Name of the trail')
    parser.add_argument('--clustering_alg', type=str, default='gmm', help='Clustering algorithm')
    parser.add_argument('--hungarian_on', type=str, default='train', choices=['train', 'eval'],
                        help="D13: 'train' = upstream, true labels name the clusters; 'eval' = cluster order in training, test-set Hungarian for evaluation")
    parser.add_argument('--print_confusion', action='store_true',
                        help='diagnostic: print the test_all confusion matrix (true ids) after each online stage')
    parser.add_argument('--s0_labels', type=str, default='true', choices=['true', 'none'],
                        help="'none' = label-free S0: cluster the base data (k = --base), Gaussians learn cluster ids; "
                             "test-set Hungarian for evaluation only (needs --hungarian_on eval)")
    parser.add_argument('--cov_prior_kappa', type=float, default=0.0,
                        help='improvement 1: inverse-Wishart shrinkage of class covariances toward the pooled '
                             'within-class covariance, kappa pseudo-samples (0 = MLE as upstream)')
    parser.add_argument('--kmeans_n_init', type=int, default=10,
                        help='k-means restarts (kept in config.json by exp.py; ignored for gmm)')
    parser.add_argument('--gmm_n_init', type=int, default=1,
                        help='GMM restarts; 1 = upstream default (ignored for kmeans)')
    parser.add_argument('--classifier_alg', type=str, default='mngmm', help='Classifier algorithm')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of classes for the classifier')
    parser.add_argument('--num_dim', type=int, default=384, help='Number of features\' dim for the classifier')
    parser.add_argument('--with_early_stop', default=True, action=argparse.BooleanOptionalAction, help='Whether to use early stop')
    parser.add_argument('--use_correct_scaling_factor', default=True, action=argparse.BooleanOptionalAction, help='Whether to use correct scaling factor')
    parser.add_argument("--n_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=4e-6, help="Learning rate")
    parser.add_argument("--scaling-factor", type=float, default=1.2, help="Scaling factor for learning from arbitary")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--early_stop_ratio", type=float, default=0, help="R in early stop")
    args = parser.parse_args()
    assert not (args.s0_labels == 'none' and args.hungarian_on != 'eval'), '--s0_labels none needs --hungarian_on eval'

    # Set the random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng_key = jax.random.PRNGKey(args.seed)

    Clustering = Clustering_alg(args.clustering_alg)
    if args.clustering_alg == 'kmeans':
        Clustering = functools.partial(Clustering, n_init=args.kmeans_n_init)
    elif args.clustering_alg == 'gmm':
        Clustering = functools.partial(Clustering, n_init=args.gmm_n_init)

    Classifier = Classifier_alg(args.classifier_alg)

    train_loader, test_loader, test_old_loader, test_all_loader = Dataloader(args)

    # if saved_models dir does not exist, create it
    log_saved_dir = f"{args.trail_name}_{get_current_time()}"
    if not os.path.exists(f"logs/{log_saved_dir}/saved_models"):
        os.makedirs(f"logs/{log_saved_dir}/saved_models")

    # same classifier for all stages, static expension
    s_classifier = Classifier(num_classes=args.num_classes, num_dim=args.num_dim, with_early_stop=args.with_early_stop)

    print(f"scaling factor: {args.scaling_factor}")
    s_classifier.cov_prior_kappa = args.cov_prior_kappa   # improvement 1 (0 = MLE, upstream)
    s_classifier.init_parameters(n_epochs=args.n_epochs, lr=args.lr, log_dir=f"logs/{log_saved_dir}/log/stage0", save_dir=f"logs/{log_saved_dir}/saved_models/stage0", batch_size=args.batch_size, increment=args.increment, base=args.base, scaling_factor=args.scaling_factor, use_correct_scaling_factor=args.use_correct_scaling_factor, early_stop_ratio=args.early_stop_ratio)

    for i, (train_data, test_data, test_old_data, test_all_data) in enumerate(zip(train_loader, test_loader, test_old_loader, test_all_loader)): 
        if i == 0:
            to_cluster = {}   # D13 eval mode: true id -> cluster-space id, filled at each discovery session
            y0 = train_data._y
            if args.s0_labels == 'none':         # label-free S0: base data clustered, no label used in training
                clustering0 = Clustering(num_classes=args.base, label_offset=0)
                clustering0.fit(train_data._x)
                y0 = clustering0.predict_raw(train_data._x)
                pi0 = hungarian_map(clustering0.predict_raw(test_data._x),
                                    np.asarray(test_data._y).astype(int), args.base)    # evaluation only
                to_cluster.update({t: c for c, t in pi0.items()})
                lut0 = np.arange(args.num_classes)
                for t, c in to_cluster.items():
                    lut0[t] = c
                test_data = SimpleNamespace(_x=test_data._x, _y=lut0[np.asarray(test_data._y).astype(int)])
                print(f"s0_labels none: pi_test={pi0} cluster sizes={np.bincount(y0, minlength=args.base).tolist()}")
            testing_set = {'test_old': test_data, 'test_all': test_data, 'known_test': test_data}

            s_classifier.run(train_data._x, y0, test_data._x, test_data._y, current_stage=i, testing_set=testing_set)

            known_test_data = test_data

        else:

            label_offset = args.base + (i-1)*args.increment

            clustering = Clustering(num_classes=args.increment, label_offset=args.base + (i-1)*args.increment)

            print(args.increment, (i-1)*args.increment)

            clustering.fit(train_data._x)

            raw = clustering.predict_raw(train_data._x)
            pi_test = hungarian_map(clustering.predict_raw(test_data._x),
                                    np.asarray(test_data._y).astype(int) - label_offset, args.increment)
            if args.hungarian_on == 'train':     # upstream: TRUE train labels name the clusters (D13)
                pred = clustering.predict(train_data._x, train_data._y, with_known=True)
                pi_train = {int(j): int(pred[raw == j][0]) - label_offset for j in np.unique(raw)}
                print(f"hungarian stage {i}: mode=train pi_train={pi_train} pi_test={pi_test} same={pi_train == pi_test}")
            else:                                # D13 fix: cluster order in training, test-set Hungarian for evaluation
                pred = raw + label_offset
                to_cluster.update({label_offset + t: label_offset + c for c, t in pi_test.items()})
                print(f"hungarian stage {i}: mode=eval pi_test={pi_test}")

            print(np.unique(pred, return_counts=True))

            # for ngmm merge
            s_classifier._set_label_offset(label_offset)

            s_classifier.update_dir_infos(log_dir=f"logs/{log_saved_dir}/log/stage{i}", save_dir=f"logs/{log_saved_dir}/saved_models/stage{i}")

            if args.hungarian_on == 'eval':      # relabel COPIES of test labels into cluster space
                lut = np.arange(args.num_classes)
                for t, c in to_cluster.items():
                    lut[t] = c
                relabel = lambda d: SimpleNamespace(_x=d._x, _y=lut[np.asarray(d._y).astype(int)])
                test_novel, test_old, test_all = relabel(test_data), relabel(test_old_data), relabel(test_all_data)
            else:
                test_novel, test_old, test_all = test_data, test_old_data, test_all_data

            testing_set = {'test_old': test_old, 'test_all': test_all, 'known_test': known_test_data}

            s_classifier.run(train_data._x, pred, test_novel._x, test_novel._y, current_stage=i, testing_set=testing_set)
            if args.print_confusion:             # diagnostic only, after the metrics table
                from sklearn.metrics import confusion_matrix
                inv = np.argsort(lut) if args.hungarian_on == 'eval' else np.arange(args.num_classes)
                tf, _ = s_classifier.pre_processing(test_all._x, test_all._y)
                pr, _ = s_classifier._predict(jax.numpy.array(tf), s_classifier.params)
                cm = confusion_matrix(np.asarray(test_all_data._y).astype(int), inv[np.asarray(pr)],
                                      labels=np.arange(label_offset + args.increment))
                print(f"confusion stage {i}: rows=true cols=pred {cm.tolist()}")
