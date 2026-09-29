#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC
from abc import abstractmethod
from typing import override

import numpy as np

from sklearn.metrics import classification_report
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

class ClusteringBase(ABC):

    def __init__(self, num_classes, label_offset = 0):
        self.num_classes = num_classes
        self.label_offset = label_offset

    @abstractmethod
    def fit(self, features):
        '''
        Fit the clustering model
        '''

    @abstractmethod
    def _pre_predict(self, features):
        '''
        Predict the cluster labels with clustering model
        '''

    def predict(self, features, labels, with_known=True):
        '''
        Predict the cluster labels and align with the ground truth labels
        '''
        pred = self._pre_predict(features)
        return self._align_clusterId_labelId(pred, labels, with_known)


    def predict_raw(self, features):
        '''
        Cluster ids in fit order, no alignment and no true labels (used by --hungarian_on eval, D13).
        '''
        return self._pre_predict(features)

    def _align_clusterId_labelId(self, pred, label, with_known=True):

        self.label = label

        novel_mask = (label >= self.label_offset) == True
        print("label_offset", self.label_offset)
        #print("novel_mask", novel_mask)
        scaled_label = label - self.label_offset

        #print("scaled_label", scaled_label)
        #print("pred", np.unique(pred, return_counts=True))

        # Compute the confusion matrix
        if with_known:
            conf_matrix = confusion_matrix(y_true = scaled_label[novel_mask], y_pred = pred[novel_mask])
        else:
            conf_matrix = confusion_matrix(y_true = scaled_label, y_pred = pred)

        # Use the Hungarian algorithm to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(-conf_matrix)

        #print("row_ind", row_ind)
        #print("col_ind", col_ind)

        # Create a new array to hold the matched cluster IDs
        self.matched_clusters = np.zeros_like(pred)

        # Assign the matched cluster IDs
        for i, j in zip(row_ind, col_ind):
            self.matched_clusters[pred == j] = i

        self.matched_clusters += self.label_offset

        return self.matched_clusters


    def evaluate(self):
        # Generate a classification report
        report = classification_report(self.label, self.matched_clusters)
        return report


def hungarian_map(cluster_ids, true_ids, k):
    '''
    Best one-to-one map cluster id -> true id, both in 0..k-1 (Hungarian on the confusion matrix).
    Called at EVALUATION time with test labels (D13 fix); never on unlabeled training data.
    '''
    cm = confusion_matrix(y_true=true_ids, y_pred=cluster_ids, labels=np.arange(k))
    rows, cols = linear_sum_assignment(-cm)
    return {int(c): int(r) for r, c in zip(rows, cols)}
