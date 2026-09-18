#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Paper Appendix C.3 says "standard k-means"; the released code only has GMM (D5).
# On c100_consmoke/F5 (4000 imgs, 384 dims) k-means reached 94.7% cluster accuracy while
# GMM(full) gave a bimodal 81 / 94.8 and its ELBO anti-correlated with accuracy (r = -0.999):
# 10 full covariances = 739k parameters for 4000 samples (D18). Same interface as GMMCluster.

from typing import override
from sklearn.cluster import KMeans
from .clustering_base import ClusteringBase


class KMeansCluster(ClusteringBase):

    def __init__(self, num_classes, label_offset=0, random_state=None, n_init=10):
        super().__init__(num_classes, label_offset)

        self.random_state = random_state
        self.n_init = n_init
        self.model = KMeans(n_clusters=num_classes, n_init=n_init, random_state=random_state)

    @override
    def fit(self, features):
        self.model.fit(features)

    @override
    def _pre_predict(self, features):
        return self.model.predict(features)
