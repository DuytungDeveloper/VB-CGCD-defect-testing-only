#!/usr/bin/env python
"""Unit test for classifier/closed_form.py (B1). Run from the repo root: python scripts/test_closed_form.py"""
import sys; sys.path.insert(0, ".")
import numpy as np
from sklearn.covariance import empirical_covariance
from classifier.closed_form import closed_form_params

rng = np.random.default_rng(0)
d, C = 8, 6
A = rng.normal(size=(d, d))
X = np.concatenate([rng.normal(size=(n, d)) @ A + 3 * k for k, n in [(0, 50), (1, 5), (2, 1)]])
y = np.repeat([0, 1, 2], [50, 5, 1])

# S0-like call: estimate classes 0, 1, 2; slots 3..5 must stay at (0, I)
p, info = closed_form_params(X, y, classes=[0, 1, 2], num_classes=C)
for k in (0, 1):
    Xk = X[y == k]; S = empirical_covariance(Xk); a = info[k][1]
    assert np.allclose(p["class_covs"][k], (1 - a) * S + a * np.trace(S) / d * np.eye(d)), k   # = report formula
    assert np.allclose(p["class_means"][k], Xk.mean(0)), k
c2 = p["class_covs"][2]
assert np.allclose(c2, np.trace(c2) / d * np.eye(d)) and np.allclose(p["class_means"][2], X[y == 2][0])  # n = 1
assert all(np.linalg.eigvalsh(p["class_covs"][k]).min() > 0 for k in range(3))                        # PD, also n = 5 < d
assert np.array_equal(p["class_means"][3:], np.zeros((3, d)))
assert np.array_equal(p["class_covs"][3:], np.tile(np.eye(d), (3, 1, 1)))

# S1-like call: old slots 0..2 copied bit for bit; class 4 has no samples -> keeps prev
X1 = rng.normal(size=(70, d)) + 1
y1 = np.repeat([3, 5], [40, 30])
q, info1 = closed_form_params(X1, y1, classes=[3, 4, 5], prev=p)
assert np.array_equal(q["class_means"][:3], p["class_means"][:3])
assert np.array_equal(q["class_covs"][:3], p["class_covs"][:3])
assert np.array_equal(q["class_covs"][4], p["class_covs"][4]) and info1[4][0] == 0

for k, (n, a, kap) in {**info, **info1}.items():
    print(f"class {k}: n={n} alpha={a} kappa_hat={kap}")
print("B1 ok")
