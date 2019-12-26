# RF
from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)
from sklearn.utils.multiclass import check_classification_targets

from scipy.stats import entropy
from joblib import Parallel, delayed
import numpy as np


class UncertaintyForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        max_depth=40,
        min_samples_leaf=None,  # k
        max_features=None,  # m
        n_estimators=300,  # B
        max_samples=None,  # s // 2
        bootstrap=False,
        parallel=True,
        finite_correction=True,
        base=2.0,
    ):

        # Tree parameters.
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap

        # Model parameters.
        self.parallel = parallel
        self.finite_correction = finite_correction
        self.base = base

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_ = X
        self.y_ = self._preprocess_y(y)
        n = X.shape[0]
        d = X.shape[1]

        if not self.max_features:
            self.max_features = int(np.floor(np.sqrt(d)))

        if not self.min_samples_leaf:
            self.min_samples_leaf = int(np.ceil(0.15 * np.sqrt(n)))

        if not self.max_samples:
            self.max_samples = int(np.ceil(0.5 * (n ** 0.95)))

        # 'max_samples' determines the number of 'structure' data points
        # that will be used to learn each tree.
        self.model = BaggingClassifier(
            DecisionTreeClassifier(  # max_depth = self.max_depth,
                max_depth=self.max.depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
        )

        self.model.fit(X, y)

        # Precompute entropy of y to use later for mutual info.
        _, counts = np.unique(y, return_counts=True)
        self.entropy = entropy(counts, base=self.base)

        self.fitted = True
        return self

    def _get_leaves(self, tree):

        # TO DO: Check this tutorial.
        # adapted from
        # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html

        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        leaf_ids = []
        stack = [(0, -1)]
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()

            # If we have a test node
            if children_left[node_id] != children_right[node_id]:
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                leaf_ids.append(node_id)

        return np.array(leaf_ids)

    def _finite_sample_correct(self, posterior_per_leaf, n_per_leaf):

        num_leaves = posterior_per_leaf.shape[0]
        K = posterior_per_leaf.shape[1]
        ret = np.zeros(posterior_per_leaf.shape)
        for i in range(num_leaves):
            leaf = posterior_per_leaf[i, :]
            c = np.divide(K - np.count_nonzero(leaf), 2 * K * n_per_leaf[i])

            ret[i, leaf == 0.0] = np.divide(1, 2 * K * n_per_leaf[i])
            ret[i, leaf != 0.0] = (1 - c) * posterior_per_leaf[i, leaf != 0.0]

        return ret

    def _preprocess_y(self, y):
        # Chance y values to be indices between 0 and K (number of classes).
        classes = np.unique(y)
        K = len(classes)
        n = len(y)

        class_to_index = {}
        for k in range(K):
            class_to_index[classes[k]] = k

        ret = np.zeros(n)
        for i in range(n):
            ret[i] = class_to_index[y[i]]

        return ret.astype(int)

    def predict(self, X):

        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):

        try:
            self.fitted
        except AttributeError:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        X = check_array(X)
        n, d_ = self.X_.shape
        v, d = X.shape

        if d != d_:
            raise ValueError(
                "Training and evaluation data must have the same number of dimensions."
            )

        def worker(tree):
            # Get indices of estimation set, i.e. those NOT used
            # in learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)

            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
            node_counts = tree.tree_.n_node_samples
            leaf_nodes = self._get_leaves(tree)
            unique_leaf_nodes = np.unique(leaf_nodes)
            class_counts_per_leaf = np.zeros(
                (len(unique_leaf_nodes), self.model.n_classes_)
            )

            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(self.X_[i].reshape((1, -1))).item()
                class_counts_per_leaf[
                    np.where(unique_leaf_nodes == temp_node)[0][0], self.y_[i]
                ] += 1

            # Count the number of data points in each leaf in.
            n_per_leaf = class_counts_per_leaf.sum(axis=1)
            n_per_leaf[n_per_leaf == 0] = 1  # Avoid divide by zero.

            # Posterior probability distributions in each leaf.
            # Each row is length num_classes.
            posterior_per_leaf = np.divide(
                class_counts_per_leaf,
                np.repeat(n_per_leaf.reshape((-1, 1)), self.model.n_classes_, axis=1),
            )
            if self.finite_correction:
                posterior_per_leaf = self._finite_sample_correct(
                    posterior_per_leaf, n_per_leaf
                )
            posterior_per_leaf.tolist()

            # Posterior probability for each element of the evaluation set.
            eval_posteriors = [
                posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]]
                for node in tree.apply(X)
            ]
            eval_posteriors = np.array(eval_posteriors)

            # Number of estimation points in the cell of each eval point.
            n_per_eval_leaf = np.asarray(
                [
                    node_counts[np.where(unique_leaf_nodes == x)[0][0]]
                    for x in tree.apply(X)
                ]
            )

            class_count_increment = np.multiply(
                eval_posteriors,
                np.repeat(
                    n_per_eval_leaf.reshape((-1, 1)), self.model.n_classes_, axis=1
                ),
            )
            return class_count_increment

        if self.parallel:
            class_counts = np.array(
                Parallel(n_jobs=-2)(delayed(worker)(tree) for tree in self.model)
            )
            class_counts = np.sum(class_counts, axis=0)
        else:
            class_counts = np.zeros((v, self.model.n_classes_))
            for tree in self.model:
                class_counts += worker(tree)

        # Normalize counts.
        norm_constant = class_counts.sum(axis=1)
        norm_constant[norm_constant == 0] = 1  # Avoid divide by zero.
        return np.divide(
            class_counts,
            np.repeat(norm_constant.reshape((-1, 1)), self.model.n_classes_, axis=1),
        )

    def estimate_cond_entropy(self, X):

        p = self.predict_proba(X)
        return np.mean(entropy(p.T, base=self.base))

    def estimate_mutual_info(self, X):

        return self.entropy - self.estimate_cond_entropy(X)
