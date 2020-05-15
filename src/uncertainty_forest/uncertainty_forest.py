# RF
# from sklearn.ensemble.forest import _generate_unsampled_indices
# from sklearn.ensemble._bagging import _generate_indices
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
    # check_random_state,
)
from sklearn.utils.multiclass import check_classification_targets

from scipy.stats import entropy
from joblib import Parallel, delayed
import numpy as np


class UncertaintyForest(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        min_samples_leaf=1,
        max_features=None,
        n_estimators=1500,
        max_samples=0.4,
        kappa=3,
        base=np.exp(1),
    ):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.kappa = kappa
        self.base = base

    # def uf(X, y, n_estimators=300, max_samples=0.4, base=np.exp(1), kappa=3):

    #     # Build forest with default parameters.
    #     model = BaggingClassifier(
    #         DecisionTreeClassifier(),
    #         n_estimators=n_estimators,
    #         max_samples=max_samples,
    #         bootstrap=False,
    #     )
    #     model.fit(X, y)
    #     n = X.shape[0]
    #     K = model.n_classes_

    #     cond_entropy = 0
    #     for tree_idx, tree in enumerate(model):

    #         # Find the indices of the training set used for partition.
    #         sampled_indices = model.estimators_samples_[tree_idx]
    #         unsampled_indices = np.delete(np.arange(0, n), sampled_indices)

    #         # Randomly split the rest into voting and evaluation.
    #         total_unsampled = len(unsampled_indices)
    #         np.random.shuffle(unsampled_indices)
    #         vote_indices = unsampled_indices[: total_unsampled // 2]
    #         eval_indices = unsampled_indices[total_unsampled // 2 :]

    #         # Store the posterior in a num_nodes-by-num_classes matrix.
    #         # Posteriors in non-leaf cells will be zero everywhere
    #         # and later changed to uniform.
    #         node_counts = tree.tree_.n_node_samples
    #         class_counts = np.zeros((len(node_counts), K))
    #         for vote_index in vote_indices:
    #             class_counts[
    #                 tree.apply(X[vote_index].reshape(1, -1)).item(), y[vote_index]
    #             ] += 1
    #         row_sums = class_counts.sum(
    #             axis=1
    #         )  # Total number of estimation points in each leaf.
    #         row_sums[row_sums == 0] = 1  # Avoid divide by zero.
    #         class_probs = class_counts / row_sums[:, None]

    #         # Make the nodes that have no estimation indices uniform.
    #         # This includes non-leaf nodes, but that will not affect the estimate.
    #         where_empty = np.argwhere(class_probs.sum(axis=1) == 0)
    #         for elem in where_empty:
    #             class_probs[elem] = [1 / K] * K

    #         # Apply finite sample correction and renormalize.
    #         where_0 = np.argwhere(class_probs == 0)
    #         for elem in where_0:
    #             class_probs[elem[0], elem[1]] = 1 / (
    #                 kappa * class_counts.sum(axis=1)[elem[0]]
    #             )
    #         row_sums = class_probs.sum(axis=1)
    #         class_probs = class_probs / row_sums[:, None]

    #         # Place evaluation points in their corresponding leaf node.
    #         # Store evaluation posterior in a num_eval-by-num_class matrix.
    #         eval_class_probs = [class_probs[x] for x in tree.apply(X[eval_indices])]
    #         eval_entropies = [entropy(posterior) for posterior in eval_class_probs]
    #         cond_entropy += np.mean(eval_entropies)

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)

        # Build forest with default parameters.
        model = BaggingClassifier(
            DecisionTreeClassifier(),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=False,
        )
        model.fit(X, y)
        n = X.shape[0]
        K = model.n_classes_

        _, counts = np.unique(y, return_counts=True)
        est_H_Y = entropy(counts, base=self.base)

        cond_entropy = 0
        self.class_probs = []
        for tree_idx, tree in enumerate(model):

            # Find the indices of the training set used for partition.
            sampled_indices = model.estimators_samples_[tree_idx]
            unsampled_indices = np.delete(np.arange(0, n), sampled_indices)

            # Randomly split the rest into voting and evaluation.
            total_unsampled = len(unsampled_indices)
            np.random.shuffle(unsampled_indices)
            vote_indices = unsampled_indices[: total_unsampled // 2]
            eval_indices = unsampled_indices[total_unsampled // 2 :]

            # Store the posterior in a num_nodes-by-num_classes matrix.
            # Posteriors in non-leaf cells will be zero everywhere
            # and later changed to uniform.
            node_counts = tree.tree_.n_node_samples
            class_counts = np.zeros((len(node_counts), K))
            for vote_index in vote_indices:
                class_counts[
                    tree.apply(X[vote_index].reshape(1, -1)).item(), y[vote_index]
                ] += 1
            row_sums = class_counts.sum(
                axis=1
            )  # Total number of estimation points in each leaf.
            row_sums[row_sums == 0] = 1  # Avoid divide by zero.
            class_probs = class_counts / row_sums[:, None]

            # Make the nodes that have no estimation indices uniform.
            # This includes non-leaf nodes, but that will not affect the estimate.
            where_empty = np.argwhere(class_probs.sum(axis=1) == 0)
            for elem in where_empty:
                class_probs[elem] = [1 / K] * K

            # Apply finite sample correction and renormalize.
            where_0 = np.argwhere(class_probs == 0)
            for elem in where_0:
                class_probs[elem[0], elem[1]] = 1 / (
                    self.kappa * class_counts.sum(axis=1)[elem[0]]
                )
            row_sums = class_probs.sum(axis=1)
            class_probs = class_probs / row_sums[:, None]

            # Remember the posterior from this tree.
            self.class_probs.append(class_probs)

            # Place evaluation points in their corresponding leaf node.
            # Store evaluation posterior in a num_eval-by-num_class matrix.
            eval_class_probs = [class_probs[x] for x in tree.apply(X[eval_indices])]
            eval_entropies = [entropy(posterior) for posterior in eval_class_probs]
            cond_entropy += np.mean(eval_entropies)

        self.cond_entropy = cond_entropy / self.n_estimators
        self.mutual_info = est_H_Y - self.cond_entropy
        self.fitted = True
        return self

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

        eval_class_probs = np.zeros((len(X), len(self.classes_)))
        for tree_idx, tree in enumerate(self.model):
            # Place evaluation points in their corresponding leaf node.
            # Store evaluation posterior in a num_eval-by-num_class matrix.
            eval_class_probs += [self.class_probs[tree_idx][x] for x in tree.apply(X)]

        return eval_class_probs / n_estimators

    def estimate_cond_entropy(self):

        try:
            self.fitted
        except AttributeError:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        return self.cond_entropy

    def estimate_mutual_info(self):

        try:
            self.fitted
        except AttributeError:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        return self.mutual_info
