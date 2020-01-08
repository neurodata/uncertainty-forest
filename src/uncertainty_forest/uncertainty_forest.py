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
        max_depth=30,
        min_samples_leaf=1,
        max_features=None,
        n_estimators=300,
        frac_struct=0.32,
        frac_est=0.34,
        frac_eval=0.34,
        bootstrap=True,
        parallel=True,
        finite_correction=True,
        base=2.0,
    ):

        # Tree parameters.
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_estimators = n_estimators
        self.frac_struct = frac_struct
        self.frac_est = frac_est
        self.frac_eval = frac_eval
        self.bootstrap = bootstrap

        # Model parameters.
        self.parallel = parallel
        self.finite_correction = finite_correction
        self.base = base

    def fit(self, X, y):

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)

        if self.frac_struct + self.frac_est + self.frac_eval != 1.0:
            raise ValueError("frac_struct + frac_est + frac_eval must equal 1.")

        if not self.max_features:
            d = X.shape[1]
            self.max_features = int(np.floor(np.sqrt(d)))

        print(self.frac_struct)
        self.model = BaggingClassifier(
            DecisionTreeClassifier(  # max_depth = self.max_depth,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            ),
            n_estimators=self.n_estimators,
            max_samples=self.frac_struct,
            bootstrap=self.bootstrap,
        )

        self.model.fit(X, y)
        self.posterior_per_tree, self.cond_entropy = self._estimate_posteriors(
            X, self._preprocess_y(y)
        )

        _, counts = np.unique(y, return_counts=True)
        self.mutual_info = entropy(counts, base=self.base) - self.cond_entropy

        self.fitted = True
        return self

    def _estimate_posteriors(self, X, y):

        n = X.shape[0]

        def worker(tree):
            # Get indices of estimation set, i.e. those NOT used
            # in learning trees of the forest.
            unsampled_data = _generate_unsampled_indices(tree.random_state, n)
            np.random.shuffle(unsampled_data)
            num_est = len(unsampled_data) // 2

            # num_est = int(np.ceil(self.frac_est * n))
            estimation_indices = unsampled_data[:num_est]
            eval_indices = unsampled_data[num_est:]

            # print("n", n)
            # print("STRUCT", n - len(unsampled_data))
            # print("EST", len(estimation_indices))
            # print("EVAL", len(eval_indices))

            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
            # node_counts = tree.tree_.n_node_samples
            unique_leaf_nodes = self._get_leaves(tree)

            class_counts_per_leaf = np.zeros(
                (len(unique_leaf_nodes), self.model.n_classes_)
            )

            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(X[i].reshape((1, -1))).item()
                class_counts_per_leaf[
                    np.where(unique_leaf_nodes == temp_node)[0][0], y[i]
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
            posterior_per_leaf = posterior_per_leaf.tolist()

            # Posterior probability for each element of the evaluation set.
            eval_posteriors = np.array(
                [
                    posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]]
                    for node in tree.apply(X[eval_indices, :])
                ]
            )

            cond_entropy = np.mean(entropy(eval_posteriors.T, base=self.base))

            return (posterior_per_leaf, tree, unique_leaf_nodes, cond_entropy)

        if self.parallel:
            uncertainty_per_tree = Parallel(n_jobs=-2)(
                delayed(worker)(tree) for tree in self.model
            )
        else:
            uncertainty_per_tree = []
            for tree in self.model:
                uncertainty_per_tree.append(worker(tree))

        cond_entropies = []
        posterior_per_tree = []
        for elem in uncertainty_per_tree:
            posterior_per_tree.append(elem[0:3])
            cond_entropies.append(elem[3])

        return posterior_per_tree, np.mean(cond_entropies)

    def _get_leaves(self, tree):

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
            c = np.divide(K - np.count_nonzero(leaf), K * n_per_leaf[i])

            ret[i, leaf == 0.0] = np.divide(1, K * n_per_leaf[i])
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

        def worker(tree_fields):

            posterior_per_leaf = tree_fields[0]
            tree = tree_fields[1]
            unique_leaf_nodes = tree_fields[2]

            # Posterior probability for each element of the evaluation set.
            eval_posteriors = [
                posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]]
                for node in tree.apply(X)
            ]

            return np.array(eval_posteriors)

        posteriors = np.array(
            Parallel(n_jobs=-2)(
                delayed(worker)(tree_fields) for tree_fields in self.posterior_per_tree
            )
        )
        return np.mean(posteriors, axis=0)

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

    def predict_proba_leaves(self, X):
        # This version of the function does not
        # apply "unique" to the result of "_get_leaves."

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
            #  for learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)

            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
            # node_counts = tree.tree_.n_node_samples
            leaf_nodes = self._get_leaves(tree)
            # unique_leaf_nodes = np.unique(leaf_nodes)
            # The commented code below shows that leaf_nodes is already a unique list.
            # print(f"Tree {tree}: {len(leaf_nodes)} leaf nodes,
            # {len(unique_leaf_nodes)} unique leaf nodes")
            class_counts_per_leaf = np.zeros(
                # (len(unique_leaf_nodes), self.model.n_classes_)
                (len(leaf_nodes), self.model.n_classes_)
            )

            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(self.X_[i].reshape((1, -1))).item()
                # class_counts_per_leaf[
                #     np.where(unique_leaf_nodes == temp_node)[0][0], self.y_[i]
                # ] += 1
                class_counts_per_leaf[
                    np.where(leaf_nodes == temp_node)[0][0], self.y_[i]
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
                # posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]]
                # for node in tree.apply(X)
                posterior_per_leaf[np.where(leaf_nodes == node)[0][0]]
                for node in tree.apply(X)
            ]

            return np.array(eval_posteriors)

        if self.parallel:
            class_counts = np.array(
                Parallel(n_jobs=-2)(delayed(worker)(tree) for tree in self.model)
            )
            class_counts = np.mean(class_counts, axis=0)
        else:
            class_counts = np.zeros((v, self.model.n_classes_))
            for tree in self.model:
                class_counts += worker(tree)
            class_counts = class_counts / self.n_estimators

        # Normalize counts.
        print("test")
        return class_counts
        # return np.divide(class_counts, class_counts.sum(axis=1, keepdims=True))

    def leaf_stats(self):

        try:
            self.fitted
        except AttributeError:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        n, d_ = self.X_.shape

        # Store the forest in a different container:
        # We're going to need a few things for every leaf in our forest:
        # leaf size, D_E posterior, and intervals for every dimension.
        # The object we'll create looks like this:
        # forest[tree][leaf_size, leaf_p_hat, [leaf_min x p], [leaf_max x p]]
        forest = []
        for b in range(self.n_estimators):
            tree = self.model[b]
            tree_list = []

            # region Get leaf size and leaf posteriors from training data (D_E).

            # Get indices of estimation set, i.e. those NOT used
            # for learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)
            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
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

            # Count the number of data points in each leaf.
            n_per_leaf = class_counts_per_leaf.sum(axis=1)
            n_per_leaf[n_per_leaf == 0] = 1  # Avoid divide by zero.

            # Posterior probability distributions in each leaf.
            # Each row is length num_classes.
            posterior_per_leaf = np.divide(
                class_counts_per_leaf,
                np.repeat(n_per_leaf.reshape((-1, 1)), self.model.n_classes_, axis=1),
            )

            # Note that we don't need anything related to the validation set
            # that is used to estimate conditional entropy or mutual information.

            if self.finite_correction:
                posterior_per_leaf = self._finite_sample_correct(
                    posterior_per_leaf, n_per_leaf
                )
            posterior_per_leaf.tolist()
            # endregion

            # region Create the leaf_min and leaf_max matrices along every dimension.
            leaf_dim_min = np.repeat(-np.inf, len(n_per_leaf) * d_).reshape(
                len(n_per_leaf), d_
            )
            leaf_dim_max = np.repeat(np.inf, len(n_per_leaf) * d_).reshape(
                len(n_per_leaf), d_
            )

            # For every leaf:
            # Use the training data to find an observation in every leaf.
            # Identify the path to the leaf.
            # Update mins and maxes along the way to each leaf.
            # If we evaluate all training data, we have to recover all leaves.
            train_leaves = tree.apply(self.X_)
            # Learn About Tree Structure
            # Obtain basic attributes of every node.
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # The tree structure can be traversed to compute various properties such
            # as the depth of each node and whether or not it is a leaf.
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1
                # If we have a test node (i.e., another split)
                if children_left[node_id] != children_right[node_id]:
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True

            # Pick a sample from each leaf in order to retrieve its path.
            # Here first_member is the first element in
            # the training data assigned to each leaf.
            first_member = np.zeros(len(unique_leaf_nodes))
            idx = 0
            for leaf in unique_leaf_nodes:
                first_member[idx] = np.asarray(np.where(train_leaves == leaf))[0, 0]
                idx = idx + 1
            first_member = first_member.astype(int)
            # What path leads to a leaf for each first member?
            node_indicator = tree.decision_path(self.X_[first_member, :])
            # Step through the path to each leaf.
            # If we encounter a split on X_j, we note the threshold and the direction,
            # and then we update the interval object.
            leaf_idx = 0
            for sample_id in range(0, len(first_member)):
                node_index = node_indicator.indices[
                    node_indicator.indptr[sample_id] : node_indicator.indptr[
                        sample_id + 1
                    ]
                ]
                for node_id in node_index:
                    # This check is for leaf nodes where the
                    # identified "feature" is always -2.
                    if feature[node_id] == -2:
                        continue
                    # Update leaf interval min and max as required.
                    if (
                        self.X_[first_member[sample_id], feature[node_id]]
                        <= threshold[node_id]
                    ):
                        leaf_dim_max[leaf_idx, feature[node_id]] = threshold[node_id]
                    else:
                        leaf_dim_min[leaf_idx, feature[node_id]] = threshold[node_id]
                leaf_idx = leaf_idx + 1
            # endregion

            # Store n_per_leaf.
            tree_list.append(n_per_leaf)
            # Store D_E posterior per leaf.
            tree_list.append(posterior_per_leaf)
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_min))
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_max))
            # Add all tree features to the forest object.
            forest.append(tree_list)

        print("test-leaf_stats - unique")
        return forest

    def leaf_stats_leaves(self):

        try:
            self.fitted
        except AttributeError:
            msg = (
                "This %(name)s instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
            raise NotFittedError(msg % {"name": type(self).__name__})

        n, d_ = self.X_.shape

        # Store the forest in a different container:
        # We're going to need a few things for every leaf in our forest:
        # leaf size, D_E posterior, and intervals for every dimension.
        # The object we'll create looks like this:
        # forest[tree][leaf_size, leaf_p_hat, [leaf_min x p], [leaf_max x p]]
        forest = []
        for b in range(self.n_estimators):
            tree = self.model[b]
            tree_list = []

            # region Get leaf size and leaf posteriors from training data (D_E).

            # Get indices of estimation set, i.e. those NOT used
            # for learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)
            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
            leaf_nodes = self._get_leaves(tree)
            # unique_leaf_nodes = np.unique(leaf_nodes)
            class_counts_per_leaf = np.zeros(
                # (len(unique_leaf_nodes), self.model.n_classes_)
                (len(leaf_nodes), self.model.n_classes_)
            )
            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(self.X_[i].reshape((1, -1))).item()
                # class_counts_per_leaf[
                #     np.where(unique_leaf_nodes == temp_node)[0][0], self.y_[i]
                # ] += 1
                class_counts_per_leaf[
                    np.where(leaf_nodes == temp_node)[0][0], self.y_[i]
                ] += 1

            # Count the number of data points in each leaf.
            n_per_leaf = class_counts_per_leaf.sum(axis=1)
            n_per_leaf[n_per_leaf == 0] = 1  # Avoid divide by zero.

            # Posterior probability distributions in each leaf.
            # Each row is length num_classes.
            posterior_per_leaf = np.divide(
                class_counts_per_leaf,
                np.repeat(n_per_leaf.reshape((-1, 1)), self.model.n_classes_, axis=1),
            )

            # Note that we don't need anything related to the validation set that
            # is used to estimate conditional entropy or mutual information.

            if self.finite_correction:
                posterior_per_leaf = self._finite_sample_correct(
                    posterior_per_leaf, n_per_leaf
                )
            posterior_per_leaf.tolist()
            # endregion

            # region Create the leaf_min and leaf_max matrices along every dimension.
            leaf_dim_min = np.repeat(-np.inf, len(n_per_leaf) * d_).reshape(
                len(n_per_leaf), d_
            )
            leaf_dim_max = np.repeat(np.inf, len(n_per_leaf) * d_).reshape(
                len(n_per_leaf), d_
            )

            # For every leaf:
            # Use the training data to find an observation in every leaf.
            # Identify the path to the leaf.
            # Update mins and maxes along the way to each leaf.
            # If we evaluate all training data, we have to recover all leaves.
            train_leaves = tree.apply(self.X_)
            # Learn About Tree Structure
            # Obtain basic attributes of every node.
            n_nodes = tree.tree_.node_count
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            # The tree structure can be traversed to compute various properties such
            # as the depth of each node and whether or not it is a leaf.
            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
            stack = [(0, -1)]  # seed is the root node id and its parent depth
            while len(stack) > 0:
                node_id, parent_depth = stack.pop()
                node_depth[node_id] = parent_depth + 1
                # If we have a test node (i.e., another split)
                if children_left[node_id] != children_right[node_id]:
                    stack.append((children_left[node_id], parent_depth + 1))
                    stack.append((children_right[node_id], parent_depth + 1))
                else:
                    is_leaves[node_id] = True

            # Pick a sample from each leaf in order to retrieve its path.
            # Here first_member is the first element
            # in the training data assigned to each leaf.
            # first_member = np.zeros(len(unique_leaf_nodes))
            first_member = np.zeros(len(leaf_nodes))
            idx = 0
            # for leaf in unique_leaf_nodes:
            for leaf in leaf_nodes:
                first_member[idx] = np.asarray(np.where(train_leaves == leaf))[0, 0]
                idx = idx + 1
            first_member = first_member.astype(int)
            # What path leads to a leaf for each first member?
            node_indicator = tree.decision_path(self.X_[first_member, :])
            # Step through the path to each leaf.
            # If we encounter a split on X_j, we note the threshold and the direction,
            # and then we update the interval object.
            leaf_idx = 0
            for sample_id in range(0, len(first_member)):
                node_index = node_indicator.indices[
                    node_indicator.indptr[sample_id] : node_indicator.indptr[
                        sample_id + 1
                    ]
                ]
                for node_id in node_index:
                    # This check is for leaf nodes where the
                    # identified "feature" is always -2.
                    if feature[node_id] == -2:
                        continue
                    # Update leaf interval min and max as required.
                    if (
                        self.X_[first_member[sample_id], feature[node_id]]
                        <= threshold[node_id]
                    ):
                        leaf_dim_max[leaf_idx, feature[node_id]] = threshold[node_id]
                    else:
                        leaf_dim_min[leaf_idx, feature[node_id]] = threshold[node_id]
                leaf_idx = leaf_idx + 1
            # endregion

            # Store n_per_leaf.
            tree_list.append(n_per_leaf)
            # Store D_E posterior per leaf.
            tree_list.append(posterior_per_leaf)
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_min))
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_max))
            # Add all tree features to the forest object.
            forest.append(tree_list)

        print("test-leaf_stats")
        return forest
