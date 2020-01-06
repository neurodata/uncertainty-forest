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
        min_samples_leaf=1,  # k
        max_features=None,  # m
        n_estimators=300,  # B
        max_samples=0.5,  # s // 2
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

        if not self.max_features:
            d = X.shape[1]
            self.max_features = int(np.floor(np.sqrt(d)))

        # 'max_samples' determines the number of 'structure' data points
        # that will be used to learn each tree.
        self.model = BaggingClassifier(
            DecisionTreeClassifier(  # max_depth = self.max_depth,
                max_depth=self.max_depth,
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

        self.posterior_per_tree = self._estimate_posteriors()
        self.leaf_stats = self._leaf_stats()

        self.fitted = True
        return self

    def _estimate_posteriors(self):

        n = self.X_.shape[0]

        def worker(tree):
            # Get indices of estimation set, i.e. those NOT used
            # in learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)

            # Count the occurences of each class in each leaf node,
            # by first extracting the leaves.
            # node_counts = tree.tree_.n_node_samples
            unique_leaf_nodes = self._get_leaves(tree)

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
            posterior_per_leaf.tolist() # what does this do if it's already in the return statement?

            return (posterior_per_leaf.tolist(), tree, unique_leaf_nodes, n_per_leaf)

        return Parallel(n_jobs=-2)(delayed(worker)(tree) for tree in self.model)

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
        n, d_ = self.X_.shape
        v, d = X.shape

        if d != d_:
            raise ValueError(
                "Training and evaluation data must have the same number of dimensions."
            )

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

    def estimate_cond_entropy(self, X):

        p = self.predict_proba(X)
        return np.mean(entropy(p.T, base=self.base))

    def estimate_mutual_info(self, X):

        return self.entropy - self.estimate_cond_entropy(X)
    
    def _leaf_stats(self):

        # Store the forest in a different container:
        # We're going to need a few things for every leaf in our forest:
        # leaf size, D_E posterior, and intervals for every dimension.
        # The object we'll create looks like this:
        # forest[tree][leaf_size, leaf_p_hat, [leaf_min x p], [leaf_max x p]]
        
        forest = []
        d_ = self.X_.shape[1]
        
        for b in range(self.n_estimators):
        
            tree = self.model[b]
            tree_list = []
            # region 
            # # First extract the leaves.
            unique_leaf_nodes = self._get_leaves(tree)
            # region Create the leaf_min and leaf_max matrices along every dimension.
            leaf_dim_min = np.repeat(-np.inf, len(unique_leaf_nodes) * d_).reshape(
                len(unique_leaf_nodes), d_
            )
            leaf_dim_max = np.repeat(np.inf, len(unique_leaf_nodes) * d_).reshape(
                len(unique_leaf_nodes), d_
            )
            # For every leaf:
            # Use the training data to find an observation in every leaf.
            # Identify the path to the leaf.
            # Update mins and maxes along the way to each leaf.
            # If we evaluate all training data, we have to recover all leaves.
            # Don't worry, we'll only analyze one observation per leaf.
            train_leaves = tree.apply(self.X_)
            # Learn About Tree Splits
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold
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
            tree_list.append(np.asarray(self.posterior_per_tree[b][3]))
            # Store D_E posterior per leaf.
            tree_list.append(np.asarray(self.posterior_per_tree[b][0]))
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_min))
            # Store feature-by-feature min per leaf.
            tree_list.append(np.asarray(leaf_dim_max))
            # Add all tree features to the forest object.
            forest.append(tree_list)

        return forest

    def predict_proba_2(
        self, X, cond_dims=np.nan,
    ):

        forest = self.leaf_stats
        
        # If the user doesn't specify which feature intervals to check, check them all.
        if np.isnan(cond_dims).any():
            cond_dims = range(0, forest[0][2].shape[1])
        if X.ndim == 1:
            # Make a single sample a row vector (p x 1).
            X = X[:, None].reshape(1, -1)
        all_probs = np.zeros((X.shape[0], forest[0][1].shape[1]))

        for test_obs in range(0, X.shape[0]):
            # This takes a while...why?  Uncomment below for progress.
            # if test_obs % 25 == 0:
            #     print(f"obs {test_obs} of {X.shape[0]}")
            # Store probability of each class for each observation.
            tree_probs = np.zeros((len(forest), forest[0][1].shape[1]))
            for tree in range(0, len(forest)):
                # Figure out which lea(f/ves) the sample would reach.
                leaves_reached = np.zeros(len(forest[tree][0]))
                # Assume X is in every leaf until you see that it isn't.
                for leaf in range(0, len(leaves_reached)):
                    in_leaf = True
                    for dim in cond_dims:
                        if X[test_obs, dim] > forest[tree][3][leaf, dim]:
                            in_leaf = False
                            break
                        if X[test_obs, dim] <= forest[tree][2][leaf, dim]:
                            in_leaf = False
                            break
                    # Uncomment below to see that the correct leaf assignment is made.
                    # if in_leaf:
                    # print(f"Tree: {tree}; Leaf: {leaf}")
                    leaves_reached[leaf] = in_leaf
                for class_num in range(0, tree_probs.shape[1]):
                    tree_probs[tree, class_num] = np.sum(
                        forest[tree][0] * forest[tree][1][:, class_num] * leaves_reached
                    )   / np.sum(forest[tree][0] * leaves_reached)
                # The line below should be 1 for all trees unless cond_dims isn't all dims.
                # print(f"leaves reached: {np.sum(leaves_reached)}")
            normed_tree_probs = np.zeros(tree_probs.shape)
            for i in range(0, tree_probs.shape[0]):
                for j in range(0, tree_probs.shape[1]):
                    normed_tree_probs[i, j] = tree_probs[i, j] / np.sum(tree_probs[i, :])
            all_probs[test_obs, :] = np.mean(normed_tree_probs, axis=0)
        return all_probs

    def estimate_cond_entropy_2(self, X, cond_dims=np.nan,
    ):

        # If the user doesn't specify which feature intervals to check, check them all.
        if np.isnan(cond_dims).any():
            cond_dims = range(0, self.leaf_stats[0][2].shape[1])
        if X.ndim == 1:
            # Make a single sample a row vector (p x 1).
            X = X[:, None].reshape(1, -1)
        p = self.predict_proba_2(X, cond_dims)
        return np.mean(entropy(p.T, base=self.base))

    def estimate_mutual_info_2(self, X,  cond_dims=np.nan,
    ):

        # If the user doesn't specify which feature intervals to check, check them all.
        if np.isnan(cond_dims).any():
            cond_dims = range(0, self.leaf_stats[0][2].shape[1])
        if X.ndim == 1:
            # Make a single sample a row vector (p x 1).
            X = X[:, None].reshape(1, -1)
        return self.entropy - self.estimate_cond_entropy_2(X, cond_dims)
