#%% What does our desired object look like?

# We'd like to capture the entire structure of our random forest as follows:
#
# For every tree:
#    For every leaf:
#       Leaf Size
#       Leaf Posterior
#       For every dimension:
#           (min, max) - leaf interval
#
# It should look like this:
# forest[tree][0] = leaf size (num_leaves x 1)
# forest[tree][1] = leaf posterior (num_leaves x num_classes)
# forest[tree][2] = leaf interval min (num_leaves rows, each like this: [X1_min, ..., Xp_min])
# forest[tree][3] = leaf interval max (num_leaves rows, each like this: [X1_max, ..., Xp_max])
#
# Let's do it!

# %%
import numpy as np
from uncertainty_forest.uncertainty_forest import UncertaintyForest

np.random.seed(123)

# %% Create an uncertainty forest object.
# Notation from the current paper.
# max_depth = 30       # D
min_samples_leaf = 1  # k
max_features = None  # m
n_estimators = 200  # B
max_samples = 0.5  # s // 2
bootstrap = False  # Whether to subsample with replacement.
parallel = False

uf = UncertaintyForest(
    # max_depth = max_depth,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    n_estimators=n_estimators,
    max_samples=max_samples,
    bootstrap=bootstrap,
    parallel=parallel,
    finite_correction=True,
)

# %% Create some random training data.
n_class = 1000
signal_d = 2
noise_d = 2
classes = [0, 1]

X_train_signal = np.concatenate(
    [
        np.random.multivariate_normal(k * np.ones(signal_d), np.eye(signal_d), n_class)
        for k in classes
    ]
)
X_train_noise = np.concatenate(
    [
        np.random.multivariate_normal(np.zeros(noise_d), np.eye(noise_d), n_class)
        for k in classes
    ]
)
X_train = np.hstack((X_train_signal, X_train_noise))
y_train = np.concatenate([k * np.ones(n_class) for k in classes])

# region  The demo doesn't require this evaluation data just yet.
X_eval_signal = np.concatenate(
    [
        np.random.multivariate_normal(k * np.ones(signal_d), np.eye(signal_d), n_class)
        for k in classes
    ]
)
X_eval_noise = np.concatenate(
    [
        np.random.multivariate_normal(np.zeros(noise_d), np.eye(noise_d), n_class)
        for k in classes
    ]
)
X_eval = np.hstack((X_eval_signal, X_eval_noise))
# endregion

# %% Fit the UF object.
uf.fit(X_train, y_train)

# %% Obtain the alternate forest representations.
forest_unique_leaves = uf.leaf_stats()
forest_leaves = uf.leaf_stats()

# %% Calculate probabilities using the alternate forest representations.
def predict_proba_2(
    X, forest, cond_dims=np.nan,
):
    # If the user doesn't specify which feature intervals to check, check them all.
    if np.isnan(cond_dims):
        cond_dims = range(0, forest[0][2].shape[1])
    # Store probability of each class for each observation.
    tree_probs = np.zeros((len(forest), forest[0][1].shape[1]))
    for tree in range(0, len(forest)):
        # Figure out which lea(f/ves) the sample would reach.
        leaves_reached = np.zeros(len(forest[tree][0]))
        # Assume X is in every leaf until you see that it isn't.
        for leaf in range(0, len(leaves_reached)):
            in_leaf = True
            for dim in cond_dims:
                if X[dim] > forest[tree][3][leaf, dim]:
                    in_leaf = False
                if X[dim] <= forest[tree][2][leaf, dim]:
                    in_leaf = False
            # Uncomment below to see that the correct leaf assignment is made.
            # if in_leaf:
            # print(f"Tree: {tree}; Leaf: {leaf}")
            leaves_reached[leaf] = in_leaf
        for class_num in range(0, tree_probs.shape[1]):
            tree_probs[tree, class_num] = np.sum(
                forest[tree][0] * forest[tree][1][:, class_num] * leaves_reached
            )  # / np.sum(forest[tree][0] * leaves_reached)
        # The line below should be 1 for all trees unless cond_dims isn't all dims.
        # print(f"leaves reached: {np.sum(leaves_reached)}")

    normed_tree_probs = np.zeros(tree_probs.shape)
    for i in range(0, tree_probs.shape[0]):
        for j in range(0, tree_probs.shape[1]):
            normed_tree_probs[i, j] = tree_probs[i, j] / np.sum(tree_probs[i, :])
    # I think the UF predict_proba function is actually doing a weighted average
    # where each tree gets a vote weight corresponding to the n_per_leaf of the
    # target observation.
    # ----
    # weighted_tree_probs_sum = np.sum(tree_probs, axis = 0)
    # return np.divide(weighted_tree_probs_sum, np.sum(weighted_tree_probs_sum))
    # ----
    return np.mean(normed_tree_probs, axis=0)


# %% Check probability calculations using the various methods.
# uf.predict_proba (original)
probs = uf.predict_proba(X_train[0:1, :])
print(probs)
# [[0.79019176 0.20980824]]
# uf.predict_proba_leaves (did not "unique" the leaves)
probs = uf.predict_proba_leaves(X_train[0:1, :])
print(probs)
# [[0.8355226 0.1644774]]

# %% Compare to marginal probability function using leaf_stats (no marginalization here).
# unique leaves used in uf.leaf_stats() function definition
probs = predict_proba_2(X_train[0, :], forest_unique_leaves)
print(probs)
# [0.81077485 0.18922515]
# no unique leaves used in uf.leaf_stats_leaves() function definition
probs = predict_proba_2(X_train[0, :], forest_leaves)
print(probs)
# [0.81077485 0.18922515]

# %% Comments
# So I get the same thing with my method whether I unique the leaves or not.
# I get something different with predict_proba depending on whether I unique
# the leaves.  Also, I'm pretty sure the trees in predict_proba have different
# voting weights corresponding to the n_per_leaf for the target observation.
# Check line 210 "class_count_increments" returned by the worker() for each
# tree.  These are influenced by n_per_eval_leaf and then the sums for the
# classes are normed at the end.

