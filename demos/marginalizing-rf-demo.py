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

# %% Create some random training data.
n_class = 100
signal_d = 2
noise_d = 2
classes = [0, 1]

# Training Data
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

# Validation Data
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

# %% Create an uncertainty forest object.
# Notation from the current paper.
min_samples_leaf = 1  # k
max_features = None  # m
n_estimators = 200  # B
max_samples = 0.5  # s // 2
bootstrap = False  # Whether to subsample with replacement.
parallel = True

uf = UncertaintyForest(
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    n_estimators=n_estimators,
    max_samples=max_samples,
    bootstrap=bootstrap,
    parallel=parallel,
)

# Fit the UF object.
uf.fit(X_train, y_train)

# %% Check probability calculations using the various methods.
print("several test observations with both approaches")
print("first with predict_proba:")
probs = uf.predict_proba(X_train[0:5, :])
print(probs)
print("now with predict_proba2:")
probs = uf.predict_proba_2(X_train[0:5, :])
print(probs)
print("done with several test observations - do they match?")
print("********************************************")
# %%
print("now for some marginal predictions")
print("first two features (2 x signal):")
probs = uf.predict_proba_2(X_train[0, :], np.array((0, 1)))
print(probs)
print("first three features (2 x signal + 1 x noise):")
probs = uf.predict_proba_2(X_train[0, :], np.array((0, 1, 2)))
print(probs)
print("first four features (2 x signal + 2 x noise):")
probs = uf.predict_proba_2(X_train[0, :], np.array((0, 1, 2, 3)))
print(probs)
print("last two features (2 x noise):")
probs = uf.predict_proba_2(X_train[0, :], np.array((2, 3)))
print(probs)
print("********************************************")

# %%
print("Checking conditional entropy and MI numbers")
cond_entropy = uf.estimate_cond_entropy(X_eval)
print("0 <= H(Y|X) = %f <= log2(3) = %f" % (cond_entropy, np.log2(3.0)))
cond_entropy = uf.estimate_cond_entropy_2(X_eval)
print("0 <= H(Y|X) = %f <= log2(3) = %f" % (cond_entropy, np.log2(3.0)))
mutual_info = uf.estimate_mutual_info(X_eval)
print("I(X, Y) = %f" % mutual_info)
mutual_info = uf.estimate_mutual_info_2(X_eval)
print("I(X, Y) = %f" % mutual_info)
print("********************************************")

# %% 
print("conditional entropy and MI on marginal distributions")
print("2 signal features:")
cond_entropy = uf.estimate_cond_entropy_2(X_eval, np.array((0, 1)))
print("0 <= H(Y|X) = %f <= log2(3) = %f" % (cond_entropy, np.log2(3.0)))
mutual_info = uf.estimate_mutual_info_2(X_eval, np.array((0, 1)))
print("I(X, Y) = %f" % mutual_info)
print("2 noise features:")
cond_entropy = uf.estimate_cond_entropy_2(X_eval, np.array((2, 3)))
print("0 <= H(Y|X) = %f <= log2(3) = %f" % (cond_entropy, np.log2(3.0)))
mutual_info = uf.estimate_mutual_info_2(X_eval, np.array((2, 3)))
print("I(X, Y) = %f" % mutual_info)