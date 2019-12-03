from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy
from joblib import Parallel, delayed

import numpy as np

class UncertaintyForest:
    def __init__(self,
                max_depth=30,           # D
                min_samples_leaf=1,     # k
                max_features = None,    # m
                n_estimators = 200,          # B
                max_samples = .32,      # s // 2
                bootstrap = False):
        

        # For inference.
        self.X_ = None
        self.y_ = None
        self.is_fitted = False

        # Outputs.
        self.model = None
        self.cond_probability = None
        self.cond_entropy = None
        self.entropy = None
        self.mutual_info = None

        # Algorithmic hyperparameters.
        self.max_depth = max_depth          
        self.min_samples_leaf = min_samples_leaf    
        self.max_features = max_features   
        self.n_estimators = n_estimators       
        self.max_samples = max_samples   
        self.bootstrap = bootstrap

    def fit(self, X_train, y_train):

        # TO DO: Bake into input validation.
        if X_train.ndim == 1:
            raise ValueError('Reshape `X_train` data n-by-1 2D array.')
            
        if not self.max_features:
            self.max_features = int(np.ceil(np.sqrt(X_train.shape[1])))
            
        # 'max_samples' determines the number of 'structure' data points that will be used to learn each tree.
        model = BaggingClassifier(DecisionTreeClassifier(max_depth = self.max_depth, 
                                                        min_samples_leaf = self.min_samples_leaf,
                                                        max_features = self.max_features),
                                                        n_estimators = self.n_estimators,
                                                        max_samples = self.max_samples,
                                                        bootstrap = self.bootstrap)
        
        model.fit(X_train, y_train)

        self.X_ = X_train
        self.y_ = y_train
        self.is_fitted = True

        self.model = model
        return model

    def _get_leaves(self, tree):

        # TO DO: Check this tutorial.
        # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        
        leaf_ids = []
        stack = [(0, -1)] 
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                leaf_ids.append(node_id)
                
        return np.array(leaf_ids)

    def _finite_sample_correct(self, posterior_per_leaf, n_per_leaf):

        l = posterior_per_leaf.shape[0]
        K = posterior_per_leaf.shape[1]
        ret = np.zeros(posterior_per_leaf.shape)
        for i in range(l):
            leaf = posterior_per_leaf[i, :]
            c = np.divide(l - np.count_nonzero(leaf), K*n_per_leaf[i])
            
            ret[i, leaf == 0.0] = np.divide(1, K*n_per_leaf[i])
            ret[i, leaf != 0.0] = (1 - c)*posterior_per_leaf[i, leaf != 0.0]
        
        return ret

    def _estimate_entropy(self, y_train):

        _, counts = np.unique(y_train, return_counts = True)
        self.entropy = entropy(counts, base = 2)
        return self.entropy

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

    def _estimate_posterior(self, X_eval, representation=None, decider=None, parallel=False):

        if not self.is_fitted:
            raise ValueError('Model not fitted')
        
        # TO DO: Bake into input validation function? 
        # HH: d and d_ need not be the same for L2F..

        # if d != d_:
        #     raise ValueError("Training and evaluation data must have the same number of dimension.")


        # If self.X_ is a list then the object is a LifelongForest
        if isinstance(self.X_, list):
            models = self.models_

            # If representation is 'all' then use all tasks
            if representation is 'all':
                representations = np.arange(self.n_tasks)
            # If representation is None and the object is a LifelongForest then use all the tasks
            elif representation is None:
                representations = np.arange(self.n_tasks)

            X_train = self.X_[decider]
            y_train = self.y_[decider]

            n_classes = len(np.unique(self.y_[decider]))


        # If self.X_ is not a list then the object is a UncertaintyForest
        else:
            # Make everything iterable
            models = [self.model]
            representations = [0]
            decider = 0

            X_train = self.X_
            y_train = self.y_

        # Probably need to preprocess y_train

        classes = self._preprocess_y(y_train)
        n_classes = len(np.unique(classes))

        v, d_ = X_eval.shape
        
        def worker(tree, X_eval, X_train, y_train, in_task):
            # Get indices of estimation set, i.e. those NOT used in for learning trees of the forest.

            n, d = X_train.shape
            v, d_ = X_eval.shape

            n_classes = len(np.unique(y_train))

            if in_task:
                estimation_indices = _generate_unsampled_indices(tree.random_state, n)
            else:
                # This assumes that all out of task data is used for estimation
                estimation_indices = np.arange(n)
            
            # Count the occurences of each class in each leaf node, by first extracting the leaves.
            node_counts = tree.tree_.n_node_samples
            leaf_nodes = self._get_leaves(tree)
            unique_leaf_nodes = np.unique(leaf_nodes)
            class_counts_per_leaf = np.zeros((len(unique_leaf_nodes), n_classes))

            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(X_train[i].reshape((1, -1))).item()
                class_counts_per_leaf[np.where(unique_leaf_nodes == temp_node)[0][0], y_train[i]] += 1
                
            # Count the number of data points in each leaf in.
            n_per_leaf = class_counts_per_leaf.sum(axis=1)
            n_per_leaf[n_per_leaf == 0] = 1 # Avoid divide by zero.

            # Posterior probability distributions in each leaf. Each row is length num_classes.
            posterior_per_leaf = np.divide(class_counts_per_leaf, np.repeat(n_per_leaf.reshape((-1, 1)), n_classes, axis=1))
            posterior_per_leaf = self._finite_sample_correct(posterior_per_leaf, n_per_leaf)
            posterior_per_leaf.tolist()

            # Posterior probability for each element of the evaluation set.
            eval_posteriors = [posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]] for node in tree.apply(X_eval)]
            eval_posteriors = np.array(eval_posteriors)
            
            # Number of estimation points in the cell of each eval point.
            n_per_eval_leaf = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(X_eval)])
            
            class_count_increment = np.multiply(eval_posteriors, np.repeat(n_per_eval_leaf.reshape((-1, 1)), n_classes, axis=1))

            return class_count_increment

        if decider == representation:
            in_task = True
        else:
            in_task = False
        
        class_counts = np.array([np.zeros((v, n_classes)) for i in representations])
        normalized_cond_probability = np.array([np.zeros((v, n_classes)) for i in representations])

        for i, reps in enumerate(representations):
            condensed_worker = lambda x: worker(x, X_eval, X_train, classes, in_task)
            if parallel: 
                class_counts[i] += np.array(Parallel(n_jobs=-2)(delayed(condensed_worker)(tree) for tree in models[reps]))
                class_counts[i] += np.sum(class_counts, axis=0)
            else:
                # class_counts = np.zeros((v, model.n_classes_))
                for tree in models[reps]:
                    temp = condensed_worker(tree)
                    class_counts[i] += condensed_worker(tree)
        
            normalized_cond_probability[i] = np.divide(class_counts, class_counts.sum(axis=1, keepdims=True))
        
        self.cond_probability = np.mean(normalized_cond_probability, axis=0)

        # Precompute entropy of y to use later for mutual info.
        self._estimate_entropy(y_train)

        return self.cond_probability


    def estimate_cond_probability(self, X_eval, representation=None, decider=None, parallel=False):

        return self._estimate_posterior(X_eval, parallel)


    def predict(self, X_eval, representation=None, decider=None, parallel=False):

        if isinstance(self.X_, list):
            y_train=self.y_[decider]
        else:
            y_train=self.y_

        classes = np.unique(y_train)
        predictions=np.argmax(self._estimate_posterior(X_eval, representation, decider, parallel), axis=1)

        return classes[predictions]


    def estimate_cond_entropy(self, X_eval, representation=None, decider=None, parallel=False):
        
        # User can supply training or evaluation data,
        # in which case rewrite stored conditional probability.
        
        # if (X_train is not None) and (y_train is not None) and (X_eval is not None):
        p = self.estimate_cond_probability(X_eval, representation, decider, parallel)
        # elif (X_train is not None) or (y_train is not None) or (X_eval is not None):
            # raise ValueError("Must supply 'X_train', 'y_train', and 'X_eval' to compute estimate.")
        # else:
            # p = self.cond_probability
            # if p is None:
                # raise ValueError("No previously computed conditional probabilities. Supply training and evaluation data.")
        
        self.cond_entropy = np.mean(entropy(p.T, base = 2))
        return self.cond_entropy


    def estimate_mutual_info(self, X_eval, parallel = False):
        
        if X_eval is not None:
            self.estimate_cond_entropy(X_eval, parallel)
        
        self.mutual_info = self.entropy - self.cond_entropy        
        
        return self.mutual_info


class LifelongForest(UncertaintyForest):
    def __init__(self):

        # Lifelong Forests attributes
        self.models_ = []
        self.X_ = []
        self.y_ = []
        self.n_tasks = 0
        self.is_fitted=False

    def new_forest(self, 
                X_train, 
                y_train, 
                n_estimators=200, 
                max_samples=0.32,
                bootstrap=True, 
                max_depth=30, 
                min_samples_leaf=1,
                acorn=None):

        """
        Input
        X: an array-like object of features; X.shape == (n_samples, n_features)
        y: an array-like object of class labels; len(y) == n_samples
        n_estimators: int; number of trees to construct (default = 200)
        max_samples: float in (0, 1]: number of samples to consider when 
            constructing a new tree (default = 0.32)
        bootstrap: bool; If True then the samples are sampled with replacement
        max_depth: int; maximum depth of a tree
        min_samples_leaf: int; minimum number of samples in a leaf node
        
        Return
        model: a BaggingClassifier fit to X, y
        """
        

        if X.ndim == 1:
            raise ValueError('1d data will cause headaches down the road')
            
        if acorn is not None:
            np.random.seed(acorn)

            
        n = X_train.shape[0]
        K = len(np.unique(y_train))
        
        max_features = int(np.ceil(np.sqrt(X.shape[1])))

        model=UncertaintyForest(max_depth, min_samples_leaf, max_features,
                            n_estimators, max_samples, bootstrap).fit(X_train, y_train)

        # model=BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, 
        #                                             min_samples_leaf=min_samples_leaf,
        #                                             max_features = max_features),
        #                         n_estimators=n_estimators,
        #                         max_samples=max_samples,
        #                         bootstrap=bootstrap)

        self.X_.append(X_train)
        self.y_.append(y_train)
        self.models_.append(model)
        self.n_tasks += 1
        
        return model

    # def _estimate_posterior(self, 
    #                         X_eval, 
    #                         representation=0, 
    #                         decider=0, 
    #                         subsample=1, 
    #                         parallel=False, 
    #                         acorn=None):

    #     if acorn is not None:
    #         np.random.seed(acorn)

    #     X_train = self.X_[decider]
    #     y_train = self.y_[decider]

    #     model = self.models_[representation]

    #     n, d  = X_train.shape
    #     v, d_ = X_eval.shape
        
    #     # TO DO: Bake into input validation function?
    #     if d != d_:
    #         raise ValueError("Training and evaluation data must be the same different dimension.")

    #     if parallel: 
    #         def worker(tree, representation, decider):

    #             if representation == decider:
    #                 # Get indices of estimation set, i.e. those NOT used in for learning trees of the forest.
    #                 estimation_indices = _generate_unsampled_indices(tree.random_state, n)
    #             else:
    #                 estimation_indices = np.random.choice(v, replace=False, p=subsample)
                
    #             # Count the occurences of each class in each leaf node, by first extracting the leaves.
    #             node_counts = tree.tree_.n_node_samples
    #             leaf_nodes = self._get_leaves(tree)
    #             unique_leaf_nodes = np.unique(leaf_nodes)
    #             class_counts_per_leaf = np.zeros((len(unique_leaf_nodes), model.n_classes_))

    #             # Drop each estimation example down the tree, and record its 'y' value.
    #             for i in estimation_indices:
    #                 temp_node = tree.apply(X_train[i].reshape((1, -1))).item()
    #                 class_counts_per_leaf[np.where(unique_leaf_nodes == temp_node)[0][0], y_train[i]] += 1
                    
    #             # Count the number of data points in each leaf in.
    #             n_per_leaf = class_counts_per_leaf.sum(axis=1)
    #             n_per_leaf[n_per_leaf == 0] = 1 # Avoid divide by zero.

    #             # Posterior probability distributions in each leaf. Each row is length num_classes.
    #             posterior_per_leaf = np.divide(class_counts_per_leaf, np.repeat(n_per_leaf.reshape((-1, 1)), model.n_classes_, axis=1))
    #             posterior_per_leaf = self._finite_sample_correct(posterior_per_leaf, n_per_leaf)
    #             posterior_per_leaf.tolist()

    #             # Posterior probability for each element of the evaluation set.
    #             eval_posteriors = [posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]] for node in tree.apply(X_eval)]
    #             eval_posteriors = np.array(eval_posteriors)
                
    #             # Number of estimation points in the cell of each eval point.
    #             n_per_eval_leaf = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(X_eval)])
                
    #             class_count_increment = np.multiply(eval_posteriors, np.repeat(n_per_eval_leaf.reshape((-1, 1)), model.n_classes_, axis=1))
    #             return class_count_increment

    #         class_counts = np.array(Parallel(n_jobs=-2)(delayed(worker)(tree) for tree in model))
    #         class_counts = np.sum(class_counts, axis = 0)
    #     else:
    #         class_counts = np.zeros((v, model.n_classes_))
    #         for tree in model:

    #             if representation == decider:
    #                 # Get indices of estimation set, i.e. those NOT used in for learning trees of the forest.
    #                 estimation_indices = _generate_unsampled_indices(tree.random_state, n)
    #             else:
    #                 estimation_indices = np.random.choice(v, replace=False, p=subsample)
                
    #             # Count the occurences of each class in each leaf node, by first extracting the leaves.
    #             node_counts = tree.tree_.n_node_samples
    #             leaf_nodes = self._get_leaves(tree)
    #             unique_leaf_nodes = np.unique(leaf_nodes)
    #             class_counts_per_leaf = np.zeros((len(unique_leaf_nodes), model.n_classes_))

    #             # Drop each estimation example down the tree, and record its 'y' value.
    #             for i in estimation_indices:
    #                 temp_node = tree.apply(X_train[i].reshape((1, -1))).item()
    #                 class_counts_per_leaf[np.where(unique_leaf_nodes == temp_node)[0][0], y_train[i]] += 1
                    
    #             # Count the number of data points in each leaf in.
    #             n_per_leaf = class_counts_per_leaf.sum(axis=1)
    #             n_per_leaf[n_per_leaf == 0] = 1 # Avoid divide by zero.

    #             # Posterior probability distributions in each leaf. Each row is length num_classes.
    #             posterior_per_leaf = np.divide(class_counts_per_leaf, np.repeat(n_per_leaf.reshape((-1, 1)), model.n_classes_, axis=1))
    #             posterior_per_leaf = self._finite_sample_correct(posterior_per_leaf, n_per_leaf)
    #             posterior_per_leaf.tolist()

    #             # Posterior probability for each element of the evaluation set.
    #             eval_posteriors = [posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]] for node in tree.apply(X_eval)]
    #             eval_posteriors = np.array(eval_posteriors)
                
    #             # Number of estimation points in the cell of each eval point.
    #             n_per_eval_leaf = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(X_eval)])
    #             class_counts += np.multiply(eval_posteriors, np.repeat(n_per_eval_leaf.reshape((-1, 1)), model.n_classes_, axis=1))
        
    #     # Normalize counts.
    #     posteriors = np.divide(class_counts, class_counts.sum(axis=1, keepdims=True))

    #     return posteriors

    # def predict(self, 
    #             X_eval, 
    #             representation='all', 
    #             decider=0,
    #             acorn=None):



    #     sum_posteriors = np.zeros((X_eval.shape[0], self.n_classes))
        
    #     if representation is 'all':
            
    #         for i in range(self.n_tasks):
    #             sum_posteriors += self._estimate_posteriors(test,
    #                                                         i,
    #                                                         decider,
    #                                                         subsample,
    #                                                         acorn)
            
    #     else:
    #         sum_posteriors += self._estimate_posteriors(test,
    #                                                     representation,
    #                                                     decider,
    #                                                     subsample,
    #                                                     acorn)
                
    #     return np.argmax(sum_posteriors, axis=1)