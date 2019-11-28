from sklearn.ensemble.forest import _generate_unsampled_indices
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np

class UncertaintyForest:

    def __init__(self,
                max_depth=30,           # D
                min_samples_leaf=1,     # k
                max_features = None,    # m
                n_trees = 200,          # B
                max_samples = .32,      # s // 2
                bootstrap = False):
        
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
        self.n_trees = n_trees       
        self.max_samples = max_samples   
        self.bootstrap = bootstrap

    def _build_forest(self, X_train, y_train):

        # TO DO: Validate input, understand this check.
        if X.ndim == 1:
            raise ValueError('HH: 1d data will cause headaches down the road.')
            
        if not max_features:
            max_features = int(np.ceil(np.sqrt(X.shape[1])))
            
        # 'max_samples' determines the number of 'structure' data points that will be used to learn each tree.
        model = BaggingClassifier(DecisionTreeClassifier(max_depth = self.max_depth, 
                                                        min_samples_leaf = self.min_samples_leaf,
                                                        max_features = self.max_features),
                                                        n_estimators = self.n_trees,
                                                        max_samples = self.max_samples,
                                                        bootstrap = self.bootstrap)
        
        model.fit(X, y)
        self.model = model
        return model

    def _get_leaves(self, tree):

        # TO DO: Check this tutorial.
        # adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
        
        # TO DO: Remove unnecessary lines.
        # n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        # feature = tree.tree_.feature
        # threshold = tree.tree_.threshold
        
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

        where_0 = np.argwhere(posterior_per_leaf == 0.0)
        where_not = np.argwhere(posterior_per_leaf != 0.0)
        K = posterior_per_leaf.shape[1]
        c = np.divide(len(where_0), K*n_per_leaf)
        
        posterior_per_leaf[where_0] = np.divide(1, K*n_per_leaf)
        posterior_per_leaf[where_not] = (1 - c)*posterior_per_leaf[where_not]
        
        return posterior_per_leaf

    def _estimate_entropy(self, y_train):

        classes = np.unique(y_train)
        K = len(classes)
        n = len(y_train)
        p = [np.sum(classes[k] == y_train)/n for k in range(K)]

        h = p
        where_not = np.argwhere(p != 0.0)
        h[where_not] = np.multiply(p[where_not], np.log2(p[where_not]))
        self.entropy = np.sum(h)
        return self.entropy

    def estimate_cond_probability(self, X_train, y_train, X_eval, model):

        n, d  = X_train.shape
        v, d_ = X_eval.shape
        
        # TO DO: Bake into input validation function.
        if d != d_:
            raise ValueError("train and test data in different dimensions")
        
        class_counts = np.zeros((v, model.n_classes_))
        for tree in model:
            # Get indices of estimation set, i.e. those NOT used in for learning trees of the forest.
            estimation_indices = _generate_unsampled_indices(tree.random_state, n)
            estimation_set = X_train[estimation_indices]
            
            # Count the occurences of each class in each leaf node, by first extracting the leaves.
            leaf_nodes = _get_leaves(tree)
            unique_leaf_nodes = np.unique(leaf_nodes)
            class_counts_per_leaf = np.zeros((len(unique_leaf_nodes), model.n_classes_))    

            # Drop each estimation example down the tree, and record its 'y' value.
            for i in estimation_indices:
                temp_node = tree.apply(estimation_set.reshape(1, -1)).item()
                class_counts_per_leaf[np.where(unique_leaf_nodes == temp_node)[0][0], y_train[i]] += 1
                
            # Count the number of data points in each leaf in.
            n_per_leaf = class_counts_per_leaf.sum(axis=1)
            n_per_leaf[n_per_leaf_sums == 0] = 1 # Avoid divide by zero.

            # Posterior probability distributions in each leaf. Each row is length num_classes.
            posterior_per_leaf = np.divide(class_counts_per_leaf, n_per_leaf)
            posterior_per_leaf = _finite_sample_correct(posterior_per_leaf, n_per_leaf)
            posterior_per_leaf.tolist()

            # Posterior probability for each element of the evaluation set.
            eval_posteriors = [posterior_per_leaf[np.where(unique_leaf_nodes == node)[0][0]] for node in tree.apply(X_eval)]
            eval_posteriors = np.array(eval_posteriors)
            
            # Number of estimation points in the cell of each eval point.
            n_per_eval_leaf = np.asarray([node_counts[np.where(unique_leaf_nodes == x)[0][0]] for x in tree.apply(X_eval)])
            class_counts += np.multiply(eval_posteriors, n_per_eval_leaf)
        
        # Normalize counts.
        self.cond_probability = np.divide(class_counts, class_counts.sum(axis=1, keepdims=True))
        return self.cond_probability

    def estimate_cond_entropy(self, X_train, y_train, X_eval):
        
        model = self._build_forest(X_train, y_train)
        p = self.estimate_cond_probability(X_train, y_train, X_eval, model)
        self.cond_entropy = np.mean(np.multiply(p, np.log2(p)).sum(axis=1)) 
        return self.cond_entropy

    def estimate_mutual_info(self, X_train, y_train, X_eval):

        self.estimate_cond_entropy(X_train, y_train, X_eval)
        self._estimate_entropy(y_train)
        self.mutual_info = self.entropy - self.cond_entropy
        return self.mutual_info
