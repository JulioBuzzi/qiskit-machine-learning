# imports
import numpy as np  #manipulação vetor e matriz
from collections import Counter #contar quantas vezes cada classe aparece

from qiskit_machine_learning.algorithms.serializable_model import SerializableModelMixin

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
#from qiskit.primitives import Sampler

class Node:
    """Representa um nó da árvore."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class QuantumDecisionTreeClassifier(SerializableModelMixin):

    def __init__(self, max_depth=5):

        self.max_depth = max_depth
        self.root = None

        # número de features (definido depois)
        self.n_features = None

        # kernel quântico (inicialmente vazio)
        self.quantum_kernel = None

    def fit(self, X, y):

        self.n_features = X.shape[1]

        # criar feature map
        feature_map = ZZFeatureMap(feature_dimension=self.n_features, reps=2)

        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=feature_map
        )

        self.root = self._grow_tree(X, y)

    def _quantum_similarity(self, x1, x2):

        x1 = np.array(x1).reshape(1, -1)
        x2 = np.array(x2).reshape(1, -1)

        kernel_matrix = self.quantum_kernel.evaluate(x1, x2)

        return kernel_matrix[0][0]

    def _grow_tree(self, X, y, depth=0):

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y)

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _most_common_label(self, y):

        counter = Counter(y)
        value = counter.most_common(1)[0][0]

        return value

    def _split(self, X_column, threshold):

        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):

        hist = np.bincount(y)
        ps = hist / len(y)

        entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])

        return entropy

    def _information_gain(self, y, X_column, threshold):

        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_left = len(left_idxs)
        n_right = len(right_idxs)

        e_left = self._entropy(y[left_idxs])
        e_right = self._entropy(y[right_idxs])

        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        information_gain = parent_entropy - child_entropy

        return information_gain

    def _best_split(self, X, y):

        best_score = -1
        split_idx = None
        split_threshold = None

        n_samples, n_features = X.shape

        for feature_idx in range(n_features):

            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:

                gain = self._information_gain(y, X_column, threshold)

                if gain == 0:
                    continue

                left_idxs, right_idxs = self._split(X_column, threshold)

                # pegar um representante de cada lado
                x_left = X[left_idxs[0]]
                x_right = X[right_idxs[0]]

                # calcular similaridade quântica
                q_sim = self._quantum_similarity(x_left, x_right)

                # score híbrido (gain information * quantum similarity)
                score = gain * q_sim

                if score > best_score:

                    best_score = score
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def predict(self, X):

        predictions = [self._traverse_tree(x, self.root) for x in X]

        return np.array(predictions)

    def _traverse_tree(self, x, node):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def score(self, X, y):

        predictions = self.predict(X)

        return np.mean(predictions == y)
