from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")


class BaseCausalDecisionTree(ABC):
    """
    Abstract Base Class for Causal Decision Trees.
    Handles tree management, recursion, pruning, and prediction.
    Subclasses must implement `_find_best_split`.
    """

    def __init__(self, max_height=5, min_samples=10):
        self.max_height = max_height
        self.min_samples = min_samples
        self.tree = None
        self.target = None
        self.predictors = None

    def fit(self, df, target, predictors):
        """Fits the tree to the data."""
        self.target = target
        self.predictors = predictors
        print(f"Building Tree (Max Height: {self.max_height})...")
        self.tree = self._tree_construct(df, predictors, height=0, edge_label=None)
        self._tree_prune(self.tree)
        print("Tree construction complete.")

    def predict(self, data):
        """Predicts outcomes for a dataset."""
        return data.apply(lambda row: self._predict_row(row, self.tree), axis=1)

    def _predict_row(self, row, node):
        """Recursive helper for row prediction."""
        if node['type'] == 'leaf':
            return node['label']
        
        feature_val = row[node['feature']]
        # Binary traversal (0 or 1)
        if feature_val in node['children']:
            return self._predict_row(row, node['children'][feature_val])
        else:
            return node['majority_class']

    def _tree_construct(self, current_data, available_attrs, height, edge_label):
        """
        Skeleton for tree construction.
        Relies on abstract method `_find_best_split` to decide branching.
        """
        # 1. Determine Majority Class (Fallback)
        if len(current_data) > 0:
            majority_class = current_data[self.target].mode()[0]
        else:
            majority_class = 0

        # 2. Initialize Node
        node = {
            'type': 'node',
            'feature': None,
            'children': {},
            'label': None,
            'edge_label': edge_label,
            'samples': len(current_data),
            'majority_class': majority_class,
            'meta': {}
        }

        # 3. Termination Checks
        if not available_attrs or height >= self.max_height or len(current_data) < self.min_samples:
            node['type'] = 'leaf'
            node['label'] = majority_class
            return node

        # 4. Find Best Split
        best_attr, split_meta = self._find_best_split(current_data, available_attrs)

        # 5. Check if a valid split was found
        if best_attr is None:
            node['type'] = 'leaf'
            node['label'] = majority_class
            return node

        # 6. Apply Split and Recurse
        node['feature'] = best_attr
        node['meta'] = split_meta
        
        next_attrs = [a for a in available_attrs if a != best_attr]

        for w in [0, 1]: # Assuming binary splitting
            subset = current_data[current_data[best_attr] == w]
            if len(subset) == 0:
                child = {'type': 'leaf', 'label': majority_class, 'edge_label': w, 'samples': 0}
            else:
                child = self._tree_construct(subset, next_attrs, height + 1, edge_label=w)
            node['children'][w] = child

        return node

    @abstractmethod
    def _find_best_split(self, data, available_attrs):
        """
        Abstract method to determine the best attribute to split on
        Must return:
            best_attr (str or None): The chosen attribute name
            split_meta (dict): Dictionary containing stats
        """
        pass

    def _tree_prune(self, node):
        """Standard post-pruning of redundant leaves."""
        if node['type'] == 'leaf': return node['label']
        
        labels = []
        is_all_leaves = True
        for key in node['children']:
            l = self._tree_prune(node['children'][key])
            labels.append(l)
            if node['children'][key]['type'] != 'leaf':
                is_all_leaves = False
        
        if is_all_leaves and len(set(labels)) == 1:
            node.update({'type': 'leaf', 'label': labels[0], 'children': {}})
            return labels[0]
        return None

    def print_tree(self, node=None, indent="", is_last=True, branch_label=""):
        if node is None:
            node = self.tree

        # Branch prefix (├── or └──)
        branch = "└── " if is_last else "├── "
        connector = indent + branch if branch_label else ""

        # Print leaf node
        if node['type'] == 'leaf':
            print(f"{connector}{branch_label}Leaf: Outcome={node['label']} (n={node['samples']})")
            return

        # Print split node
        meta_str = ", ".join([f"{k}: {v}" for k, v in node['meta'].items()])
        print(f"{connector}{branch_label}Split: {node['feature']} [{meta_str}]")

        # Prepare indentation for children
        child_indent = indent + ("    " if is_last else "│   ")

        # Iterate through children
        children_items = list(node['children'].items())
        for i, (val, child) in enumerate(children_items):
            last = (i == len(children_items) - 1)
            label = f"{node['feature']} == {val}: "
            self.print_tree(child, child_indent, last, label)

