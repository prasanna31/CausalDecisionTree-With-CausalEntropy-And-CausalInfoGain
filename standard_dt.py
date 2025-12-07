from sklearn.tree import DecisionTreeClassifier

class StandardDecisionTree:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, random_state=42):
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        self.is_fitted = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, X_test):
        if not self.is_fitted:
            raise Exception("Model has not been trained yet.")
        return self.model.predict(X_test)
