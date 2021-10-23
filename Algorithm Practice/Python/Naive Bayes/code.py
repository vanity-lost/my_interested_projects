import numpy as np
import pandas as pd


class NB():
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = []

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.classes = np.unique(y)
        for outcome in self.classes:
            self.priors[outcome] = sum(y == outcome) / len(y)

        for feature in X.columns:
            self.likelihoods[feature] = {}
            for outcome in self.classes:
                outcome_count = sum(y == outcome)
                likelihood = X[feature][y[y == outcome].index.values.tolist()].value_counts().to_dict()
                for feat_val, count in likelihood.items():
                    self.likelihoods[feature][feat_val] = {}
                    self.likelihoods[feature][feat_val][outcome] = (count + 1) / (outcome_count + len(self.classes))
        return self

    def predict(self, data):
        results = []
        data = pd.DataFrame(data)
        X = np.array(data)

        for sample in X:
            probs = {}
            for outcome in self.classes:
                prob = np.log(self.priors[outcome])
                for feature, feat_val in zip(data.columns, sample):
                    try:
                        prob += np.log(self.likelihoods[feature][feat_val][outcome])
                    except KeyError:
                        prob += np.log(1/(len(self.likelihoods[feature])+1))

                probs[outcome] = prob
            results.append(max(probs, key=lambda x: probs[x]))
        return results
