import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

class SimpleModel():

    def __init__(self):
        self.clf = None

    def train(self, X_train, y_train):
        self.clf = DecisionTreeClassifier()
        self.clf.fit(X_train,y_train)

    def predict(self, X):
        y_pred = self.clf.predict(X)
        return y_pred
    
    def serialize(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.clf, f)

    @staticmethod
    def deserialize(fname):
        model = SimpleModel()
        with open(fname, 'rb') as f:
            model.clf = pickle.load(f)

        return model
