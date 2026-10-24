# --OPTION--
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Model:
    def __init__(self):
      self.classifier = RandomForestClassifier() 
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X): # currntly returns 0 or 1
       return self.classifier.predict(X)

    def predict_prob(self, X):
       return self.classifier.predict_proba(X)[:, 1]
