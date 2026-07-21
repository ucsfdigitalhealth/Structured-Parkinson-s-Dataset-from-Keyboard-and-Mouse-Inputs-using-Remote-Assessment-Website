from typing import Optional, List, Tuple
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd


class DataPreprocessor:

    def __init__(self, featureset:pd.DataFrame, label):
        self.featureset = featureset
        self.label = label
        self.categorical_cols = self.featureset.select_dtypes(include='object').columns
        self.numeric_cols = self.featureset.select_dtypes(include=['int64', 'float64']).columns

    def process_dataset(self):
        self.featureset = pd.get_dummies(self.featureset, columns=self.categorical_cols, 
                                         prefix=self.categorical_cols, dtype='int')
        
        scaler = MinMaxScaler()
        self.featureset[self.numeric_cols] = scaler.fit_transform(self.featureset[self.numeric_cols])
        self.label = LabelEncoder().fit_transform(self.label)
        return self.featureset, self.label