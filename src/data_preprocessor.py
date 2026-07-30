import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:

    def __init__(self, featureset:pd.DataFrame, label):
        self.featureset = featureset
        self.label = label
        self.categorical_cols = self.featureset.select_dtypes(include='object').columns

    def process_dataset(self):
        self.featureset = pd.get_dummies(self.featureset, columns=self.categorical_cols, 
                                         prefix=self.categorical_cols, dtype='int')
        
        self.label = LabelEncoder().fit_transform(self.label)
        return self.featureset, self.label