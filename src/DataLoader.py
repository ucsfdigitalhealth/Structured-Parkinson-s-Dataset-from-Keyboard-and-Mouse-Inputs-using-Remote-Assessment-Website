from pathlib import Path
from typing import Optional, List
import pandas as pd


class DataLoader:

    def __init__(self, data_path: str, target_col: Optional[str] = None):
        self.data_path = Path(data_path)
        self.target_col = target_col
        self.featureset = None
        self.label = None

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data not found: {self.data_path}")
        return pd.read_csv(self.data_path)
    
    def run(self) -> pd.DataFrame:
        data = self.load()
        data = data.drop("Session ID", axis=1)
        data["Parkinson's Disease status"] = data["Parkinson's Disease status"].replace("suspectedpd", "pd")
        self.featureset = data.drop(self.target_col, axis=1)
        self.label= data[self.target_col]
        return data
    
