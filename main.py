from src.DataLoader import DataLoader
from src.DataPreprocessor import DataPreprocessor
from src.ModelTrainer import ModelTrainer

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression


def main():
    try:
        from catboost import CatBoostClassifier
        from interpret.glassbox import ExplainableBoostingClassifier
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
    except ImportError as e:
        print(f"Module Not Installed: {e.Message}")
    
    dataset_path = "./dataset/feature_engineered_dataset.csv"
    loader = DataLoader(dataset_path, target_col="Parkinson's Disease status")
    dataset = loader.run()
    featureset, label = loader.featureset, loader.label

    processor = DataPreprocessor(featureset, label)
    feature_processed, label_encoded = processor.process_dataset()

    base_models = [
        ('catboost', CatBoostClassifier(random_state=63, verbose=False)),
        ('xgboost', XGBClassifier(random_state= 63)),
        ('lightgbm', LGBMClassifier(random_state=63, verbose=-1))
    ]

    meta_learner = LogisticRegression()
    models = {
        "CatBoost": CatBoostClassifier(random_state=63, verbose=False),
        "Explanaible Boosting Classifier": ExplainableBoostingClassifier(random_state=63),
        "Meta Learner": StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            passthrough=True,
            cv=None
        )
    }

    trainer = ModelTrainer(feature_processed, label_encoded, models)
    trainer.run()

main()