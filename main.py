from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer


def main():
    try:
        from catboost import CatBoostClassifier
        from interpret.glassbox import ExplainableBoostingClassifier
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as e:
        print(f"Module Not Installed: {e}")
    
    dataset_path = "./dataset/03_feature_extracted_dataset.csv"
    loader = DataLoader(dataset_path, target_col="Parkinson's Disease status")
    loader.run()
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
        "Explainable Boosting Classifier": ExplainableBoostingClassifier(random_state=63),
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