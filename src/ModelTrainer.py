import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve, average_precision_score

from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


class PlotManager:

    def __init__(self, results):
        self.results = results
        self.result_path = './analysis/'

    def plot_roc_prc_curve(self, model_name):
        tpr_array = np.array(self.results[model_name]['fpr'])
        tpr_mean = tpr_array.mean(axis=0)
        tpr_std  = tpr_array.std(axis=0)

        prec_array = np.array(self.results[model_name]['rec'])
        prec_mean = prec_array.mean(axis=0)
        prec_std  = prec_array.std(axis=0)

        fpr_interp    = np.linspace(0, 1, 100)
        recall_interp = np.linspace(0, 1, 100)

        mean_auc  = np.mean(self.results[model_name]['auc'])
        std_auc   = np.std(self.results[model_name]['auc'])
        mean_auprc = np.mean(self.results[model_name]['auprc'])
        std_auprc  = np.std(self.results[model_name]['auprc'])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=1200)

        ax = axes[0]
        ax.plot(fpr_interp, tpr_mean, color='#1b9e77', lw=2)
        ax.fill_between(fpr_interp, tpr_mean - tpr_std, tpr_mean + tpr_std,
                        color='#1b9e77', alpha=0.2, zorder=1)
        ax.plot([0, 1], [0, 1], linestyle='--', color='#aaaaaa', lw=1)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Mean AUROC (5-Fold CV)')

        ax.set_axisbelow(True)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

        auroc_handles = [
            Line2D([0], [0], color='#1b9e77', lw=2, label=f'Mean AUROC = {mean_auc:.3f}'),
            Patch(facecolor='#1b9e77', alpha=0.2, label=f'Standard Deviation ± {std_auc:.3f}')
        ]
        ax.legend(handles=auroc_handles, loc='lower right')

        ax = axes[1]
        ax.plot(recall_interp, prec_mean, color='#d95f02', lw=2)
        ax.fill_between(recall_interp, prec_mean - prec_std, prec_mean + prec_std,
                        color='#d95f02', alpha=0.2, zorder=1)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Mean AUPRC (5-Fold CV)')

        ax.set_axisbelow(True)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)

        auprc_handles = [
            Line2D([0], [0], color='#d95f02', lw=2, label=f'Mean AUPRC = {mean_auprc:.3f}'),
            Patch(facecolor='#d95f02', alpha=0.2, label=f'Standard Deviation ± {std_auprc:.3f}')
        ]
        ax.legend(handles=auprc_handles, loc='lower left')

        fig.tight_layout()
        plt.savefig(self.result_path + model_name + "_roc_prc_curve.png", dpi=1200)
        plt.close(fig)

    def plot_bar_chart(self):
        models = list(self.results.keys())

        mean_f1 = [np.mean(results['f1']) for results in self.results.values()]
        std_f1 = [np.std(results['f1']) for results in self.results.values()]
        mean_auc = [np.mean(results['auc']) for results in self.results.values()]
        std_auc = [np.std(results['auc']) for results in self.results.values()]
        mean_accuracy = [np.mean(results['accuracy']) for results in self.results.values()]
        std_accuracy = [np.std(results['accuracy']) for results in self.results.values()]
        mean_precision = [np.mean(results['precision']) for results in self.results.values()]
        std_precision = [np.std(results['precision']) for results in self.results.values()]
        mean_sensitivity = [np.mean(results['sensitivity']) for results in self.results.values()]
        std_sensitivity = [np.std(results['sensitivity']) for results in self.results.values()]
        mean_specificity = [np.mean(results['specificity']) for results in self.results.values()]
        std_specificity = [np.std(results['specificity']) for results in self.results.values()]

        metric_dataframes = [
            pd.DataFrame({"Metric": "F1 Score", "Algorithms": models, "Mean": mean_f1, "Std": std_f1}),
            pd.DataFrame({"Metric": "Accuracy", "Algorithms": models, "Mean": mean_accuracy, "Std": std_accuracy}),
            pd.DataFrame({"Metric": "Precision", "Algorithms": models, "Mean": mean_precision, "Std": std_precision}),
            pd.DataFrame({"Metric": "Sensitivity", "Algorithms": models, "Mean": mean_sensitivity, "Std": std_sensitivity}),
            pd.DataFrame({"Metric": "Specificity", "Algorithms": models, "Mean": mean_specificity, "Std": std_specificity}),
        ]

        metrics_df = pd.concat(metric_dataframes, ignore_index=True)

        model_colors = {
            "CatBoost": "#8da8d3",
            "Explainable Boosting Classifier": "#a1d99b",
            "Meta Learner": "#f6c969"
        }

        sns.set(style="whitegrid")
        plt.figure(figsize=(15, 6), dpi=1200)

        ax = sns.barplot(
            data=metrics_df,
            y="Metric",
            x="Mean",
            hue="Algorithms",
            palette=model_colors,
            errorbar=None
        )

        for i, row in metrics_df.iterrows():
            x = row["Mean"]
            xerr = row["Std"]
            metric_index = list(metrics_df["Metric"].unique()).index(row["Metric"])
            group_offset = -0.3 + 0.6 * models.index(row["Algorithms"]) / max(len(models) - 1, 1)
            y = metric_index + group_offset
            ax.errorbar(x, y, xerr=xerr, fmt='none', ecolor='black', capsize=2, lw=0.8)

            ax.annotate(f"{x:.4f} ± {xerr:.4f}",
                        (x + xerr + 0.005, y),
                        va='center', ha='left', fontsize=8)

        plt.title("Comparison of Models across Multiple Metrics")
        plt.xlabel("Score")
        plt.ylabel("Metrics")
        plt.xlim(0, metrics_df["Mean"].max() + metrics_df["Std"].max() + 0.05)
        plt.xticks(np.arange(0, 1.01, 0.1))

        plt.legend(title="Algorithms", loc='lower left', frameon=True)
        plt.tight_layout()
        plt.savefig(self.result_path+"comparison.png", dpi=1200)




class ModelTrainer:

    def __init__(self, featureset, label, models):
        self.featureset = featureset
        self.label = label
        self.models = models
        self.results = dict()

    def calculate_performance(self, y_test, y_pred, y_pred_proba):
        results = defaultdict(list)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

        fpr_interp = np.linspace(0, 1, 100)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        recall_interp = np.linspace(0, 1, 100)
        precision_interp = np.interp(recall_interp, recall[::-1], precision[::-1])

        results["f1"].append(f1_score(y_test, y_pred))
        results["accuracy"].append(accuracy_score(y_test, y_pred))
        results["precision"].append(precision_score(y_test, y_pred))
        results["sensitivity"].append(recall_score(y_test, y_pred, pos_label=1))
        results["specificity"].append(recall_score(y_test, y_pred, pos_label=0))
        results["fpr"].append(tpr_interp)
        results["auc"].append(auc(fpr, tpr))
        results["rec"].append(precision_interp)
        results["auprc"].append(auc(recall, precision))

        return results


    def k_fold_validation(self, model, n_splits=5, random_state=63):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_results = defaultdict(list)

        for fold, (train_index, test_index) in enumerate(skf.split(self.featureset, self.label)):
            X_train, X_test = self.featureset.iloc[train_index], self.featureset.iloc[test_index] 
            y_train, y_test = self.label[train_index], self.label[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            results = self.calculate_performance(y_test, y_pred, y_pred_proba)

            for key, value in results.items():
                fold_results[key].extend(value)

        return fold_results
        
    def save_analysis(self):
        plot_manager = PlotManager(self.results)
        
        for model_name, _ in self.models.items():
            plot_manager.plot_roc_prc_curve(model_name)
        
        plot_manager.plot_bar_chart()
        
    def print_results(self):
        for model_name, model in self.models.items():
            print(f"{model_name}")
            for metric, value in self.results[model_name].items():
                print(f"{metric}: {np.mean(value):0.4f} +/- {np.std(value):0.4f}")
            print("")

    def run(self):
        for model_name, model in tqdm(self.models.items()):
            print(f"Training {model_name}")
            self.results[model_name] = self.k_fold_validation(model)
        
        self.save_analysis()
        self.print_results()

            






