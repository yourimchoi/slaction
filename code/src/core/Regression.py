import os
import json
import argparse
import yaml
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.patches import Patch
from sklearn.linear_model import (
    LinearRegression, HuberRegressor, Ridge, Lasso
)
from matplotlib.lines import Line2D
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, mean_absolute_error, cohen_kappa_score, make_scorer
)

from sklearn.preprocessing import StandardScaler, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import NotFittedError


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but PolynomialFeatures was fitted with feature names"
)


# ============== plot ============== #
def plot_regression(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    y_pos: pd.Series,
    output_path: str,
    test_set: str,
    exp_number: int,
    train_sets: list,
    model_name: str,
    num_classes: int = 4,
    x_col: str = 'Prediction_RA',
    fit_intercept: bool = False,
    save_data: bool = True
):
    """
    regression plot
    """
    plt.figure(figsize=(14, 6))

    # prediction
    y_pred = model.predict(X[x_col].values.reshape(-1, 1))

    # x range
    x_min, x_max = -0.1, 1.02

    # y range
    y_min, y_max = -5, max(y) + 5

    # dense prediction plot
    x_dense = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    x_dense_df = pd.DataFrame(x_dense, columns=[x_col])
    y_pred_dense = model.predict(x_dense)

    # save
    if save_data:
        plot_data = {
            "x_dense": x_dense.tolist(),
            "y_pred_dense": y_pred_dense.tolist(),
            "x_test": X[x_col].tolist(),
            "y_test": y.tolist(),
            "y_pred": y_pred.tolist(),  
            "y_pos": y_pos.tolist(),
            "ra_thresholds": [],
        }
        for val in ([5, 15, 30] if num_classes == 4 else [15]):
            crossing_idx = np.where(np.diff(np.signbit(y_pred_dense - val)))[0]
            crossing_points = x_dense[crossing_idx]
            plot_data["ra_thresholds"].extend(crossing_points.flatten().tolist())
        
        plot_data["ra_thresholds"] = sorted(set(plot_data["ra_thresholds"]))

        data_output_path = output_path.replace(".png", "_plot_data.json")
        with open(data_output_path, "w") as f:
            json.dump(plot_data, f, indent=4)
    
    # AHI Class boundaries
    if num_classes == 2:
        ahi_boundaries = [15]
    else:  # 4-class
        ahi_boundaries = [5, 15, 30]

    # RA threshold 
    ra_thresholds = []
    for val in ahi_boundaries:
        crossing_idx = np.where(np.diff(np.signbit(y_pred_dense - val)))[0]
        crossing_points = x_dense[crossing_idx]
        ra_thresholds.extend(crossing_points.flatten().tolist())

    ra_thresholds = sorted(set(ra_thresholds))

    # section color
    ra_regions = [x_min] + ra_thresholds + [x_max]
    ahi_regions = [y_min] + ahi_boundaries + [y_max]

    for i in range(len(ra_regions) - 1):
        for j in range(len(ahi_regions) - 1):
            ra_start, ra_end = ra_regions[i], ra_regions[i + 1]
            ahi_start, ahi_end = ahi_regions[j], ahi_regions[j + 1]

            ra_mid = (ra_start + ra_end) / 2
            pred_y = np.interp(ra_mid, x_dense.flatten(), y_pred_dense)

            is_correct = (ahi_start <= pred_y <= ahi_end)

            if i == 0 and ahi_end <= 5:
                is_correct = True
            
            color = 'paleturquoise' if is_correct else 'mistyrose'
            plt.fill_between([ra_start, ra_end], ahi_start, ahi_end, color=color, alpha=0.2, zorder=1)
            

    # RA threshold vertical line
    for thr in ra_thresholds:
        pred_val = model.predict([[thr]])[0]
        plt.axvline(x=thr, color='indianred', linestyle='--', alpha=0.6, zorder=2)
        plt.text(
            thr+0.01, y_max * 0.95,
            f'RA Ratio={thr:.2f}',
            rotation=90, verticalalignment='top', fontsize=18
        )

    plt.plot(x_dense, y_pred_dense, 'r-', linewidth=2, zorder=3, label=model_name)
    plt.scatter(X[x_col], y, color='blue', alpha=0.6, zorder=4, label='Actual AHI')

    # AHI boundary
    if num_classes == 2:
        for val, label_text in zip(ahi_boundaries, ['Moderate']):
            plt.axhline(y=val, color='gray', linestyle='--', alpha=0.7, zorder=2)
            plt.text(x_max * 1.02, val, f'AHI={val}\n{label_text}', verticalalignment='center')
    else:
        for val, label_text in zip(ahi_boundaries, ['Mild', 'Moderate', 'Severe']):
            plt.axhline(y=val, color='gray', linestyle='--', alpha=0.7, zorder=2)
            plt.text(x_max * 1.02, val, f'AHI={val}\n{label_text}', verticalalignment='center')

    
    plt.grid(True, alpha=0.3, zorder=1)
    plt.title(f'Exp{exp_number} - {model_name}\nTrain: {"+".join(train_sets)} / Test: {test_set}')
    plt.xlabel("RA Ratio", fontsize=20)
    plt.ylabel('AHI', fontsize=20)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    legend_elements = [
        Line2D([0], [0], color='indianred', linewidth=2, label='Regression'),
        Line2D([0], [0], color='gray', linestyle='--', label='AHI Boundaries'),
        Line2D([0], [0], color='indianred', linestyle='--', alpha=0.7, label='RA Decision Points'),
        Patch(facecolor='paleturquoise', alpha=0.3, label='Correct Region'),
        Patch(facecolor='mistyrose', alpha=0.3, label='Incorrect Region'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', alpha=0.8, label='Actual AHI', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.5, 1), fontsize=18)
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
        
# ============== Positional ============== #
def evaluate_positional_classification(
    model,
    X_supine: pd.Series,
    X_non_supine: pd.Series,
    y_positional: pd.Series
):

    pred_supine_ahi = model.predict(X_supine.values.reshape(-1, 1))
    pred_non_supine_ahi = model.predict(X_non_supine.values.reshape(-1, 1))

    pred_positional_bool = (pred_supine_ahi - 2.0 * pred_non_supine_ahi) > 0
    pred_positional = pred_positional_bool.astype(int)

    acc = accuracy_score(y_positional, pred_positional)

    prec_weighted = precision_score(y_positional, pred_positional, average='weighted', zero_division=0)
    prec_macro = precision_score(y_positional, pred_positional, average='macro', zero_division=0)

    rec_weighted = recall_score(y_positional, pred_positional, average='weighted')
    rec_macro = recall_score(y_positional, pred_positional, average='macro')

    f1_weighted = f1_score(y_positional, pred_positional, average='weighted')
    f1_macro = f1_score(y_positional, pred_positional, average='macro')

    return {
        'accuracy': acc,
        'precision_weighted': prec_weighted,
        'precision_macro': prec_macro,
        'recall_weighted': rec_weighted,
        'recall_macro': rec_macro,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
    }
    
    
# ============== ExperimentRunner ============== #
class ExperimentRunner:
    def __init__(self, exp_number: str, epsilon: float = 1.35, fit_intercept: bool = False):
        self.exp_number = exp_number
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.config = {}
        self.train_sets = []
        self.test_sets = []
        self.label_dir = ""
        self.control_confidence = False  # Default value
        self.skip_threshold_optimization = False
        self.signal = False

        self.ahi_data = pd.DataFrame()

        self.vis_models = ['Polynomial Regression']

    
    def load_config(self, yaml_provided=True):
        if yaml_provided:
            yaml_path = f'./config/{self.exp_number}.yaml'
            with open(yaml_path, 'r') as f:
                self.config = yaml.safe_load(f)

        self.train_sets = self.config.get('train_sets', ['A_valid'])
        self.test_sets = self.config.get('test_sets', ['A_valid', 'A_test', 'B_test', 'D_test'])
        self.label_dir = self.config.get('label_dir', None)
        self.control_confidence = self.config.get('control_confidence', True)
        self.skip_threshold_optimization = self.config.get('skip_threshold_optimization', True)
        self.signal = self.config.get('signal', False)

        
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if 'Case_num' in df.columns:
            df.rename(columns={'Case_num': 'Case_Number'}, inplace=True)
        return df

    
    def create_prediction_ra_column(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:

        if self.signal:
            df['Prediction_RA'] = df['prob_event']
        else:
            def parse_ra_probability(prob_str: str) -> float:
                prob_str = prob_str.strip('[]')
                probs = [float(x) for x in prob_str.split()]
                return probs[2] + probs[3]
    
            df['Prediction_RA'] = df['probabilities'].apply(
                lambda x: parse_ra_probability(x)
            )
            
        def parse_pose(pose: int) -> str:
            if pose == 0 or pose == 2:
                return 'Supine'
            elif pose == 1 or pose == 3:
                return 'Non-Supine'
            else:
                print(f'Warning: Unknown pose value {pose}')
        
        df['Prediction_Pose'] = df['prediction'].apply(
            lambda x: parse_pose(x)
        )
        return df

    
    @staticmethod
    def aggregate_prediction_ra(df: pd.DataFrame, control_confidence=False) -> pd.DataFrame:
        
        info_all = pd.read_csv('/home/data_6/yrchoi/Sleep_Video_OSA/Data_analysis/Labeling/Labels_30sec/Data_info/all_info.csv')
        
        df = df.copy()

        df['RA_Binary'] = (df['Prediction_RA'] > 0.5).astype(int)
    
        if not control_confidence:
            df_RA = df.groupby('Case_Number')['RA_Binary'].mean().reset_index()
            df_supine_RA = df[df['Prediction_Pose'] == 'Supine'].groupby('Case_Number')['RA_Binary'].mean().reset_index()
            df_non_supine_RA = df[df['Prediction_Pose'] == 'Non-Supine'].groupby('Case_Number')['RA_Binary'].mean().reset_index()
    
            df_RA.rename(columns={'RA_Binary': 'Prediction_RA'}, inplace=True)
            df_supine_RA.rename(columns={'RA_Binary': 'Supine_RA'}, inplace=True)
            df_non_supine_RA.rename(columns={'RA_Binary': 'Non_Supine_RA'}, inplace=True)
    
            df_aggregated = df_RA.merge(df_supine_RA, on='Case_Number', how='outer').fillna(0)
            df_aggregated = df_aggregated.merge(df_non_supine_RA, on='Case_Number', how='outer').fillna(0)
    
        else:
            df['Prev_RA'] = df.groupby('Case_Number')['Prediction_RA'].shift(1, fill_value=0)
            df['Next_RA'] = df.groupby('Case_Number')['Prediction_RA'].shift(-1, fill_value=0)

            df['Is_Consecutive'] = (df['Prediction_RA'] > 0.5) & (
                ((df['Prev_RA'] > 0.5)) | ((df['Next_RA'] > 0.5))
            )

            df['Temp_Contribution'] = np.where(df['Prediction_RA'] > 0.5, 1, 0)  
            df.loc[df['Is_Consecutive'], 'Temp_Contribution'] = df['Prediction_RA']  

            df['count_event'] = 0
            df['Event_Start'] = (df['Prediction_RA'] > 0.5) & (~df['Is_Consecutive'].shift(1, fill_value=False))

            df['count_event'] = df.groupby('Case_Number')['Event_Start'].cumsum()

            df_RA = df.groupby('Case_Number')['Temp_Contribution'].mean().reset_index()
            df_supine_RA = df[df['Prediction_Pose'] == 'Supine'].groupby('Case_Number')['Temp_Contribution'].mean().reset_index()
            df_non_supine_RA = df[df['Prediction_Pose'] == 'Non-Supine'].groupby('Case_Number')['Temp_Contribution'].mean().reset_index()
    
            df_RA.rename(columns={'Temp_Contribution': 'Prediction_RA'}, inplace=True)
            df_supine_RA.rename(columns={'Temp_Contribution': 'Supine_RA'}, inplace=True)
            df_non_supine_RA.rename(columns={'Temp_Contribution': 'Non_Supine_RA'}, inplace=True)
    
            df_aggregated = df_RA.merge(df_supine_RA, on='Case_Number', how='outer').fillna(0)
            df_aggregated = df_aggregated.merge(df_non_supine_RA, on='Case_Number', how='outer').fillna(0)

        return df_aggregated


    @staticmethod
    def classify_ahi_severity(ahi: float) -> int:
        if ahi < 5:
            return 0
        elif 5 <= ahi < 15:
            return 1
        elif 15 <= ahi < 30:
            return 2
        else:
            return 3

    
    @staticmethod
    def classify_ahi_binary(ahi: float) -> int:
        return 1 if ahi >= 15 else 0

    
    @staticmethod
    def add_ahi_classification(df: pd.DataFrame) -> pd.DataFrame:
        df['AHI_Classification'] = df['AHI'].apply(ExperimentRunner.classify_ahi_severity)
        df['AHI_Binary_Classification'] = df['AHI'].apply(ExperimentRunner.classify_ahi_binary)
        return df

    
    def load_ahi_data(self):
        summary_file_path = self.config.get('summary_file_path', None)
        df_summary = self.load_data(summary_file_path)

        self.ahi_data = df_summary.loc[:, [
            'Case_Number', 'AHI',
            'Apnea_Index_in_the_Supine_Position',
            'Apnea_Index_in_the_Lateral_Position',
            'Hypopnea_Index_in_the_Supine_Position',
            'Hypopnea_Index_in_the_Lateral_Position'
        ]]
        self.ahi_data.fillna(0, inplace=True)

        # supine AHI, non-supine AHI column 생성
        self.ahi_data['Supine_AHI'] = (
            self.ahi_data['Apnea_Index_in_the_Supine_Position'] +
            self.ahi_data['Hypopnea_Index_in_the_Supine_Position']
        )
        self.ahi_data['Non-Supine_AHI'] = (
            self.ahi_data['Apnea_Index_in_the_Lateral_Position'] +
            self.ahi_data['Hypopnea_Index_in_the_Lateral_Position']
        )
        self.ahi_data.drop([
            'Apnea_Index_in_the_Supine_Position',
            'Apnea_Index_in_the_Lateral_Position',
            'Hypopnea_Index_in_the_Supine_Position',
            'Hypopnea_Index_in_the_Lateral_Position'
        ], axis=1, inplace=True)

        # Positional AHI = (Supine AHI - 2 * Non-Supine AHI) > 0
        self.ahi_data['Positional AHI'] = (
            self.ahi_data['Supine_AHI'] - 2 * self.ahi_data['Non-Supine_AHI'] > 0
        ).astype(int)

        self.ahi_data.sort_values('Case_Number', inplace=True)

    
    @staticmethod
    def filter_ahi_data(ahi_data: pd.DataFrame, df_aggregated: pd.DataFrame) -> pd.DataFrame:
        return ahi_data[ahi_data['Case_Number'].isin(df_aggregated['Case_Number'])]

    
    @staticmethod
    def merge_aggregated_with_ahi(df_aggregated: pd.DataFrame, ahi_data: pd.DataFrame) -> pd.DataFrame:
        return pd.merge(df_aggregated, ahi_data, on='Case_Number')

    
    def load_and_prepare_data(
        self, datasets: list, data_type='train', threshold: float = 0.5, verbose=True
    ):
        data_list = []
        all_case_numbers = set()
        label_dir = self.config.get('label_dir', None)
        fallback_dir = self.config.get('fallback_csv_file_path', None)
        results_dir = self.config.get('results_dir', None)

        for dataset in datasets:
            label_file = f'{label_dir}/{dataset}.csv'
            result_file = f'{results_dir}/{self.exp_number}/{dataset}_results.csv'

            if not os.path.exists(label_file):
                fallback_path = f'{fallback_dir}/{dataset}.csv'
                if os.path.exists(fallback_path):
                    label_file = fallback_path
                else:
                    raise FileNotFoundError(
                        f"[{dataset}] label file does net exist: {label_file}, {fallback_path}"
                    )
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"[{dataset}] result file does not exist: {result_file}")

            df_label = self.load_data(label_file)
            df_result = self.load_data(result_file)

            if len(df_label) != len(df_result):
                raise ValueError(
                    f"[{dataset}] label({len(df_label)}) vs. result({len(df_result)}) size mismatch."
                )

            df_merged = pd.concat([df_label, df_result], axis=1)
            df_merged = self.create_prediction_ra_column(df_merged, threshold=threshold)

            # RA aggregation
            df_aggregated = self.aggregate_prediction_ra(df_merged, control_confidence=self.control_confidence)

            all_case_numbers.update(df_aggregated['Case_Number'])
            data_list.append(df_aggregated)

        combined_data = pd.concat(data_list, ignore_index=True).drop_duplicates(subset='Case_Number')

        # AHI filtering and merging
        ahi_filtered = self.filter_ahi_data(self.ahi_data, combined_data)
        merged_data = self.merge_aggregated_with_ahi(combined_data, ahi_filtered)
        merged_data = self.add_ahi_classification(merged_data)

        if verbose:
            print(f"[{data_type.upper()} DATA] threshold={threshold:.2f} | samples={len(merged_data)}")

        # X, y 
        X = merged_data[['Prediction_RA']]
        y_reg = merged_data['AHI']
        y_class_4 = merged_data['AHI_Classification']
        y_class_2 = merged_data['AHI_Binary_Classification']

        # pos AHI
        X_supine = merged_data['Supine_RA']
        X_non_supine = merged_data['Non_Supine_RA']
        y_positional = merged_data['Positional AHI']

        y_supine = merged_data['Supine_AHI']
        y_non_supine = merged_data['Non-Supine_AHI']

        # merged_data.to_csv('./merged.csv', index=False)
        return X, y_reg, y_class_4, y_class_2, merged_data, X_supine, X_non_supine, y_positional, y_supine, y_non_supine
    

    def train_and_plot_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pos: pd.Series,
        output_path: str,
        test_set: str,
        num_classes: int = 4) -> dict:


        self.fit_intercept = False
    
        models = {
            'Polynomial Regression': make_pipeline(
                PolynomialFeatures(degree=3, include_bias=False),
                LinearRegression(fit_intercept=self.fit_intercept)
            )
        }
    
        all_predictions = {}
        trained_models = {}

        for model_name, model_instance in models.items():
            model_instance.fit(X_train[['Prediction_RA']], y_train)
            model_output_path = output_path.rsplit('.', 1)[0] + f'_{model_name.lower().replace(" ", "_")}.png'
            
            if model_name in self.vis_models:
                plot_regression(
                    model=model_instance,
                    X=X_test,
                    y=y_test,
                    y_pos=y_pos,
                    output_path=model_output_path,
                    test_set=test_set,
                    exp_number=self.exp_number,
                    train_sets=self.train_sets,
                    model_name=model_name,
                    num_classes=num_classes,
                    x_col='Prediction_RA',
                    fit_intercept=self.fit_intercept
                )
            
            trained_models[model_name] = model_instance
    
        metrics = {}
        for name, model_obj in models.items():
            y_pred = model_obj.predict(X_test[['Prediction_RA']])
            all_predictions[name] = y_pred
            metrics[name] = {
                'R-squared': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }

        if num_classes == 4:
            base_path = output_path.rsplit('.', 1)[0]
            metrics_path = f"{base_path}_regression_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

        return {
            'predictions': all_predictions,
            'models': trained_models,
            'metrics': metrics
        }


    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series):
        """
        Binary & Multi-class Classification eval metrics
        """
        accuracy = accuracy_score(y_true, y_pred)

        is_binary = len(set(y_true)) == 2
    
        precision = precision_score(y_true, y_pred, average='binary' if is_binary else 'weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary' if is_binary else 'weighted')
        f1_weighted = f1_score(y_true, y_pred, average='binary' if is_binary else 'weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')  
    
        # Class-wise F1-score 
        class_f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        class_f1_dict = {f'class_{i}_f1': f1 for i, f1 in enumerate(class_f1_scores)}
    
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,  
            **class_f1_dict  
        }


    @staticmethod
    def create_output_directory(base_dir: str, train_sets: list, test_set: str) -> str:
        dir_name = f"{'_'.join(train_sets)}_vs_{test_set}"
        output_dir = os.path.join(base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir


    def run(self):
        # (1) Conditionally load from YAML only if config is empty:
        base_dir = self.config.get('base_dir', None)
        output_base_dir = os.path.join(base_dir, f'{self.exp_number}', 'regression')
        os.makedirs(output_base_dir, exist_ok=True)

        experiments = self.config.get('experiments', None)
        if not experiments:
            experiments = [{
                'name': 'default',
                'train_sets': self.train_sets,
                'test_sets': self.test_sets
            }]

        for exp in experiments:
            self.train_sets = exp['train_sets']
            self.test_sets = exp['test_sets']
            print(f"Running experiment: {exp['name']}")

            # (1) train data load
            X_train, y_train_reg, y_train_class_4, y_train_class_2, train_data, \
                X_train_supine, X_train_nonsupine, y_train_positional, y_train_supine, y_train_nonsupine = \
                    self.load_and_prepare_data(self.train_sets, 'train', threshold=0.5, verbose=True)

            # (2) test
            for test_set in self.test_sets:
                X_test, y_test_reg, y_test_class_4, y_test_class_2, test_data, \
                    X_test_supine, X_test_nonsupine, y_test_positional, y_test_supine, y_test_nonsupine = \
                        self.load_and_prepare_data([test_set], 'test', threshold=0.5, verbose=True)

                overlap = set(train_data['Case_Number']).intersection(set(test_data['Case_Number']))
                if overlap:
                    print(f"[!] train/test overlap={len(overlap)}")

                output_dir = self.create_output_directory(output_base_dir, self.train_sets, test_set)

                # ---------------- (A) OSA severity classification (4class) ----------------
                plot_path_4class = os.path.join(output_dir, '4class.png')
                result_4class = self.train_and_plot_regression(
                    X_train, y_train_reg,
                    X_test, y_test_reg, y_test_positional,
                    output_path=plot_path_4class,
                    test_set=test_set,
                    num_classes=4
                )
                predictions_4class = result_4class['predictions']  
                trained_models_4class = result_4class['models']    

                y_true_dict_4class = {}
                y_pred_dict_4class = {}
                four_class_metrics = {}

                for model_name, y_pred_reg in predictions_4class.items():
                    y_pred_class = pd.Series(y_pred_reg).apply(self.classify_ahi_severity)

                    class_metrics = self.calculate_metrics(y_test_class_4, y_pred_class)
                    
                    four_class_metrics[model_name] = class_metrics
    
                    y_true_dict_4class[model_name] = y_test_class_4
                    y_pred_dict_4class[model_name] = y_pred_class
                    
                with open(os.path.join(output_dir, '4class_metrics.json'), 'w') as f:
                    json.dump(four_class_metrics, f, indent=4)

                # ---------------- (B) Positional AHI ----------------
                positional_metrics = {}
                for model_name in trained_models_4class:
                    model_obj = trained_models_4class[model_name]
                    pos_metrics = evaluate_positional_classification(
                        model_obj,
                        X_test_supine,
                        X_test_nonsupine,
                        y_test_positional
                    )
                    positional_metrics[model_name] = pos_metrics

                with open(os.path.join(output_dir, 'positional_metrics.json'), 'w') as f:
                    json.dump(positional_metrics, f, indent=4)

                # ---------------- (C) OSA classification (AHI > 15, binary) ----------------
                plot_path_2class = os.path.join(output_dir, '2class.png')
                result_2class = self.train_and_plot_regression(
                    X_train, y_train_reg,
                    X_test, y_test_reg, y_test_positional,
                    output_path=plot_path_2class,
                    test_set=test_set,
                    num_classes=2
                )
                predictions_2class = result_2class['predictions']

                y_true_dict_binary = {}
                y_pred_dict_binary = {}
                binary_metrics = {}
                for model_name, y_pred_reg in predictions_2class.items():
                    y_pred_class = pd.Series(y_pred_reg).apply(self.classify_ahi_binary)

                    
                    bi_metrics = self.calculate_metrics(y_test_class_2, y_pred_class)
                    binary_metrics[model_name] = bi_metrics

                    y_true_dict_binary[model_name] = y_test_class_2
                    y_pred_dict_binary[model_name] = y_pred_class

                with open(os.path.join(output_dir, '2class_metrics.json'), 'w') as f:
                    json.dump(binary_metrics, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description='Run regression analysis on experiment results.')
    parser.add_argument('--exp_number', type=str, help='Experiment number')
    args = parser.parse_args()

    if args.exp_number is not None:
        runner = ExperimentRunner(exp_number=args.exp_number, epsilon=1.35, fit_intercept=False)
        print(f'Running experiment: {args.exp_number}')
        runner.load_config(yaml_provided=True)
        runner.load_ahi_data()
    else:
        raise ValueError("Error: --exp_number must be provided.")
    runner.run()

if __name__ == "__main__":
    main()