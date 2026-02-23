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
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score, mean_absolute_error, cohen_kappa_score
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but HuberRegressor was fitted with feature names"
)

# ============== 공통 플롯 함수 ==============
def plot_regression(
    model,
    X: pd.DataFrame,
    y: pd.Series,
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
    회귀 플롯을 그리는 공통 함수.
    x_col 파라미터로 'Prediction_RA', 'Supine_RA', 'Non_Supine_RA' 등을 지정 가능.
    """
    plt.figure(figsize=(10, 8))

    # X_test_sorted, y_test_sorted
    sorted_idx = X[x_col].argsort()
    X_test_sorted = X.iloc[sorted_idx]
    y_test_sorted = y.iloc[sorted_idx]

    # 예측
    y_pred = model.predict(X[x_col].values.reshape(-1, 1))
    # y_pred = model.predict(X[[x_col]])
    y_pred_sorted = y_pred[sorted_idx]

    # x 범위
    x_min, x_max = -0.1, 1.1
    if X[x_col].min() < x_min:
        x_min = X[x_col].min() - 0.1
    if X[x_col].max() > x_max:
        x_max = X[x_col].max() + 0.1

    # y 범위
    y_min, y_max = -5, max(y_test_sorted) + 5

    # 촘촘한 구간으로 예측 곡선을 그리기
    x_dense = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    x_dense_df = pd.DataFrame(x_dense, columns=[x_col])
    y_pred_dense = model.predict(x_dense)

    # 데이터 저장 (추가)
    if save_data:
        plot_data = {
            "x_dense": x_dense.tolist(),
            "y_pred_dense": y_pred_dense.tolist(),
            "x_test_sorted": X_test_sorted[x_col].tolist(),
            "y_test_sorted": y_test_sorted.tolist(),
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
    
    # AHI Class 구간
    if num_classes == 2:
        ahi_boundaries = [15]
    else:  # 4-class
        ahi_boundaries = [5, 15, 30]

    # RA threshold (모델 예측치 vs AHI boundaries 교차점)
    ra_thresholds = []
    for val in ahi_boundaries:
        crossing_idx = np.where(np.diff(np.signbit(y_pred_dense - val)))[0]
        crossing_points = x_dense[crossing_idx]
        ra_thresholds.extend(crossing_points.flatten().tolist())

    ra_thresholds = sorted(set(ra_thresholds))

    # 구간별 채색
    ra_regions = [x_min] + ra_thresholds + [x_max]
    ahi_regions = [y_min] + ahi_boundaries + [y_max]

    for i in range(len(ra_regions) - 1):
        for j in range(len(ahi_regions) - 1):
            ra_start, ra_end = ra_regions[i], ra_regions[i + 1]
            ahi_start, ahi_end = ahi_regions[j], ahi_regions[j + 1]

            ra_mid = (ra_start + ra_end) / 2
            pred_y = model.predict(pd.DataFrame({x_col: [ra_mid]}).values.reshape(-1, 1))[0]

            is_correct = (ahi_start <= pred_y <= ahi_end)
            color = 'lightgreen' if is_correct else 'mistyrose'
            plt.fill_between([ra_start, ra_end], ahi_start, ahi_end, color=color, alpha=0.2, zorder=1)

    # RA threshold 수직선
    for thr in ra_thresholds:
        pred_val = model.predict([[thr]])[0]
        plt.axvline(x=thr, color='r', linestyle='--', alpha=0.7, zorder=2)
        plt.text(
            thr, y_max * 0.95,
            f'RA={thr:.2f}\nAHI={pred_val:.1f}',
            rotation=90, verticalalignment='top'
        )

    # 실제점 / 예측곡선
    plt.plot(x_dense, y_pred_dense, 'r-', linewidth=2, zorder=3, label=model_name)
    plt.scatter(X_test_sorted[x_col], y_test_sorted, color='blue', alpha=0.6, zorder=4, label='Actual AHI')

    # AHI boundary (수평선)
    if num_classes == 2:
        for val, label_text in zip(ahi_boundaries, ['Moderate']):
            plt.axhline(y=val, color='gray', linestyle='--', alpha=0.7, zorder=2)
            plt.text(x_max * 1.02, val, f'AHI={val}\n{label_text}', verticalalignment='center')
    else:
        for val, label_text in zip(ahi_boundaries, ['Mild', 'Moderate', 'Severe']):
            plt.axhline(y=val, color='gray', linestyle='--', alpha=0.7, zorder=2)
            plt.text(x_max * 1.02, val, f'AHI={val}\n{label_text}', verticalalignment='center')

    # 꾸미기
    plt.grid(True, alpha=0.3, zorder=1)
    plt.title(f'Exp{exp_number} - {model_name}\nTrain: {"+".join(train_sets)} / Test: {test_set}')
    plt.xlabel(x_col)
    plt.ylabel('AHI')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linewidth=2, label=model_name),
        Line2D([0], [0], color='gray', linestyle='--', label='AHI Boundaries'),
        Line2D([0], [0], color='r', linestyle='--', alpha=0.7, label='RA Decision Points'),
        Patch(facecolor='lightgreen', alpha=0.2, label='Correct Region'),
        Patch(facecolor='mistyrose', alpha=0.2, label='Incorrect Region'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Actual AHI', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()

    # 저장
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# ============== Positional 분류 평가 함수 ==============
def evaluate_positional_classification(
    model,
    X_supine: pd.Series,
    X_non_supine: pd.Series,
    y_positional: pd.Series
):
    """
    이미 학습된 회귀 모델(model)에 supine RA, non-supine RA 를 각각 넣어
    예측된 supine/non-supine AHI 를 얻고, 이를 통해 Positional 여부를 분류(1/0).
    y_positional 과 비교하여 분류 메트릭을 반환한다.
    """
    pred_supine_ahi = model.predict(X_supine.values.reshape(-1, 1))
    pred_non_supine_ahi = model.predict(X_non_supine.values.reshape(-1, 1))

    # (supine_ahi - 2 * nonsupine_ahi) > 0 → positional=1
    pred_positional_bool = (pred_supine_ahi - 2.0 * pred_non_supine_ahi) > 0
    pred_positional = pred_positional_bool.astype(int)

    # 분류 메트릭
    acc = accuracy_score(y_positional, pred_positional)
    prec = precision_score(y_positional, pred_positional, zero_division=0)
    rec = recall_score(y_positional, pred_positional)
    f1_ = f1_score(y_positional, pred_positional)
    kappa = cohen_kappa_score(y_positional, pred_positional)  # Add Cohen's Kappa

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1_,
        'kappa': kappa  # Add Kappa to return dictionary
    }

# ============== ExperimentRunner ==============
class ExperimentRunner:
    def __init__(self, exp_number: int, epsilon: float = 1.35, fit_intercept: bool = False):
        self.exp_number = exp_number
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.config = {}
        self.train_sets = []
        self.test_sets = []
        self.csv_file_path = ""
        self.control_confidence = False  # Default value
        self.skip_threshold_optimization = False

        # summary.csv에서 불러올 AHI 정보
        self.ahi_data = pd.DataFrame()

        # 회귀 결과를 시각화할 모델 리스트
        self.vis_models = ['Huber Regression']

    def load_config(self, yaml_provided=True):
        if yaml_provided:
            yaml_path = f'../../exp_config/{self.exp_number}.yaml'
            with open(yaml_path, 'r') as f:
                self.config = yaml.safe_load(f)

        self.train_sets = self.config.get('train_sets', ['A_valid'])
        self.test_sets = self.config.get('test_sets', ['A_valid', 'A_test', 'B_test', 'D_test'])
        self.csv_file_path = self.config.get('csv_file_path', '/home/raid/sleep_video/label')
        self.control_confidence = self.config.get('control_confidence', False)
        self.skip_threshold_optimization = self.config.get('skip_threshold_optimization', False)

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        if 'Case_num' in df.columns:
            df.rename(columns={'Case_num': 'Case_Number'}, inplace=True)
        return df

    @staticmethod
    def create_prediction_ra_column(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
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
        """
        Aggregates RA predictions by Case_Number.

        When control_confidence is False:
            - Calculates the mean of Prediction_RA for all positions.
            - Calculates the mean of Prediction_RA for Supine and Non-Supine positions separately.
        When control_confidence is True:
            - Adds a temporary column to handle contributions based on conditions:
            (1) 0 if Prediction_RA <= 0.5,
            (2) 1 if Prediction_RA > 0.5 but not consecutive,
            (3) Prediction_RA if consecutive and > 0.5.
            - Groups by Case_Number and averages the values.
            - Removes the temporary column after aggregation.
        """
        df['RA_Binary'] = (df['Prediction_RA'] > 0.5).astype(int)
        # df['RA_Binary'] = df['Prediction_RA']
        if control_confidence == False:
            df_RA = df.groupby('Case_Number')['RA_Binary'].mean().reset_index()
            df_supine_RA = (
            df[df['Prediction_Pose'] == 'Supine']
                .groupby('Case_Number')['RA_Binary']
                .mean()
                .reset_index()
            )
            df_non_supine_RA = (
            df[df['Prediction_Pose'] == 'Non-Supine']
                .groupby('Case_Number')['RA_Binary']
                .mean()
                .reset_index()
            )
            df_RA.rename(columns={'RA_Binary': 'Prediction_RA'}, inplace=True)
            df_supine_RA.rename(columns={'RA_Binary': 'Supine_RA'}, inplace=True)
            df_non_supine_RA.rename(columns={'RA_Binary': 'Non_Supine_RA'}, inplace=True)

            df_aggregated = pd.merge(df_RA, df_supine_RA, on='Case_Number', how='outer').fillna(0)
            df_aggregated = pd.merge(df_aggregated, df_non_supine_RA, on='Case_Number', how='outer').fillna(0)
        else:
            # Add a temporary column for the custom logic
            df['Prev_RA'] = df['Prediction_RA'].shift(1)
            df['Prev_Case'] = df['Case_Number'].shift(1)
            df['Is_Consecutive'] = (df['Case_Number'] == df['Prev_Case']) & (df['Prev_RA'] > 0.5)

            df['Temp_Contribution'] = 0
            df.loc[df['Prediction_RA'] > 0.5, 'Temp_Contribution'] = 1
            df.loc[df['Is_Consecutive'], 'Temp_Contribution'] = df['Prediction_RA']

            # Aggregate the temporary column
            df_RA = df.groupby('Case_Number')['Temp_Contribution'].mean().reset_index()
            df_supine_RA = (
                df[df['Prediction_Pose'] == 'Supine']
                .groupby('Case_Number')['Temp_Contribution']
                .mean()
                .reset_index()
            )
            df_non_supine_RA = (
                df[df['Prediction_Pose'] == 'Non-Supine']
                .groupby('Case_Number')['Temp_Contribution']
                .mean()
                .reset_index()
            )

            df_supine_RA.rename(columns={'Temp_Contribution': 'Supine_RA'}, inplace=True)
            df_non_supine_RA.rename(columns={'Temp_Contribution': 'Non_Supine_RA'}, inplace=True)
            df_RA.rename(columns={'Temp_Contribution': 'Prediction_RA'}, inplace=True)

            df_aggregated = pd.merge(df_RA, df_supine_RA, on='Case_Number', how='outer').fillna(0)
            df_aggregated = pd.merge(df_aggregated, df_non_supine_RA, on='Case_Number', how='outer').fillna(0)

            # Remove the temporary columns
            df.drop(columns=['Prev_RA', 'Prev_Case', 'Is_Consecutive', 'Temp_Contribution'], inplace=True)

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
        summary_file_path = self.config.get('summary_file_path', '/home/data_4/sleep_video/Info/summary.csv')
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
        label_dir = self.config.get('csv_file_path', '../../data/demo_labels')
        fallback_dir = self.config.get('fallback_csv_file_path', '../../data/demo_labels')
        results_dir = self.config.get('results_dir', '../../results/result')

        for dataset in datasets:
            label_file = f'{label_dir}/{dataset}.csv'
            result_file = f'{results_dir}/{self.exp_number}/{dataset}_results.csv'

            if not os.path.exists(label_file):
                fallback_path = f'{fallback_dir}/{dataset}.csv'
                if os.path.exists(fallback_path):
                    label_file = fallback_path
                else:
                    raise FileNotFoundError(
                        f"[{dataset}] 라벨 파일이 존재하지 않음: {label_file}, {fallback_path}"
                    )
            if not os.path.exists(result_file):
                raise FileNotFoundError(f"[{dataset}] result 파일이 존재하지 않음: {result_file}")

            df_label = self.load_data(label_file)
            df_result = self.load_data(result_file)

            if len(df_label) != len(df_result):
                raise ValueError(
                    f"[{dataset}] label({len(df_label)}) vs. result({len(df_result)}) 크기가 다름."
                )

            # 라벨 + 결과 결합
            df_merged = pd.concat([df_label, df_result], axis=1)

            # RA값 생성 (threshold 적용)
            df_merged = self.create_prediction_ra_column(df_merged, threshold=threshold)

            # RA aggregation
            df_aggregated = self.aggregate_prediction_ra(df_merged, control_confidence=self.control_confidence)

            all_case_numbers.update(df_aggregated['Case_Number'])
            data_list.append(df_aggregated)

        combined_data = pd.concat(data_list, ignore_index=True).drop_duplicates(subset='Case_Number')

        # AHI 필터 후 머지
        ahi_filtered = self.filter_ahi_data(self.ahi_data, combined_data)
        merged_data = self.merge_aggregated_with_ahi(combined_data, ahi_filtered)
        merged_data = self.add_ahi_classification(merged_data)

        if verbose:
            print(f"[{data_type.upper()} DATA] threshold={threshold:.2f} | samples={len(merged_data)}")

        # X, y 준비
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

        return X, y_reg, y_class_4, y_class_2, merged_data, X_supine, X_non_supine, y_positional, y_supine, y_non_supine

    def train_and_plot_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_path: str,
        test_set: str,
        num_classes: int = 4
    ) -> dict:
        """
        (1) 모델 학습
        (2) Prediction_RA 기준으로 회귀 플롯
        (3) 최종 예측값 반환 + (원한다면) 모델 객체도 반환
        """
        models = {
            'Linear Regression': LinearRegression(fit_intercept=self.fit_intercept),
            'Ridge Regression': Ridge(fit_intercept=self.fit_intercept),
            'Lasso Regression': Lasso(fit_intercept=self.fit_intercept),
            'Huber Regression': HuberRegressor(fit_intercept=self.fit_intercept, epsilon=self.epsilon)
        }

        all_predictions = {}
        trained_models = {}  # 각 모델별 학습된 모델 객체

        # (A) self.vis_models 에 대해서만 플롯
        # for model_name in self.vis_models:
        for model_name in models.keys():
            model_instance = models[model_name]
            # 학습
            model_instance.fit(X_train[['Prediction_RA']], y_train)

            # 플롯
            model_output_path = (
                output_path.rsplit('.', 1)[0] + f'_{model_name.lower().replace(" ", "_")}.png'
            )
            if model_name in self.vis_models:
                plot_regression(
                    model=model_instance,
                    X=X_test,
                    y=y_test,
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

        # (B) 모든 모델에 대한 예측/메트릭
        metrics = {}
        for name, model_obj in models.items():
            y_pred = model_obj.predict(X_test[['Prediction_RA']])
            all_predictions[name] = y_pred
            metrics[name] = {
                'R-squared': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            }

        # (C) metrics 저장 (num_classes==4일 때만)
        if num_classes == 4:
            base_path = output_path.rsplit('.', 1)[0]
            metrics_path = f"{base_path}_regression_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

        # (D) 모델 객체 딕셔너리도 같이 반환하여 Positional 성능평가 등에 재활용 가능
        return {
            'predictions': all_predictions,
            'models': trained_models,
            'metrics': metrics
        }

    @staticmethod
    def calculate_metrics(y_true: pd.Series, y_pred: pd.Series):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        kappa = cohen_kappa_score(y_true, y_pred)  # Add Cohen's Kappa calculation
        return accuracy, precision, recall, f1, kappa

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, save_path: str = None):
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_combined_confusion_matrix(
        y_true_dict: dict, y_pred_dict: dict, labels: list,
        title: str, exp_num: int, train_sets: list, test_set: str,
        save_path: str = None
    ):
        fig, axes = plt.subplots(1, len(y_true_dict), figsize=(4 * len(y_true_dict), 5))
        fig.suptitle(f'Exp{exp_num} - {title}\nTrain: {"+".join(train_sets)} / Test: {test_set}')

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax, (model_name, y_true) in zip(axes, y_true_dict.items()):
            y_pred = y_pred_dict[model_name]
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            ax.title.set_text(model_name)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def create_output_directory(base_dir: str, train_sets: list, test_set: str) -> str:
        dir_name = f"{'_'.join(train_sets)}_vs_{test_set}"
        output_dir = os.path.join(base_dir, dir_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    # ----------------------------------------------------------------------
    # threshold sweep 함수 (기존)
    # ----------------------------------------------------------------------
    def threshold_sweep_for_huber(
        self, thresholds, test_set: str, output_dir: str, criterion: str = 'f1'
    ):
        """
        thresholds 범위에 대해,
        매번 threshold로 load_and_prepare_data → Huber Regression 학습 → 4class/2class 변환 후 메트릭 측정.
        """
        metrics_4class = []
        metrics_binary = []

        progress_bar = tqdm(
            thresholds,
            desc=f"Threshold sweep ({test_set})",
            bar_format="{l_bar}{bar:20}{r_bar}",
            ncols=100
        )

        for thr in progress_bar:
            # (A) train set
            X_train, y_train_reg, y_train_4class, y_train_2class, _, _, _, _, _, _ = \
                self.load_and_prepare_data(self.train_sets, 'train', threshold=thr, verbose=False)
            # (B) test set
            X_test, y_test_reg, y_test_4class, y_test_2class, _, _, _, _, _, _ = \
                self.load_and_prepare_data([test_set], 'test', threshold=thr, verbose=False)

            # Huber 모델 학습
            huber_reg = HuberRegressor(fit_intercept=self.fit_intercept, epsilon=self.epsilon)
            huber_reg.fit(X_train, y_train_reg)
            y_pred_reg = huber_reg.predict(X_train)

            # Huber loss
            def huber_loss(y_true, y_pred, delta=1.0):
                error = y_true - y_pred
                is_small_error = np.abs(error) <= delta
                squared_loss = 0.5 * error**2
                linear_loss = delta * (np.abs(error) - 0.5 * delta)
                return np.mean(np.where(is_small_error, squared_loss, linear_loss))

            huber_loss_value = huber_loss(y_train_reg, y_pred_reg, delta=huber_reg.epsilon)

            # 4class 변환
            y_pred_4class = pd.Series(y_pred_reg).apply(self.classify_ahi_severity)
            acc4, prec4, rec4, f14, kappa4 = self.calculate_metrics(y_train_4class, y_pred_4class)
            metrics_4class.append((thr, acc4, prec4, rec4, f14, huber_loss_value, kappa4))

            # 2class 변환
            y_pred_2class = pd.Series(y_pred_reg).apply(self.classify_ahi_binary)
            acc2, prec2, rec2, f12, kappa2 = self.calculate_metrics(y_train_2class, y_pred_2class)
            metrics_binary.append((thr, acc2, prec2, rec2, f12, huber_loss_value, kappa2))

        arr_4class = np.array(metrics_4class)
        arr_binary = np.array(metrics_binary)

        # best threshold (4class)
        if criterion == 'accuracy':
            best_idx_4 = np.argmax(arr_4class[:, 1])
        elif criterion == 'precision':
            best_idx_4 = np.argmax(arr_4class[:, 2])
        elif criterion == 'recall':
            best_idx_4 = np.argmax(arr_4class[:, 3])
        elif criterion == 'huber_loss':
            best_idx_4 = np.argmin(arr_4class[:, 5])
        else:  # f1
            best_idx_4 = np.argmax(arr_4class[:, 4])
        best_thr_4class = arr_4class[best_idx_4, 0]

        # best threshold (2class)
        if criterion == 'accuracy':
            best_idx_2 = np.argmax(arr_binary[:, 1])
        elif criterion == 'precision':
            best_idx_2 = np.argmax(arr_binary[:, 2])
        elif criterion == 'recall':
            best_idx_2 = np.argmax(arr_binary[:, 3])
        elif criterion == 'huber_loss':
            best_idx_2 = np.argmin(arr_binary[:, 5])
        else:  # f1
            best_idx_2 = np.argmax(arr_binary[:, 4])
        best_thr_binary = arr_binary[best_idx_2, 0]

        # best threshold 로 최종 학습/테스트
        X_train_final, y_train_reg_final, y_train_4c_final, y_train_2c_final, _, _, _, _, _, _ = \
            self.load_and_prepare_data(self.train_sets, 'train', threshold=best_thr_4class)
        X_test_final, y_test_reg_final, y_test_4c_final, y_test_2c_final, _, _, _, _, _, _ = \
            self.load_and_prepare_data([test_set], 'test', threshold=best_thr_4class)

        # 4class 평가
        huber_reg_final = HuberRegressor(fit_intercept=self.fit_intercept, epsilon=self.epsilon)
        huber_reg_final.fit(X_train_final, y_train_reg_final)
        y_pred_final = huber_reg_final.predict(X_test_final)
        acc4_final, prec4_final, rec4_final, f14_final, kappa4_final = self.calculate_metrics(
            y_test_4c_final, pd.Series(y_pred_final).apply(self.classify_ahi_severity)
        )

        # 2class 평가
        X_train_final_bin, y_train_reg_final_bin, _, y_train_2c_final_bin, _, _, _, _, _, _ = \
            self.load_and_prepare_data(self.train_sets, 'train', threshold=best_thr_binary)
        X_test_final_bin, y_test_reg_final_bin, _, y_test_2c_final_bin, _, _, _, _, _, _ = \
            self.load_and_prepare_data([test_set], 'test', threshold=best_thr_binary)
        huber_reg_final_bin = HuberRegressor(fit_intercept=self.fit_intercept, epsilon=self.epsilon)
        huber_reg_final_bin.fit(X_train_final_bin, y_train_reg_final_bin)
        y_pred_final_bin = huber_reg_final_bin.predict(X_test_final_bin)
        acc2_final, prec2_final, rec2_final, f12_final, kappa2_final = self.calculate_metrics(
            y_test_2c_final_bin, pd.Series(y_pred_final_bin).apply(self.classify_ahi_binary)
        )

        best_threshold_info = {
            "4class": {
                "threshold": float(best_thr_4class),
                "accuracy": float(acc4_final),
                "precision": float(prec4_final),
                "recall": float(rec4_final),
                "f1_score": float(f14_final),
                "kappa": float(kappa4_final)
            },
            "binary": {
                "threshold": float(best_thr_binary),
                "accuracy": float(acc2_final),
                "precision": float(prec2_final),
                "recall": float(rec2_final),
                "f1_score": float(f12_final),
                "kappa": float(kappa2_final)
            }
        }

        # 4class 플롯
        best_plot_path = os.path.join(output_dir, 'best_threshold_4class.png')
        self.train_and_plot_regression(
            X_train_final, y_train_reg_final,
            X_test_final, y_test_reg_final,
            output_path=best_plot_path,
            test_set=f"{test_set}_bestThr_{best_thr_4class:.2f}",
            num_classes=4
        )

        # 2class 플롯
        best_plot_path_binary = os.path.join(output_dir, 'best_threshold_binary.png')
        self.train_and_plot_regression(
            X_train_final_bin, y_train_reg_final_bin,
            X_test_final_bin, y_test_reg_final_bin,
            output_path=best_plot_path_binary,
            test_set=f"{test_set}_bestThr_{best_thr_binary:.2f}",
            num_classes=2
        )

        # JSON 저장
        with open(os.path.join(output_dir, 'best_threshold_info.json'), 'w') as f:
            json.dump(best_threshold_info, f, indent=4)

    def run(self):
        # (1) Conditionally load from YAML only if config is empty:


        base_dir = self.config.get('base_dir', '../../results/analysis')
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

            # (1) train 데이터 로드(기본 threshold=0.5)
            X_train, y_train_reg, y_train_class_4, y_train_class_2, train_data, \
                X_train_supine, X_train_nonsupine, y_train_positional, y_train_supine, y_train_nonsupine = \
                    self.load_and_prepare_data(self.train_sets, 'train', threshold=0.5, verbose=True)

            # (2) 테스트 세트별 실행
            for test_set in self.test_sets:
                X_test, y_test_reg, y_test_class_4, y_test_class_2, test_data, \
                    X_test_supine, X_test_nonsupine, y_test_positional, y_test_supine, y_test_nonsupine = \
                        self.load_and_prepare_data([test_set], 'test', threshold=0.5, verbose=True)

                overlap = set(train_data['Case_Number']).intersection(set(test_data['Case_Number']))
                if overlap:
                    print(f"[주의] train/test overlap={len(overlap)}")

                output_dir = self.create_output_directory(output_base_dir, self.train_sets, test_set)

                # ---------------- (A) 회귀(4class) ----------------
                plot_path_4class = os.path.join(output_dir, '4class.png')
                result_4class = self.train_and_plot_regression(
                    X_train, y_train_reg,
                    X_test, y_test_reg,
                    output_path=plot_path_4class,
                    test_set=test_set,
                    num_classes=4
                )
                predictions_4class = result_4class['predictions']  # 모델별 예측값
                trained_models_4class = result_4class['models']    # 모델별 학습된 모델 객체

                # 4class 분류 메트릭
                y_true_dict_4class = {}
                y_pred_dict_4class = {}
                four_class_metrics = {}
                for model_name, y_pred_reg in predictions_4class.items():
                    y_pred_class = pd.Series(y_pred_reg).apply(self.classify_ahi_severity)
                    acc, prec, rec, f1_, kappa = self.calculate_metrics(y_test_class_4, y_pred_class)
                    four_class_metrics[model_name] = {
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1_,
                        'kappa': kappa
                    }
                    y_true_dict_4class[model_name] = y_test_class_4
                    y_pred_dict_4class[model_name] = y_pred_class
                    
                cm_4class_path = os.path.join(output_dir, '4class_standard_confusion.png')
                self.plot_combined_confusion_matrix(
                    y_true_dict_4class, y_pred_dict_4class,
                    labels=[0, 1, 2, 3],
                    title="4-Class Classification",
                    exp_num=self.exp_number,
                    train_sets=self.train_sets,
                    test_set=test_set,
                    save_path=cm_4class_path
                )
                with open(os.path.join(output_dir, '4class_standard_metrics.json'), 'w') as f:
                    json.dump(four_class_metrics, f, indent=4)

                # ---------------- (B) Positional AHI 성능 ----------------
                #   이미 학습된 모델 중 하나(또는 여러 개)에 대해 성능 확인
                #   예: Huber Regression
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

                    # 추가: supine / non-supine 회귀 플롯
                    #   x_col='Supine_RA' / 'Non_Supine_RA' 로 각각 그릴 수 있음
                    if model_name == 'Huber Regression':
                        # supine
                        supine_plot_path = os.path.join(output_dir, 'supine_regression.png')
                        X_test_supine_df = X_test.copy()
                        X_test_supine_df['Supine_RA'] = X_test_supine
                        plot_regression(
                            model=model_obj,
                            X=X_test_supine_df,
                            y=y_test_supine,
                            output_path=supine_plot_path,
                            test_set=f'{test_set}_supine',
                            exp_number=self.exp_number,
                            train_sets=self.train_sets,
                            model_name=model_name,
                            num_classes=4,
                            x_col='Supine_RA'
                        )

                        # non-supine
                        nonsupine_plot_path = os.path.join(output_dir, 'nonsupine_regression.png')
                        X_test_nonsupine_df = X_test.copy()
                        X_test_nonsupine_df['Non_Supine_RA'] = X_test_nonsupine
                        plot_regression(
                            model=model_obj,
                            X=X_test_nonsupine_df,
                            y=y_test_nonsupine,
                            output_path=nonsupine_plot_path,
                            test_set=f'{test_set}_nonsupine',
                            exp_number=self.exp_number,
                            train_sets=self.train_sets,
                            model_name=model_name,
                            num_classes=4,
                            x_col='Non_Supine_RA'
                        )

                with open(os.path.join(output_dir, 'positional_metrics.json'), 'w') as f:
                    json.dump(positional_metrics, f, indent=4)

                # ---------------- (C) 이진 분류(2class) ----------------
                plot_path_2class = os.path.join(output_dir, '2class.png')
                result_2class = self.train_and_plot_regression(
                    X_train, y_train_reg,
                    X_test, y_test_reg,
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
                    acc, prec, rec, f1_, kappa = self.calculate_metrics(y_test_class_2, y_pred_class)
                    binary_metrics[model_name] = {
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1_,
                        'kappa': kappa
                    }
                    y_true_dict_binary[model_name] = y_test_class_2
                    y_pred_dict_binary[model_name] = y_pred_class

                cm_binary_path = os.path.join(output_dir, '2class_standard_confusion.png')
                self.plot_combined_confusion_matrix(
                    y_true_dict_binary, y_pred_dict_binary,
                    labels=[0, 1],
                    title="Binary Classification",
                    exp_num=self.exp_number,
                    train_sets=self.train_sets,
                    test_set=test_set,
                    save_path=cm_binary_path
                )
                with open(os.path.join(output_dir, '2class_standard_metrics.json'), 'w') as f:
                    json.dump(binary_metrics, f, indent=4)

                # ---------------- (D) Huber 전용 Threshold Sweep ----------------
                if not self.skip_threshold_optimization:
                    thresholds = np.arange(0, 1.01, 0.01)
                    self.threshold_sweep_for_huber(
                        thresholds=thresholds,
                        test_set=test_set,
                        output_dir=output_dir,
                        criterion=self.config.get('criterion', 'f1')
                    )

def main():
    parser = argparse.ArgumentParser(description='Run regression analysis on experiment results.')
    parser.add_argument('--exp_number', type=int, help='Experiment number')
    args = parser.parse_args()

    if args.exp_number is not None:
        runner = ExperimentRunner(exp_number=args.exp_number, epsilon=1.35, fit_intercept=False)
        print(f'Running experiment: {args.exp_number}')
        runner.load_config(yaml_provided=True)
        runner.load_ahi_data()
    else:
        '''
        편집할 것
        (1) exp_number                      : 실험번호 (아무거나 쓰세요)
        (2) base_dir                        : 결과가 저장될 디렉토리
        (3) summary_file_path               : subject 별 label 정보
        (4) csv_file_path                   : 각 clip 에 대한 label 정보 ['A_valid.csv', 'A_test.csv', ...]
        (5) fallback_csv_file_path          : csv file path 를 한번 더 써주면 됨
        (6) results_dir                     : classfication 의 결과가 저장된 위치 ['A_valid_results.csv', 'A_test_results.csv', ...]
        (7) test_sets                       : 안 써도 됨
        (8) analysis_sets                   : 안 써도 됨
        (9) control_confidence              : 안 써도 됨
        (10) skip_threshold_optimization    : 안 써도 됨
        (11) experiments                    : 실험 설정
        
        디렉토리 형식 (중요)
        base_dir: 결과 지정할 아무 디렉토리 만들면 됨.
        summary_file_path: /data_AIoT4/sleep_video/Info/summary.csv
        csv_file_path: 여기 하위에 label csv 파일이 있음(A_train.csv, A_valid.csv). Case_num 만 잘 있으면 됨.
            Case_num    : subject number
        results_dir: 
            설명:
                inference 의 결과가 '순서대로' 저장되어 있음 (Case_num, epoch 번호 순으로 정렬)
            column 은 probability 하나만 필요함
                probabilities   : [확률0, 확률1, 확률2, 확률3] (str)
                0: Supine-Normal, 1: Non-Supine-Normal, 2: Supine-RA, 3: Non-Supine-RA
            {results_dir}/{exp_number}/A_valid_results.csv 와 같은 형식으로 저장되어 있어야 함.

        '''
        exp_number = 220 # (1) 실험번호 (아무거나 쓰세요)
        runner = ExperimentRunner(exp_number=exp_number, epsilon=1.35, fit_intercept=False)  # Default exp_number
        print(f'Running experiment: {exp_number}')
        runner.config = {
            'criterion': 'f1',
            
            'base_dir': '../../results/analysis',                                # 결과가 저장될 디렉토리
            'summary_file_path': '/home/data_4/sleep_video/Info/summary.csv', # /data_AIoT4/sleep_video/Info/summary.csv   (subject 별 label 정보)
            'csv_file_path': '../../data/demo_labels',                        # 각 clip 에 대한 label 정보 ['A_valid.csv', 'A_test.csv', ...]
            'fallback_csv_file_path': '../../data/demo_labels',               # csv file path 를 한번 더 써주면 됨
            'results_dir': '../../results/result',                           # classfication 의 결과가 저장된 위치 ['{exp_number}/A_valid_results.csv', '{exp_number}/A_test_results.csv', ...]
            
            'test_sets': ['A_valid', 'A_test', 'B_valid', 'B_test', 'D_valid', 'D_test'],
            'analysis_sets': ['A', 'B', 'D'],
            
            'control_confidence': True,
            'skip_threshold_optimization': True,
            
            'experiments': [
                {
                    'name': 'A_valid_vs_A_valid',
                    'train_sets': ['A_valid'],
                    'test_sets': ['A_valid']
                },
                {
                    'name': 'A_valid_vs_A_test',
                    'train_sets': ['A_valid'],
                    'test_sets': ['A_test']
                },
                {
                    'name': 'A_valid_vs_B_test',
                    'train_sets': ['A_valid'],
                    'test_sets': ['B_test']
                },
                {
                    'name': 'A_valid_vs_D_test',
                    'train_sets': ['A_valid'],
                    'test_sets': ['D_test']
                },
                {
                    'name': 'B_valid_vs_B_test',
                    'train_sets': ['B_valid'],
                    'test_sets': ['B_test']
                },
                {
                    'name': 'D_valid_vs_D_test',
                    'train_sets': ['D_valid'],
                    'test_sets': ['D_test']
                }
            ]
        }
        runner.load_config(yaml_provided=False)
        runner.load_ahi_data()
    runner.run()

if __name__ == "__main__":
    main()
