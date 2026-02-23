#%% ExpCollect.py

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#########################################################
# (1) Experiment 설정 정보 (explanation)
#########################################################
explanation = {
    # 202: {"desc": "30초 RA-Pose", "centers": ["A", "B", "D"]},
    # 205: {"desc": "30초 Apnea-Pose", "centers": ["A", "B", "D"]},
    # 206: {"desc": "60초 RA-Pose, 30초 Sliding Window, Pretrained weight 사용", "centers": ["A", "D"]},
    # 208: {"desc": "60초 RA-Pose, 30초 Sliding Window... etc", "centers": ["A", "D"]},
    # 211: {"desc": "60초, A center만 사용", "centers": ["A"]},
    # 212: {"desc": "30초, A center만 사용", "centers": ["A"]},
    # 214: {"desc": "30초, 새 label-all, A center", "centers": ["A"]},
    # 215: {'desc': '30초, 새 label-r', 'centers': ['A', 'B', 'D']},
    # 217: {"desc": "30초, 새 label-r, rotation=45", "centers": ["A"]},
    217: {"desc": "30초, 새 label-r, rotation=45", "centers": ["A", 'B', 'D']},
    # 218: {"desc": "30초, 새 label-r, undersample normal", "centers": ["A", "B", "D"]},
    219: {"desc": "30초, 새 label-r, 후처리 confidence 기반", "centers": ["A", "B", "D"]},
    # 220: {"desc": "30초, Case별 weighted sampling 새 label-r, 후처리 confidence 기반", "centers": ["A", "B", "D"]},
    # 221: {"desc": "30초, Case별 RA only weighted sampling", "centers": ["A", "B", "D"]},
    # 222: {"desc": "30초, 219에 D fine-tuning", "centers": ["D"]},
    223: {"desc": "30초, 219에 D fine-tuning , 파라미터 정상화", "centers": ["D"]},
    224: {"desc": "30초, 219 버전 + Apnea/Hypopnea label", "centers": ["A", "B", "D"]},
    225: {"desc": "224랑 똑같은데, 후처리 제외", "centers": ["A", "B", "D"]},
    226: {"desc": "D_train 만 사용", 'centers': ["D"]},
    227: {"desc": "Apnea/Hypopnea, D_train 만 사용, 30초", 'centers': ["D"]},
    228: {"desc": "Apnea/Hypopnea, Transfer Learning D_train 만 사용, 30초", 'centers': ["D"]},
}
exp_numbers = list(explanation.keys())

#########################################################
# (2) 폴더 이름 결정, helper 함수들
#########################################################
def get_experiment_folders(centers):
    """Return folder names based on centers."""
    if set(centers) == {"A"}:
        return ["A_valid_vs_A_valid", "A_valid_vs_A_test"]
    elif set(centers) == {"A", "D"}:
        return [
            "A_valid_vs_A_valid", "A_valid_vs_A_test",
            "A_valid_vs_B_test", "A_valid_vs_D_test",
            "D_valid_vs_D_test"
        ]
    elif set(centers) == {"A", "B", "D"}:
        return [
            "A_valid_vs_A_valid", "A_valid_vs_A_test",
            "A_valid_vs_B_test", "A_valid_vs_D_test",
            "B_valid_vs_B_test", "D_valid_vs_D_test"
        ]
    elif set(centers) == {'D'}:
        return ["D_valid_vs_D_test"]
    return []

def parse_centers_from_file(file_path):
    """Parse centers from folder name (e.g. A_valid_vs_A_test -> {A, A})."""
    folder_name = os.path.basename(os.path.dirname(file_path))
    train_str, test_str = folder_name.split('_vs_')
    return {train_str[0], test_str[0]}

#########################################################
# (3) DL classification 결과 수집
#########################################################
def collect_experiment_results(exp_numbers):
    """Collect and aggregate classification metrics from experiments."""
    all_data = []
    files_by_center = {
        "A": "A/classification_metrics.csv",
        "B": "B/classification_metrics.csv",
        "D": "D/classification_metrics.csv"
    }

    for exp_number in exp_numbers:
        base_path = f"/home/dipark/analysis/{exp_number}"
        # Read only the centers defined for this experiment
        for center in explanation[exp_number]["centers"]:
            file_path = f"{base_path}/{files_by_center[center]}"
            if not os.path.exists(file_path):
                print(f"[Warning] Not found: {file_path}")
                continue
            df = pd.read_csv(file_path)
            df['Center'] = center
            df['Exp_Number'] = exp_number
            all_data.append(df)

    if not all_data:
        print("[INFO] No classification CSV found. Returning empty DataFrame.")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    # 예: 컬럼명 변경
    combined_df.columns = combined_df.columns.str.replace('Binary RA vs Non-RA', 'Binary Event vs Non-Event')

    # 예: pivot
    pivot_df = combined_df.pivot_table(
        index=['Exp_Number', 'Center'],
        columns='Metric',
        values=['Original 4-Class', 'Binary Event vs Non-Event', 'Binary Supine vs Non-Supine']
    )

    # 원하는 순서로 정렬
    ordered_columns = ['Accuracy', 'Precision', 'Recall', 'Macro-F1 Score', 'AUROC']
    pivot_df = pivot_df.reindex(ordered_columns, axis=1, level=1)
    
    # Macro-F1 Score column 이름을 F1 으로 변경
    pivot_df = pivot_df.rename(columns={'Macro-F1 Score': 'F1'}, level=1)

    # ----- (정렬 순서 복원) -----
    # MultiIndex 정렬
    pivot_df = pivot_df.sort_index(axis=0)  # (Exp_Number, Center) 순서대로
    # 필요 시 .sort_index(axis=1) 도 가능

    return pivot_df

#########################################################
# (4) Huber Regression 결과 수집
#########################################################
def collect_huber_regression_results(exp_numbers):
    """
    regression_metrics.json, 4class_metrics.json, binary_metrics.json
    -> "Huber Regression" 키 추출 -> DF
    """
    all_data = []
    for exp_number in exp_numbers:
        base_path = f"/home/dipark/analysis/{exp_number}/regression"
        folders = get_experiment_folders(explanation[exp_number]["centers"])

        # (a) 4class_metrics.json, binary_metrics.json
        for folder in folders:
            for filename in ["2class_standard_metrics.json", "4class_standard_metrics.json"]:
                file_path = f"{base_path}/{folder}/{filename}"
                if not os.path.exists(file_path):
                    print(f"[Warning] Not found: {file_path}")
                    continue
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    huber_results = data.get("Huber Regression", {})
                    if not huber_results:
                        continue
                    huber_results.update({
                        'Exp_Number': exp_number,
                        'Metric_Type': "AHI Severity 2-Class" if "2class" in filename else "AHI Severity 4-Class",
                        'Valid': folder.split('_vs_')[0],
                        'Test': folder.split('_vs_')[1],
                    })
                    all_data.append(huber_results)

        # (b) regression_metrics.json
        for folder in folders:
            file_path = os.path.join(base_path, folder, "4class_regression_metrics.json")
            if not os.path.exists(file_path):
                print(f"[Warning] Not found: {file_path}")
                continue
            with open(file_path, 'r') as f:
                data = json.load(f)
                huber_regression_results = data.get("Huber Regression", {})
                if not huber_regression_results:
                    continue
                huber_regression_results.update({
                    'Exp_Number': exp_number,
                    'Metric_Type': "Regression",
                    'Valid': folder.split('_vs_')[0],
                    'Test': folder.split('_vs_')[1],
                })
                all_data.append(huber_regression_results)

    combined_df = pd.DataFrame(all_data)
    if combined_df.empty:
        return combined_df
    
    combined_df.set_index(['Exp_Number','Valid','Test','Metric_Type'], inplace=True)
    stacked_df = combined_df.stack().reset_index()
    stacked_df.columns = ['Exp_Number','Valid','Test','Metric_Type','Metric','Value']
    
    pivot_df = stacked_df.pivot_table(
        index=['Exp_Number','Valid','Test'],
        columns=['Metric_Type','Metric'],
        values='Value',
        aggfunc='first'
    )
    pivot_df = pivot_df.reindex(['AHI Severity 2-Class','AHI Severity 4-Class','Regression'], axis=1, level=0)
    ordered_columns = ['accuracy','f1_score','precision','recall', 'auroc', 'R-squared','MAE','RMSE']
    pivot_df = pivot_df.sort_index(axis=1, level=[0,1])

    # (정렬 순서 복원) - index: (Exp_Number, Valid, Test)
    pivot_df = pivot_df.sort_index(axis=0)
    pivot_df = pivot_df.reindex(ordered_columns, axis=1, level=1)

    return pivot_df

#########################################################
# (5) 시각화 함수 (기존 구분선 복원)
#########################################################
def save_visualization(df, filename, title, invert_colors=False):
    """단일 heatmap 시각화 -> PNG, + 실험/센터 구분선 복원"""
    if df.empty:
        print(f"[WARNING] DataFrame is empty. Skip saving {filename}")
        return
    out_dir = "/home/dipark/analysis/overview"
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)

    plt.figure(figsize=(12, 8))
    cmap = 'coolwarm_r' if invert_colors else 'coolwarm'
    plt.imshow(df, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xticks(range(df.columns.size), df.columns, rotation=90)
    plt.yticks(range(df.index.size), df.index)

    # 값 표시
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            val = df.iloc[i, j]
            if pd.notnull(val):
                plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')

    # >>> 복원: "실험끼리" 혹은 "센터끼리" 구분선
    # 예: df.index = (Exp_Number, Center)라면 df_index_list[i][0]=Exp_Number
    df_index_list = df.index.tolist()
    for i in range(1, len(df_index_list)):
        # Compare 'Exp_Number' (df_index_list[i][0]) with previous row
        if df_index_list[i][0] != df_index_list[i-1][0]:
            # Exp_Number가 달라지는 지점에 수평선
            plt.axhline(y=i - 0.5, color='black', linewidth=1.5)

    plt.tight_layout()
    print(f"[INFO] Saving figure to {file_path}")
    plt.savefig(file_path, dpi=300)
    plt.close()

def save_combined_visualization(df1, df2, df3, df4, filename, title1, title2, title3, title4, invert_colors=False):
    """4개 subplot으로 df1, df2, df3, df4 시각화, + 실험/센터 구분선 복원"""
    out_dir = "/home/dipark/analysis/overview"
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)

    fig, axes = plt.subplots(1, 4, figsize=(16, 8))
    cmap1 = 'coolwarm'
    cmap2 = 'coolwarm'
    cmap3 = 'coolwarm_r' if invert_colors else 'coolwarm'
    cmap4 = 'coolwarm'

    # subplot 0: df1
    if not df1.empty:
        axes[0].imshow(df1, cmap=cmap1, aspect='auto')
        axes[0].set_title(title1)
        axes[0].set_xticks(range(df1.columns.size))
        axes[0].set_xticklabels(df1.columns, rotation=90)
        axes[0].set_yticks(range(df1.index.size))
        axes[0].set_yticklabels(df1.index)
        for i in range(df1.shape[0]):
            for j in range(df1.shape[1]):
                val = df1.iloc[i,j]
                if pd.notnull(val):
                    axes[0].text(j, i, f"{val:.3f}", ha='center', va='center', color='black')
        # ---- 구분선 복원 ----
        df1_index_list = df1.index.tolist()
        for i in range(1, len(df1_index_list)):
            if df1_index_list[i][:-1] != df1_index_list[i-1][:-1]:
                axes[0].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[0].set_title(f"No data for {title1}")
        axes[0].axis('off')

    # subplot 1: df2
    if not df2.empty:
        axes[1].imshow(df2, cmap=cmap2, aspect='auto')
        axes[1].set_title(title2)
        axes[1].set_xticks(range(df2.columns.size))
        axes[1].set_xticklabels(df2.columns, rotation=90)
        axes[1].set_yticks(range(df2.index.size))
        axes[1].set_yticklabels(['']*df2.index.size)
        for i in range(df2.shape[0]):
            for j in range(df2.shape[1]):
                val = df2.iloc[i,j]
                if pd.notnull(val):
                    axes[1].text(j, i, f"{val:.3f}", ha='center', va='center', color='black')
        # 구분선 복원
        df_index_list = df2.index.tolist()
        for i in range(1, len(df_index_list)):
            if df_index_list[i][:-1] != df_index_list[i-1][:-1]:
                axes[1].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[1].set_title(f"No data for {title2}")
        axes[1].axis('off')

    # subplot 2: df3
    if not df3.empty:
        axes[2].imshow(df3, cmap=cmap3, aspect='auto')
        axes[2].set_title(title3)
        axes[2].set_xticks(range(df3.columns.size))
        axes[2].set_xticklabels(df3.columns, rotation=90)
        axes[2].set_yticks(range(df3.index.size))
        axes[2].set_yticklabels(['']*df3.index.size)
        for i in range(df3.shape[0]):
            for j in range(df3.shape[1]):
                val = df3.iloc[i,j]
                if pd.notnull(val):
                    axes[2].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # 구분선 복원
        df_index_list = df3.index.tolist()
        for i in range(1, len(df_index_list)):
            if df_index_list[i][:-1] != df_index_list[i-1][:-1]:
                axes[2].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[2].set_title(f"No data for {title3}")
        axes[2].axis('off')

    # subplot 3: df4
    if isinstance(df4, pd.Series) and not df4.empty:
        tmp_df = df4.to_frame()
        axes[3].imshow(tmp_df, cmap=cmap4, aspect='auto')
        axes[3].set_title(title4)
        axes[3].set_xticks([0])
        axes[3].set_xticklabels([tmp_df.columns[0]], rotation=90)
        axes[3].set_yticks(range(tmp_df.shape[0]))
        axes[3].set_yticklabels(['']*tmp_df.shape[0])
        for i in range(tmp_df.shape[0]):
            val = tmp_df.iloc[i,0]
            if pd.notnull(val):
                axes[3].text(0, i, f"{val:.2f}", ha='center', va='center', color='black')
        df_index_list = df1.index.tolist()
        for i in range(1, len(df_index_list)):
            if df_index_list[i][:-1] != df_index_list[i-1][:-1]:
                axes[3].axhline(y=i - 0.5, color='black', linewidth=1.5)
    elif isinstance(df4, pd.DataFrame) and not df4.empty:
        axes[3].imshow(df4, cmap=cmap4, aspect='auto')
        axes[3].set_title(title4)
        axes[3].set_xticks(range(df4.columns.size))
        axes[3].set_xticklabels(df4.columns, rotation=90)
        axes[3].set_yticks(range(df4.index.size))
        axes[3].set_yticklabels(['']*df4.index.size)
        for i in range(df4.shape[0]):
            for j in range(df4.shape[1]):
                val = df4.iloc[i,j]
                if pd.notnull(val):
                    axes[3].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # 구분선 복원
        df_index_list = df4.index.tolist()
        for i in range(1, len(df_index_list)):
            if df_index_list[i][:-1] != df_index_list[i-1][:-1]:
                axes[3].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[3].set_title(f"No data for {title4}")
        axes[3].axis('off')

    plt.tight_layout()
    print(f"[INFO] Saving figure to {file_path}")
    plt.savefig(file_path, dpi=300)
    plt.close()

#########################################################
# (6) best_threshold_info.json 수집
#########################################################
def collect_best_threshold_results(exp_numbers):
    """
    best_threshold_info.json -> (4class/binary) => (threshold/accuracy/precision/recall/f1_score)
    -> (Valid,Test,Exp_Number) x (Metric_Type, Metric) pivot
    """
    all_data = []
    for exp_number in exp_numbers:
        base_path = f"/home/dipark/analysis/{exp_number}/regression"
        folders = get_experiment_folders(explanation[exp_number]["centers"])

        for folder in folders:
            file_path = os.path.join(base_path, folder, "best_threshold_info.json")
            if not os.path.exists(file_path):
                continue
            
            valid_set, test_set = folder.split('_vs_')
            with open(file_path, 'r') as f:
                data = json.load(f)  # {"4class":{...}, "binary":{...}}
            
            for classification_type in ['4class','binary']:
                if classification_type not in data:
                    continue
                info = data[classification_type]
                row = {
                    'Exp_Number': exp_number,
                    'Valid': valid_set,
                    'Test': test_set,
                    'Metric_Type': classification_type,
                    'Threshold': info.get('threshold'),
                    'Accuracy': info.get('accuracy'),
                    'Precision': info.get('precision'),
                    'Recall': info.get('recall'),
                    'F1': info.get('f1_score')
                }
                all_data.append(row)

    df = pd.DataFrame(all_data)
    if df.empty:
        print("[INFO] No best_threshold_info.json data found for any experiment.")
        return df
    
    df.set_index(['Valid','Test','Exp_Number','Metric_Type'], inplace=True)
    stacked_df = df.stack().reset_index()
    stacked_df.columns = ['Valid','Test','Exp_Number','Metric_Type','Metric','Value']
    
    pivot_df = stacked_df.pivot_table(
        index=['Valid','Test','Exp_Number'],
        columns=['Metric_Type','Metric'],
        values='Value',
        aggfunc='first'
    )
    pivot_df = pivot_df.sort_index(axis=1, level=[0,1])

    # 하위 컬럼 순서
    desired_cols = ['Threshold','Accuracy','Precision','Recall','F1']
    for top in ['4class','binary']:
        if top in pivot_df.columns.levels[0]:
            existing_sub = pivot_df[top].columns.intersection(desired_cols)
            pivot_df = pivot_df.reindex(existing_sub, axis=1, level=1)
    
    return pivot_df

#########################################################
# (6-추가) regression_plot_best_threshold_metrics.json 결과 수집
#########################################################
def collect_regression_plot_best_threshold_metrics(exp_numbers):
    from_exp_data = []
    for exp_number in exp_numbers:
        base_path = f"/home/dipark/analysis/{exp_number}/regression"
        folders = get_experiment_folders(explanation[exp_number]["centers"])

        for folder in folders:
            file_path = os.path.join(base_path, folder, "best_threshold_4class_regression_metrics.json")
            if not os.path.exists(file_path):
                continue
            valid_set, test_set = folder.split('_vs_')

            with open(file_path, 'r') as f:
                data = json.load(f)
                data = data.get("Huber Regression", {}) # 수정함
                if not data:
                    continue
                data.update({
                    'Exp_Number': exp_number,
                    'Valid': valid_set,
                    'Test': test_set
                })  
                from_exp_data.append(data)

    if not from_exp_data:
        print("[INFO] No best_threshold_4class_regression_metrics.json found.")
        return pd.DataFrame()

    df = pd.DataFrame(from_exp_data)
    index_cols = ['Exp_Number', 'Valid', 'Test']
    df.set_index(index_cols, inplace=True)

    # (model_name, metric_key) 튜플을 MultiIndex로 지정
    new_col_tuples = []
    for c in df.columns:
        if isinstance(c, tuple) and len(c) == 2:
            new_col_tuples.append(c)
        else:
            new_col_tuples.append((str(c), 'Unknown'))

    df.columns = pd.MultiIndex.from_tuples(new_col_tuples, names=["Model", "Metric"])
    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True)
    
    # Sort the DataFrame by (Valid, Test)
    df = df.sort_index(level=['Valid', 'Test'])
    
    return df

#########################################################
# (7) best_threshold_info.json + regression_plot_best_threshold_metrics
#     시각화: subplot(0)=binary, (1)=4class, (2)=error, (3)=r2
#########################################################
def save_combined_visualization_for_best_threshold(df_best_thr, df_regression, filename):
    """
    기존 best_threshold_info.json 에서 구한 df_best_thr: 
        (Valid,Test,Exp_Number) x ((4class|binary),(Threshold..))
    df_regression: 
        (Exp_Number,Valid,Test) x (Model,Metric) 구조
        예: (Huber Regression, MAE/RMSE/R-squared) 등

    총 4개 subplot:
      - subplot(0) : binary threshold 결과
      - subplot(1) : 4class threshold 결과
      - subplot(2) : regression errors (예: MAE, RMSE)
      - subplot(3) : regression R-squared
    """
    out_dir = "/home/dipark/analysis/overview"
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, filename)

    # 먼저 binary, 4class만 분리
    best_thr_4class = pd.DataFrame()
    best_thr_binary = pd.DataFrame()
    if '4class' in df_best_thr.columns.levels[0]:
        best_thr_4class = df_best_thr['4class']
    if 'binary' in df_best_thr.columns.levels[0]:
        best_thr_binary = df_best_thr['binary']

    # regression에서 error (MAE, RMSE) / r2(R-squared) 분리
    # df_regression 은 columns=(Model,Metric)
    # 모델이 여러 개 있을 수 있으므로, 일단 모든 (metric)이 MAE,RMSE에 해당하는 걸 error로, 
    # 'R-squared'는 r2로 모읍니다.
    df_error = pd.DataFrame()
    df_r2 = pd.DataFrame()
    if not df_regression.empty:
        # index 순서를 df_best_thr와 맞추려면 join/sort를 고려할 수도 있음
        # df_regression = df_regression.sort_index(axis=0)
        # error 메트릭 후보
        error_candidates = ['MAE', 'RMSE']
        # 실제 존재하는 (모델,메트릭)만 필터
        error_cols = []
        r2_cols = []
        for col in df_regression.columns:
            if col[1] in error_candidates:
                error_cols.append(col)
            elif col[1] == 'R-squared':
                r2_cols.append(col)

        if error_cols:
            df_error = df_regression[error_cols]
        if r2_cols:
            df_r2 = df_regression[r2_cols]

    df_error = df_regression[['MAE', 'RMSE']]
    df_r2 = df_regression[['R-squared']]
    # df_error 의 column 을 level별로 출력
    df_error.columns = df_error.columns.droplevel(1)
    df_r2.columns = df_r2.columns.droplevel(1)
    
    # 이제 4개의 subplot 생성
    fig, axes = plt.subplots(1, 4, figsize=(16, 8))

    # ========== subplot(0) : binary threshold ==========
    if not best_thr_binary.empty:
        axes[0].imshow(best_thr_binary, cmap='coolwarm', aspect='auto')
        axes[0].set_title("Best Thr (binary)")
        axes[0].set_xticks(range(best_thr_binary.columns.size))
        axes[0].set_xticklabels(best_thr_binary.columns, rotation=90)
        axes[0].set_yticks(range(best_thr_binary.index.size))
        axes[0].set_yticklabels(best_thr_binary.index)
        # 값 표시
        for i in range(best_thr_binary.shape[0]):
            for j in range(best_thr_binary.shape[1]):
                val = best_thr_binary.iloc[i, j]
                if pd.notnull(val):
                    axes[0].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # Exp_Number 달라지는 지점에 수평선
        idx_list = best_thr_binary.index.tolist()
        for i in range(1, len(idx_list)):
            # (Valid,Test,Exp_Number) 형태이므로 비교시 idx_list[i][2] (또는 -1) 비교
            if idx_list[i][:-1] != idx_list[i-1][:-1]:
                axes[0].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[0].set_title("No binary Data")
        axes[0].axis('off')

    # ========== subplot(1) : 4class threshold ==========
    if not best_thr_4class.empty:
        axes[1].imshow(best_thr_4class, cmap='coolwarm', aspect='auto')
        axes[1].set_title("Best Thr (4class)")
        axes[1].set_xticks(range(best_thr_4class.columns.size))
        axes[1].set_xticklabels(best_thr_4class.columns, rotation=90)
        axes[1].set_yticks(range(best_thr_4class.index.size))
        axes[1].set_yticklabels([''] * best_thr_4class.index.size)  # y축 라벨 안보이게
        for i in range(best_thr_4class.shape[0]):
            for j in range(best_thr_4class.shape[1]):
                val = best_thr_4class.iloc[i, j]
                if pd.notnull(val):
                    axes[1].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # Exp_Number 경계
        idx_list = best_thr_4class.index.tolist()
        for i in range(1, len(idx_list)):
            if idx_list[i][:-1] != idx_list[i-1][:-1]:
                axes[1].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[1].set_title("No 4class Data")
        axes[1].axis('off')

    # ========== subplot(2) : regression error (MAE,RMSE) ==========
    if not df_error.empty:
        axes[2].imshow(df_error, cmap='coolwarm_r', aspect='auto')  # 색상 반전(r) 가능
        axes[2].set_title("Regression Error")
        axes[2].set_xticks(range(df_error.columns.size))
        # 멀티컬럼 -> 문자열로 변환
        axes[2].set_xticklabels([f"{c}" for c in df_error.columns], rotation=90)
        axes[2].set_yticks(range(df_error.index.size))
        axes[2].set_yticklabels([''] * df_error.index.size)  # y축 라벨 숨김
        for i in range(df_error.shape[0]):
            for j in range(df_error.shape[1]):
                val = df_error.iloc[i, j]
                if pd.notnull(val):
                    axes[2].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # Exp_Number 경계
        idx_list = best_thr_4class.index.tolist()
        
        for i in range(1, len(idx_list)):
            if idx_list[i][:-1] != idx_list[i-1][:-1]:
                axes[2].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[2].set_title("N/A (Reg. Error)")
        axes[2].axis('off')

    # ========== subplot(3) : regression R-squared ==========
    if not df_r2.empty:
        axes[3].imshow(df_r2, cmap='coolwarm', aspect='auto')
        axes[3].set_title("Regression R-squared")
        axes[3].set_xticks(range(df_r2.columns.size))
        axes[3].set_xticklabels([f"{c}" for c in df_r2.columns], rotation=90)
        axes[3].set_yticks(range(df_r2.index.size))
        axes[3].set_yticklabels([''] * df_r2.index.size)
        for i in range(df_r2.shape[0]):
            for j in range(df_r2.shape[1]):
                val = df_r2.iloc[i, j]
                if pd.notnull(val):
                    axes[3].text(j, i, f"{val:.2f}", ha='center', va='center', color='black')
        # Exp_Number 경계
        idx_list = best_thr_4class.index.tolist()
        
        for i in range(1, len(idx_list)):
            if idx_list[i][:-1] != idx_list[i-1][:-1]:
                axes[3].axhline(y=i - 0.5, color='black', linewidth=1.5)
    else:
        axes[3].set_title("N/A (R-squared)")
        axes[3].axis('off')

    plt.tight_layout()
    print(f"[INFO] Saving figure to {file_path}")
    plt.savefig(file_path, dpi=300)
    plt.close()
    
def plot_best_threshold_combined_results(metrics_file_path: str, output_plot_path: str):
    with open(metrics_file_path, 'r') as f:
        metrics_data = json.load(f)

    models = list(metrics_data.keys())
    errors = [metrics_data[m]['RMSE'] for m in models]  # or 'MAE' instead
    r2_vals = [metrics_data[m]['R-squared'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, errors, width, color='coral', label='Error (RMSE)')
    bars2 = ax2.bar(x + width/2, r2_vals, width, color='skyblue', label='R-squared')

    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('R-squared')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    plt.title('AHI Regression Error & R-squared')
    ax1.legend(handles=[bars1, bars2], loc='best')

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.close()

#=======================================================
# 최종 실행 예시
#=======================================================
if __name__ == "__main__":
    out_dir = "/home/dipark/analysis/overview"
    os.makedirs(out_dir, exist_ok=True)

    #----------------------------------------------------
    # 1) experiment 정보 저장
    #----------------------------------------------------
    with open(os.path.join(out_dir, 'exp_details.json'), 'w') as f:
        json.dump(explanation, f, ensure_ascii=False, indent=4)

    #----------------------------------------------------
    # 2) DL classification 결과
    #   -> dl_classification_df
    #   -> dl_classification_results.csv
    #   -> dl_classification_results.png
    #----------------------------------------------------
    dl_classification_df = collect_experiment_results(exp_numbers)
    # 정렬(원래 방식대로)
    # df = df.reorder_levels(['Center','Exp_Number'], axis=0)
    # df = df.sort_index(axis=0)
    if not dl_classification_df.empty:
        dl_classification_df = dl_classification_df.reorder_levels(['Center','Exp_Number'], axis=0)
        dl_classification_df = dl_classification_df.sort_index(axis=0)
    
    dl_classification_df.to_csv(os.path.join(out_dir, 'dl_classification_results.csv'))

    save_visualization(
        dl_classification_df,
        "dl_classification_results.png",
        "DL Classification Results"
    )

    #----------------------------------------------------
    # 3) Huber Regression 결과
    #   -> ahi_classification_df
    #   -> ahi_classification_results.csv
    #   -> ahi_classification_results.png
    #----------------------------------------------------
    ahi_classification_df = collect_huber_regression_results(exp_numbers)
    if not ahi_classification_df.empty:
        ahi_classification_df = ahi_classification_df.reorder_levels(['Valid','Test','Exp_Number'], axis=0)
        ahi_classification_df = ahi_classification_df.sort_index(axis=0)
        ahi_classification_df.to_csv(os.path.join(out_dir, 'ahi_classification_results.csv'))

        # 예) severity + regression 등 분리
        ahi_severity_df = ahi_classification_df.loc[:, ['AHI Severity 2-Class','AHI Severity 4-Class']]
        ahi_regression_df = ahi_classification_df.loc[:, 'Regression']

        # 정렬
        ahi_classification_df = ahi_classification_df.sort_index(axis=0)
        # (사용자 환경에 맞게 인덱스 level 재배치도 가능)

        # 예시로 2-class,4-class, regression (R-squared, MAE,RMSE)
        if isinstance(ahi_severity_df.columns, pd.MultiIndex):
            ahi_2_class_df = ahi_severity_df['AHI Severity 2-Class']
            ahi_4_class_df = ahi_severity_df['AHI Severity 4-Class']
        else:
            ahi_2_class_df = ahi_severity_df[[c for c in ahi_severity_df.columns if '2-Class' in c]]
            ahi_4_class_df = ahi_severity_df[[c for c in ahi_severity_df.columns if '4-Class' in c]]
        
        # regression -> error / r-squared
        if isinstance(ahi_regression_df.columns, pd.MultiIndex):
            # multi-level
            if 'R-squared' in ahi_regression_df.columns.get_level_values(1):
                ahi_R_squared_df = ahi_regression_df.xs('R-squared', level=1, axis=1, drop_level=False)
            else:
                ahi_R_squared_df = pd.DataFrame()
            possible_err_cols = ['MAE','RMSE']
            existing_err_cols = [c for c in possible_err_cols if c in ahi_regression_df.columns.get_level_values(1)]
            ahi_error_df = ahi_regression_df.loc[:, (slice(None), existing_err_cols)]
        else:
            # 단일 level
            if 'R-squared' in ahi_regression_df.columns:
                ahi_R_squared_df = ahi_regression_df['R-squared']
            else:
                ahi_R_squared_df = pd.Series()
            err_cols = []
            for c in ['MAE','RMSE']:
                if c in ahi_regression_df.columns:
                    err_cols.append(c)
            ahi_error_df = ahi_regression_df[err_cols] if err_cols else pd.DataFrame()

        # 4subplot
        save_combined_visualization(
            ahi_2_class_df,
            ahi_4_class_df,
            ahi_error_df,
            ahi_R_squared_df,
            filename='ahi_classification_results.png',
            title1='AHI Severity 2-Class',
            title2='AHI Severity 4-Class',
            title3='AHI Regression Error',
            title4='AHI Regression R-squared',
            invert_colors=True
        )
    else:
        print("[INFO] ahi_classification_df is empty. No AHI classification results found.")

    #----------------------------------------------------
    # 4) best_threshold_info.json
    #----------------------------------------------------
    # (가) best_threshold_info.json -> 기존 방식
    best_thr_df = collect_best_threshold_results(exp_numbers)

    # (나) regression_plot_best_threshold_metrics.json -> 새로 추가된 함수
    df_reg_best_thr = collect_regression_plot_best_threshold_metrics(exp_numbers)

    # (다) 둘 모두 비어있지 않다면 => 함께 subplot으로 시각화
    if not best_thr_df.empty or not df_reg_best_thr.empty:
        save_combined_visualization_for_best_threshold(
            best_thr_df,
            df_reg_best_thr,
            filename="best_threshold_combined_results.png"
        )
    else:
        print("[INFO] No threshold or regression best-threshold data found.")

#%%