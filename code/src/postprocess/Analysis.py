import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import argparse  # Add argparse module for command-line arguments
import yaml  # Add yaml module to read YAML configuration file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run analysis on experiment results.')
parser.add_argument('--exp_number', type=str, required=True, help='Experiment number')
args = parser.parse_args()

# Define experiment number
exp_number = args.exp_number

# Load YAML configuration file to get test sets
yaml_path = f'../../exp_config/{exp_number}.yaml'
with open(yaml_path, 'r') as file:
    config = yaml.safe_load(file)
test_sets = config['analysis_sets']

# Extract centers from test sets
centers = test_sets
all_metrics = {center: {} for center in centers}

for center in centers:
    # Set up the directory for saving analysis outputs
    output_dir = f'../../results/analysis/{exp_number}/{center}'
    os.makedirs(output_dir, exist_ok=True)

    # Load the CSV file
    file_path = f'../../results/result/{exp_number}/{center}_test_results.csv'
    df = pd.read_csv(file_path)

    # Original 4-class classification metrics
    y_true = df['target']
    y_pred = df['prediction']

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')  # Calculate macro-F1 score
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=["Non-RA, Supine", "Non-RA, Non-Supine", "RA, Supine", "RA, Non-Supine"])

    # Calculate AUROC for each class in the original 4-class classification
    auroc_4_class = roc_auc_score(pd.get_dummies(y_true), pd.get_dummies(y_pred), average=None)
    
    # Calculate Cohen's Kappa for 4-class
    kappa_4_class = cohen_kappa_score(y_true, y_pred)

    # Convert NumPy arrays to lists for JSON serialization
    original_metrics = {
        "Accuracy": accuracy,
        "Precision": precision.tolist(),
        "Recall": recall.tolist(),
        "F1 Score": f1,  # Store macro-F1 score
        "Confusion Matrix": conf_matrix.tolist(),
        "Classification Report": class_report,
        "AUROC": auroc_4_class.tolist(),
        "Kappa": kappa_4_class  # Add Kappa score
    }

    # RA vs Non-RA binary classification metrics
    df['target_binary'] = df['target'].apply(lambda x: 0 if x in [0, 1] else 1)
    df['prediction_binary'] = df['prediction'].apply(lambda x: 0 if x in [0, 1] else 1)

    y_true_binary = df['target_binary']
    y_pred_binary = df['prediction_binary']

    accuracy_binary = accuracy_score(y_true_binary, y_pred_binary)
    precision_binary = precision_score(y_true_binary, y_pred_binary, average='weighted')
    recall_binary = recall_score(y_true_binary, y_pred_binary, average='weighted')
    f1_binary = f1_score(y_true_binary, y_pred_binary, average='weighted')  # Calculate macro-F1 score
    conf_matrix_binary = confusion_matrix(y_true_binary, y_pred_binary)
    class_report_binary = classification_report(y_true_binary, y_pred_binary, target_names=["Non-RA", "RA"])

    # Calculate AUROC for binary RA vs Non-RA classification
    auroc_binary = roc_auc_score(y_true_binary, y_pred_binary)
    
    # Calculate Cohen's Kappa for binary RA classification
    kappa_binary = cohen_kappa_score(y_true_binary, y_pred_binary)

    binary_metrics = {
        "Accuracy": accuracy_binary,
        "Precision": precision_binary,
        "Recall": recall_binary,
        "F1 Score": f1_binary,  # Store macro-F1 score
        "Confusion Matrix": conf_matrix_binary.tolist(),
        "Classification Report": class_report_binary,
        "AUROC": auroc_binary,
        "Kappa": kappa_binary  # Add Kappa score
    }

    # Supine vs Non-Supine binary classification metrics
    df['target_supine'] = df['target'].apply(lambda x: 0 if x in [0, 2] else 1)
    df['prediction_supine'] = df['prediction'].apply(lambda x: 0 if x in [0, 2] else 1)

    y_true_supine = df['target_supine']
    y_pred_supine = df['prediction_supine']

    accuracy_supine = accuracy_score(y_true_supine, y_pred_supine)
    precision_supine = precision_score(y_true_supine, y_pred_supine, average='weighted')
    recall_supine = recall_score(y_true_supine, y_pred_supine, average='weighted')
    f1_supine = f1_score(y_true_supine, y_pred_supine, average='weighted')  # Calculate macro-F1 score
    conf_matrix_supine = confusion_matrix(y_true_supine, y_pred_supine)
    class_report_supine = classification_report(y_true_supine, y_pred_supine, target_names=["Supine", "Non-Supine"])

    # Calculate AUROC for binary Supine vs Non-Supine classification
    auroc_supine = roc_auc_score(y_true_supine, y_pred_supine)
    
    # Calculate Cohen's Kappa for supine classification
    kappa_supine = cohen_kappa_score(y_true_supine, y_pred_supine)

    supine_metrics = {
        "Accuracy": accuracy_supine,
        "Precision": precision_supine,
        "Recall": recall_supine,
        "F1 Score": f1_supine,  # Store macro-F1 score
        "Confusion Matrix": conf_matrix_supine.tolist(),
        "Classification Report": class_report_supine,
        "AUROC": auroc_supine,
        "Kappa": kappa_supine  # Add Kappa score
    }

    # Save metrics to a CSV file
    metrics_csv_path = os.path.join(output_dir, 'classification_metrics.csv')
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "Macro-F1 Score", "AUROC", "Kappa"],  # Add Kappa to metrics list
        "Original 4-Class": [original_metrics["Accuracy"], np.mean(original_metrics["Precision"]), 
                           np.mean(original_metrics["Recall"]), original_metrics["F1 Score"], 
                           np.mean(original_metrics["AUROC"]), original_metrics["Kappa"]],
        "Binary RA vs Non-RA": [binary_metrics["Accuracy"], binary_metrics["Precision"], 
                               binary_metrics["Recall"], binary_metrics["F1 Score"], 
                               binary_metrics["AUROC"], binary_metrics["Kappa"]],
        "Binary Supine vs Non-Supine": [supine_metrics["Accuracy"], supine_metrics["Precision"], 
                                       supine_metrics["Recall"], supine_metrics["F1 Score"], 
                                       supine_metrics["AUROC"], supine_metrics["Kappa"]]
    })
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Labels for each class type in confusion matrices
    labels_4_class = ["Non-RA, Supine", "Non-RA, Non-Supine", "RA, Supine", "RA, Non-Supine"]
    labels_binary = ["Non-RA", "RA"]
    labels_supine = ["Supine", "Non-Supine"]

    # Normalize confusion matrices
    confusion_matrices_normalized = {
        "Original 4-Class": conf_matrix.astype(float) / conf_matrix.sum(axis=1, keepdims=True),
        "Binary RA vs Non-RA": conf_matrix_binary.astype(float) / conf_matrix_binary.sum(axis=1, keepdims=True),
        "Binary Supine vs Non-Supine": conf_matrix_supine.astype(float) / conf_matrix_supine.sum(axis=1, keepdims=True)
    }

    # Define a function to save confusion matrix plots
    def save_confusion_matrix_plot(matrix, labels, title, file_name, normalized=False):
        plt.figure(figsize=(6, 5))
        fmt = ".2f" if normalized else "g"  # Use 'g' for general format to handle both int and float
        sns.heatmap(matrix, annot=True, fmt=fmt, cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, file_name))
        plt.close()

    # Plot and save original, binary RA vs Non-RA, and binary Supine vs Non-Supine confusion matrices
    save_confusion_matrix_plot(conf_matrix, labels_4_class, "Confusion Matrix: Original 4-Class", "confusion_matrix_4_class.png")
    save_confusion_matrix_plot(conf_matrix_binary, labels_binary, "Confusion Matrix: RA vs Non-RA", "confusion_matrix_ra_vs_non_ra.png")
    save_confusion_matrix_plot(conf_matrix_supine, labels_supine, "Confusion Matrix: Supine vs Non-Supine", "confusion_matrix_supine_vs_non_supine.png")

    # Plot and save normalized confusion matrices
    save_confusion_matrix_plot(confusion_matrices_normalized["Original 4-Class"], labels_4_class, "Normalized Confusion Matrix: Original 4-Class", "normalized_confusion_matrix_4_class.png", normalized=True)
    save_confusion_matrix_plot(confusion_matrices_normalized["Binary RA vs Non-RA"], labels_binary, "Normalized Confusion Matrix: RA vs Non-RA", "normalized_confusion_matrix_ra_vs_non_ra.png", normalized=True)
    save_confusion_matrix_plot(confusion_matrices_normalized["Binary Supine vs Non-Supine"], labels_supine, "Normalized Confusion Matrix: Supine vs Non-Supine", "normalized_confusion_matrix_supine_vs_non_supine.png", normalized=True)

    # Bar plot of accuracy, precision, recall, F1 score, and AUROC
    metrics_values = {
        "Original 4-Class": [original_metrics["Accuracy"], np.mean(original_metrics["Precision"]), 
                           np.mean(original_metrics["Recall"]), original_metrics["F1 Score"], 
                           np.mean(original_metrics["AUROC"]), original_metrics["Kappa"]],
        "Binary RA vs Non-RA": [binary_metrics["Accuracy"], binary_metrics["Precision"], 
                               binary_metrics["Recall"], binary_metrics["F1 Score"], 
                               binary_metrics["AUROC"], binary_metrics["Kappa"]],
        "Binary Supine vs Non-Supine": [supine_metrics["Accuracy"], supine_metrics["Precision"], 
                                       supine_metrics["Recall"], supine_metrics["F1 Score"], 
                                       supine_metrics["AUROC"], supine_metrics["Kappa"]]
    }
    metrics_names = ["Accuracy", "Precision", "Recall", "Macro-F1 Score", "AUROC", "Kappa"]  # Update F1 Score to Macro-F1 Score

    fig, ax = plt.subplots(figsize=(12, 6))
    index = np.arange(len(metrics_names))
    bar_width = 0.25

    # Create bars for each scenario
    for i, (title, values) in enumerate(metrics_values.items()):
        bars = ax.bar(index + i * bar_width, values, bar_width, label=title, edgecolor='grey')
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # Add details to plot
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Comparison of Classification Metrics Across Scenarios for Center {center}", fontsize=14)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_comparison_bar_chart.png"))
    plt.close()

    print(f"Metrics and visualizations for center {center} have been successfully saved to {output_dir}.")

    # Store metrics for overall comparison
    all_metrics[center] = metrics_values

# Compare metrics across centers
overall_metrics = {metric: [] for metric in metrics_names}
for metric in metrics_names:
    for center in centers:
        overall_metrics[metric].append(all_metrics[center]["Original 4-Class"][metrics_names.index(metric)])

fig, ax = plt.subplots(figsize=(12, 6))
index = np.arange(len(metrics_names))
bar_width = 0.25

# Create bars for each center
colors = sns.color_palette("husl", len(centers))
for i, center in enumerate(centers):
    bars = ax.bar(index + i * bar_width, [overall_metrics[metric][i] for metric in metrics_names], bar_width, label=center, edgecolor='grey', color=colors[i])
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

# Add details to plot
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Comparison of Classification Metrics Across Centers", fontsize=14)
ax.set_xticks(index + bar_width)
ax.set_xticklabels(metrics_names, fontsize=10)
ax.legend()

plt.tight_layout()
plt.savefig(f'../../results/analysis/{exp_number}/overall_metrics_comparison_bar_chart.png')
plt.close()

print(f"Overall metrics comparison bar chart has been successfully saved to ../../results/analysis/{exp_number}.")
