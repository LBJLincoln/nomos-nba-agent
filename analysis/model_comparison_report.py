import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.calibration import calibration_curve
from pathlib import Path

def load_model_predictions(model_dir: str) -> dict:
    """
    Load predictions from multiple model directories.
    
    Parameters:
    model_dir (str): Directory containing model subdirectories
    
    Returns:
    dict: Dictionary with model names as keys and prediction data as values
    """
    predictions = {}
    model_paths = sorted(Path(model_dir).iterdir())
    
    for model_path in model_paths:
        if model_path.is_dir():
            model_name = model_path.name
            try:
                # Load predictions.csv
                pred_file = model_path / 'predictions.csv'
                if pred_file.exists():
                    df = pd.read_csv(pred_file)
                    predictions[model_name] = df
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
    
    return predictions

def compute_model_metrics(predictions: dict, true_labels: np.array) -> pd.DataFrame:
    """
    Compute performance metrics for all models.
    
    Parameters:
    predictions (dict): Dictionary of model predictions
    true_labels (np.array): True labels for comparison
    
    Returns:
    pd.DataFrame: DataFrame with model metrics
    """
    metrics = []
    
    for model_name, data in predictions.items():
        if 'probabilities' in data.columns:
            probs = data['probabilities'].values
            preds = (probs > 0.5).astype(int)
            
            brier = brier_score_loss(true_labels, probs)
            logloss = log_loss(true_labels, probs, eps=1e-15)
            accuracy = accuracy_score(true_labels, preds)
            
            # Calibration metrics
            prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=10)
            calibration_error = np.mean(np.abs(prob_true - prob_pred))
            
            metrics.append({
                'model': model_name,
                'brier_score': brier,
                'log_loss': logloss,
                'accuracy': accuracy,
                'calibration_error': calibration_error,
                'num_predictions': len(probs)
            })
    
    return pd.DataFrame(metrics)

def plot_model_comparison(metrics_df: pd.DataFrame):
    """
    Create comparison plots for all models.
    
    Parameters:
    metrics_df (pd.DataFrame): DataFrame with model metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Brier score comparison
    sns.barplot(data=metrics_df.sort_values('brier_score'), 
                x='brier_score', y='model', ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Brier Score Comparison')
    axes[0, 0].set_xlabel('Brier Score')
    axes[0, 0].set_ylabel('Model')
    
    # Accuracy comparison
    sns.barplot(data=metrics_df.sort_values('accuracy'), 
                x='accuracy', y='model', ax=axes[0, 1], palette='plasma')
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].set_xlabel('Accuracy')
    axes[0, 1].set_ylabel('Model')
    
    # Calibration error comparison
    sns.barplot(data=metrics_df.sort_values('calibration_error'), 
                x='calibration_error', y='model', ax=axes[1, 0], palette='coolwarm')
    axes[1, 0].set_title('Calibration Error Comparison')
    axes[1, 0].set_xlabel('Calibration Error')
    axes[1, 0].set_ylabel('Model')
    
    # Sample size comparison
    sns.barplot(data=metrics_df.sort_values('num_predictions'), 
                x='num_predictions', y='model', ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title('Sample Size Comparison')
    axes[1, 1].set_xlabel('Number of Predictions')
    axes[1, 1].set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_calibration_plots(predictions: dict, true_labels: np.array):
    """
    Generate calibration plots for all models.
    
    Parameters:
    predictions (dict): Dictionary of model predictions
    true_labels (np.array): True labels for comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (model_name, data) in enumerate(predictions.items()):
        if idx >= 4:  # Only show top 4 models
            break
            
        if 'probabilities' in data.columns:
            probs = data['probabilities'].values
            prob_true, prob_pred = calibration_curve(true_labels, probs, n_bins=10)
            
            axes[idx].plot(prob_pred, prob_true, marker='o', linewidth=2)
            axes[idx].plot([0, 1], [0, 1], linestyle='--', color='gray')
            axes[idx].set_title(f'{model_name} Calibration')
            axes[idx].set_xlabel('Predicted Probability')
            axes[idx].set_ylabel('True Probability')
            axes[idx].grid(True, alpha=0.3)
    
    for idx in range(idx + 1, 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('calibration_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table with model rankings.
    
    Parameters:
    metrics_df (pd.DataFrame): DataFrame with model metrics
    
    Returns:
    pd.DataFrame: Summary table with rankings
    """
    summary = metrics_df.copy()
    
    # Rank models
    for col in ['brier_score', 'log_loss', 'accuracy', 'calibration_error']:
        summary[f'{col}_rank'] = summary[col].rank(ascending=col in ['brier_score', 'log_loss', 'calibration_error'])
    
    # Calculate average rank
    summary['average_rank'] = summary[['brier_score_rank', 'log_loss_rank', 'accuracy_rank', 'calibration_error_rank']].mean(axis=1)
    
    # Sort by average rank
    summary = summary.sort_values('average_rank').reset_index(drop=True)
    
    return summary

def main():
    # Load data
    print("Loading model predictions...")
    predictions = load_model_predictions('models/')
    
    if not predictions:
        print("No model predictions found. Please ensure models/*/predictions.csv files exist.")
        return
    
    # Load true labels (assuming they're in the first model's data)
    first_model = next(iter(predictions.values()))
    true_labels = (first_model['actual'] > 0.5).astype(int).values
    
    # Compute metrics
    print("Computing model metrics...")
    metrics_df = compute_model_metrics(predictions, true_labels)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_model_comparison(metrics_df)
    generate_calibration_plots(predictions, true_labels)
    
    # Create summary
    print("Creating performance summary...")
    summary_df = create_performance_summary(metrics_df)
    
    # Save results
    print("Saving results...")
    metrics_df.to_csv('model_comparison_metrics.csv', index=False)
    summary_df.to_csv('model_comparison_summary.csv', index=False)
    
    # Print top models
    print("\nTOP PERFORMING MODELS:")
    print("="*60)
    print(summary_df[['model', 'brier_score', 'accuracy', 'average_rank']].head(5))
    
    print("\nMODEL COMPARISON COMPLETE")
    print(f"Report saved to: model_comparison_metrics.csv, model_comparison_summary.csv")
    print(f"Plots saved to: model_comparison.png, calibration_plots.png")

if __name__ == "__main__":
    main()
