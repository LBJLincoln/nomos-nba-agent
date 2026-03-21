import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.calibration import calibration_curve

def load_predictions(predictions_path: str = 'predictions.csv') -> pd.DataFrame:
    """
    Load prediction results from CSV file.

    Parameters:
    predictions_path (str): Path to predictions CSV file

    Returns:
    pd.DataFrame: Loaded prediction data
    """
    predictions_file = Path(predictions_path)

    if not predictions_file.exists():
        print(f"Error: Predictions file not found at {predictions_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(predictions_path)
        print(f"Loaded {len(df)} predictions from {predictions_path}")
        return df
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return pd.DataFrame()

def compute_brier_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Brier score by month.

    Parameters:
    df (pd.DataFrame): Prediction data with 'date' and 'probabilities' columns

    Returns:
    pd.DataFrame: Brier scores by month
    """
    if 'date' not in df.columns or 'probabilities' not in df.columns or 'actual' not in df.columns:
        print("Missing required columns for Brier score calculation")
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    results = []
    for month, group in df.groupby('month'):
        probs = group['probabilities'].values
        actual = (group['actual'] > 0.5).astype(int).values

        if len(probs) < 2:
            continue

        brier = np.mean((probs - actual) ** 2)
        results.append({
            'month': str(month),
            'brier_score': brier,
            'num_games': len(probs)
        })

    return pd.DataFrame(results)

def compute_roi_by_confidence_bucket(df: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """
    Compute ROI by confidence probability buckets.

    Parameters:
    df (pd.DataFrame): Prediction data
    n_buckets (int): Number of confidence buckets

    Returns:
    pd.DataFrame: ROI metrics by confidence bucket
    """
    df['predicted_label'] = (df['probabilities'] > 0.5).astype(int)
    df['confidence'] = df['probabilities'].apply(lambda x: max(x, 1-x))

    # Create buckets
    df['bucket'] = pd.cut(df['confidence'], bins=n_buckets, labels=range(n_buckets))

    results = []
    for bucket, group in df.groupby('bucket'):
        if len(group) < 2:
            continue

        # Calculate ROI
        group['profit'] = np.where(group['predicted_label'] == group['actual'],
                                   group['odds'] - 1, -1)
        roi = group['profit'].sum() / len(group)

        results.append({
            'bucket': int(bucket),
            'confidence_range': f"{(bucket/n_buckets):.2f}-{(bucket+1)/n_buckets:.2f}",
            'brier_score': np.mean((group['probabilities'] - group['actual']) ** 2),
            'accuracy': np.mean(group['predicted_label'] == group['actual']),
            'roi': roi,
            'num_games': len(group)
        })

    return pd.DataFrame(results)

def compute_calibration_curve_data(df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute calibration curve data points.

    Parameters:
    df (pd.DataFrame): Prediction data
    n_bins (int): Number of calibration bins

    Returns:
    pd.DataFrame: Calibration curve data points
    """
    if 'probabilities' not in df.columns or 'actual' not in df.columns:
        print("Missing required columns for calibration calculation")
        return pd.DataFrame()

    # Use sklearn's calibration_curve
    probs = df['probabilities'].values
    actual = (df['actual'] > 0.5).astype(int).values

    if len(probs) < 10:
        print("Not enough data for calibration curve")
        return pd.DataFrame()

    prob_true, prob_pred = calibration_curve(actual, probs, n_bins=n_bins)

    return pd.DataFrame({
        'bin': range(len(prob_true)),
        'predicted_prob': prob_pred,
        'true_prob': prob_true
    })

def generate_backtest_summary(df: pd.DataFrame) -> dict:
    """
    Generate comprehensive backtest summary.

    Parameters:
    df (pd.DataFrame): Prediction data

    Returns:
    dict: Summary statistics
    """
    summary = {}

    # Overall Brier score
    if 'probabilities' in df.columns and 'actual' in df.columns:
        probs = df['probabilities'].values
        actual = (df['actual'] > 0.5).astype(int).values
        summary['overall_brier'] = np.mean((probs - actual) ** 2)
        summary['num_predictions'] = len(probs)

    # Brier by month
    brier_by_month = compute_brier_by_month(df)
    summary['brier_by_month'] = brier_by_month

    # ROI by confidence bucket
    roi_by_bucket = compute_roi_by_confidence_bucket(df)
    summary['roi_by_bucket'] = roi_by_bucket

    # Calibration curve data
    calibration_data = compute_calibration_curve_data(df)
    summary['calibration_data'] = calibration_data

    return summary

def plot_backtest_results(summary: dict):
    """
    Generate backtest visualization plots.

    Parameters:
    summary (dict): Backtest summary data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Brier score by month
    if 'brier_by_month' in summary:
        brier_data = summary['brier_by_month']
        axes[0, 0].plot(brier_data['month'], brier_data['brier_score'], marker='o')
        axes[0, 0].set_title('Brier Score by Month')
        axes[0, 0].set_ylabel('Brier Score')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].grid(True, alpha=0.3)

    # ROI by confidence bucket
    if 'roi_by_bucket' in summary:
        roi_data = summary['roi_by_bucket']
        axes[0, 1].bar(roi_data['bucket'], roi_data['roi'])
        axes[0, 1].set_title('ROI by Confidence Bucket')
        axes[0, 1].set_ylabel('ROI')
        axes[0, 1].set_xlabel('Confidence Bucket')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].grid(True, alpha=0.3)

    # Calibration curve
    if 'calibration_data' in summary:
        cal_data = summary['calibration_data']
        axes[1, 0].plot(cal_data['predicted_prob'], cal_data['true_prob'], marker='o')
        axes[1, 0].plot([0, 1], [0, 1], linestyle='--', color='gray')
        axes[1, 0].set_title('Calibration Curve')
        axes[1, 0].set_xlabel('Predicted Probability')
        axes[1, 0].set_ylabel('True Probability')
        axes[1, 0].grid(True, alpha=0.3)

    # Overall metrics
    if 'overall_brier' in summary:
        axes[1, 1].axis('off')
        text = f"Overall Brier Score: {summary['overall_brier']:.4f}\n"
        text += f"Total Predictions: {summary['num_predictions']}\n"
        axes[1, 1].text(0.5, 0.5, text, ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig('backtest_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary_report(summary: dict):
    """
    Print text summary report.

    Parameters:
    summary (dict): Backtest summary data
    """
    print("\nBACKTEST SUMMARY REPORT")
    print("="*60)

    if 'overall_brier' in summary:
        print(f"Overall Brier Score: {summary['overall_brier']:.4f}")
        print(f"Total Predictions: {summary['num_predictions']}")
        print()

    if 'brier_by_month' in summary:
        print("Brier Score by Month:")
        print("-"*40)
        for idx, row in summary['brier_by_month'].iterrows():
            print(f"{row['month']}: {row['brier_score']:.4f} (n={row['num_games']})")
        print()

    if 'roi_by_bucket' in summary:
        print("ROI by Confidence Bucket:")
        print("-"*40)
        for idx, row in summary['roi_by_bucket'].iterrows():
            print(f"Bucket {row['bucket']}: ROI={row['roi']:.4f}, Accuracy={row['accuracy']:.2f}, n={row['num_games']}")
        print()

    if 'calibration_data' in summary:
        print("Calibration Curve Data:")
        print("-"*40)
        for idx, row in summary['calibration_data'].iterrows():
            print(f"Bin {row['bin']}: Pred={row['predicted_prob']:.2f}, True={row['true_prob']:.2f}")

def main():
    print("NBA Backtest Report Generator")
    print("="*40)

    # Load predictions
    df = load_predictions()

    if df.empty:
        print("No predictions to analyze. Exiting.")
        return

    # Generate summary
    print("Generating backtest summary...")
    summary = generate_backtest_summary(df)

    # Print report
    print_summary_report(summary)

    # Generate plots
    print("Generating visualizations...")
    plot_backtest_results(summary)

    # Save summary to CSV
    if 'brier_by_month' in summary:
        summary['brier_by_month'].to_csv('brier_by_month.csv', index=False)

    if 'roi_by_bucket' in summary:
        summary['roi_by_bucket'].to_csv('roi_by_bucket.csv', index=False)

    if 'calibration_data' in summary:
        summary['calibration_data'].to_csv('calibration_data.csv', index=False)

    print("\nBacktest report complete!")
    print(f"Summary saved to: backtest_summary.png")
    print(f"Detailed data saved to: brier_by_month.csv, roi_by_bucket.csv, calibration_data.csv")

if __name__ == "__main__":
    main()
