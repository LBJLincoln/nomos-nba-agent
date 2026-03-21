import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

def load_game_data(data_path: str = 'data/games.csv') -> pd.DataFrame:
    """
    Load game data from CSV file.

    Parameters:
    data_path (str): Path to game data CSV file

    Returns:
    pd.DataFrame: Loaded game data
    """
    data_file = Path(data_path)

    if not data_file.exists():
        print(f"Error: Data file not found at {data_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} games from {data_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

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

def categorize_game_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create game type categories based on various factors.

    Parameters:
    df (pd.DataFrame): Game data

    Returns:
    pd.DataFrame: Original dataframe with game type categories
    """
    df = df.copy()

    # Basic game type categories
    df['is_playoff'] = df['game_type'].apply(lambda x: 1 if 'playoff' in str(x).lower() else 0)
    df['is_back_to_back'] = df['rest_days'].apply(lambda x: 1 if x < 1 else 0)
    df['is_road_trip'] = df['travel_distance'].apply(lambda x: 1 if x > 1000 else 0)
    df['is_rivalry'] = df['is_rivalry_game'].fillna(0).astype(int)

    # Advanced game type categories
    df['game_context'] = 'regular'
    df.loc[df['is_playoff'] == 1, 'game_context'] = 'playoff'
    df.loc[df['is_back_to_back'] == 1, 'game_context'] = 'back_to_back'
    df.loc[df['is_road_trip'] == 1, 'game_context'] = 'road_trip'
    df.loc[df['is_rivalry'] == 1, 'game_context'] = 'rivalry'

    # Combined categories
    df['game_complexity'] = 'simple'
    df.loc[(df['is_playoff'] == 1) | (df['is_road_trip'] == 1), 'game_complexity'] = 'complex'
    df.loc[(df['is_back_to_back'] == 1) & (df['is_road_trip'] == 1), 'game_complexity'] = 'very_complex'

    return df

def compute_accuracy_metrics(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """
    Compute accuracy metrics for different game types.

    Parameters:
    df (pd.DataFrame): Prediction data with actual results
    group_by (str): Column to group by (e.g., 'game_context')

    Returns:
    pd.DataFrame: Accuracy metrics by game type
    """
    results = []

    for group_name, group_data in df.groupby(group_by):
        if len(group_data) < 5:
            continue

        # Calculate metrics
        group_data['predicted_label'] = (group_data['probabilities'] > 0.5).astype(int)
        group_data['actual_label'] = (group_data['actual'] > 0.5).astype(int)

        accuracy = np.mean(group_data['predicted_label'] == group_data['actual_label'])
        brier_score = np.mean((group_data['probabilities'] - group_data['actual_label']) ** 2)
        log_loss = -np.mean(group_data['actual_label'] * np.log(group_data['probabilities'] + 1e-15) +
                           (1 - group_data['actual_label']) * np.log(1 - group_data['probabilities'] + 1e-15))

        results.append({
            'game_type': group_name,
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss,
            'num_games': len(group_data)
        })

    return pd.DataFrame(results)

def plot_accuracy_comparison(accuracy_df: pd.DataFrame):
    """
    Create comparison plots for accuracy by game type.

    Parameters:
    accuracy_df (pd.DataFrame): Accuracy metrics by game type
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Accuracy bar plot
    axes[0, 0].bar(accuracy_df['game_type'], accuracy_df['accuracy'])
    axes[0, 0].set_title('Prediction Accuracy by Game Type')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xlabel('Game Type')
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', label='Random')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Brier score bar plot
    axes[0, 1].bar(accuracy_df['game_type'], accuracy_df['brier_score'])
    axes[0, 1].set_title('Brier Score by Game Type')
    axes[0, 1].set_ylabel('Brier Score')
    axes[0, 1].set_xlabel('Game Type')
    axes[0, 1].axhline(y=0.25, color='red', linestyle='--', label='Random')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Log loss bar plot
    axes[1, 0].bar(accuracy_df['game_type'], accuracy_df['log_loss'])
    axes[1, 0].set_title('Log Loss by Game Type')
    axes[1, 0].set_ylabel('Log Loss')
    axes[1, 0].set_xlabel('Game Type')
    axes[1, 0].grid(True, alpha=0.3)

    # Sample size bar plot
    axes[1, 1].bar(accuracy_df['game_type'], accuracy_df['num_games'])
    axes[1, 1].set_title('Number of Games by Type')
    axes[1, 1].set_ylabel('Number of Games')
    axes[1, 1].set_xlabel('Game Type')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('accuracy_by_game_type.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_confusion_matrices(df: pd.DataFrame):
    """
    Generate confusion matrices for different game types.

    Parameters:
    df (pd.DataFrame): Prediction data
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    game_types = ['regular', 'playoff', 'back_to_back', 'road_trip']
    for idx, game_type in enumerate(game_types):
        if idx >= 4:
            break

        game_data = df[df['game_context'] == game_type]
        if len(game_data) < 5:
            axes[idx].axis('off')
            continue

        predicted = (game_data['probabilities'] > 0.5).astype(int)
        actual = (game_data['actual'] > 0.5).astype(int)

        # Create confusion matrix
        tp = np.sum((predicted == 1) & (actual == 1))
        tn = np.sum((predicted == 0) & (actual == 0))
        fp = np.sum((predicted == 1) & (actual == 0))
        fn = np.sum((predicted == 0) & (actual == 1))

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        # Plot confusion matrix
        cax = axes[idx].imshow(confusion_matrix, cmap='Blues', aspect='auto')
        axes[idx].set_title(f'{game_type.capitalize()} Confusion Matrix')
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Actual 0', 'Actual 1'])
        axes[idx].set_yticklabels(['Predicted 0', 'Predicted 1'])

        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, f'{confusion_matrix[i, j]}',
                             ha='center', va='center', color='black')

        axes[idx].grid(False)

    for idx in range(idx + 1, 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_game_type_performance():
    """
    Main analysis function for game type performance.
    """
    print("NBA Prediction Accuracy by Game Type Analysis")
    print("="*60)

    # Load data
    games = load_game_data()
    predictions = load_predictions()

    if games.empty or predictions.empty:
        print("Error: Could not load required data files")
        return

    # Merge data
    merged_data = pd.merge(predictions, games, on=['game_id', 'team_id'], how='left')

    # Categorize game types
    merged_data = categorize_game_types(merged_data)

    # Compute accuracy metrics
    print("Computing accuracy metrics...")
    accuracy_by_context = compute_accuracy_metrics(merged_data, 'game_context')
    accuracy_by_complexity = compute_accuracy_metrics(merged_data, 'game_complexity')

    # Print summary
    print("\nACCURACY BY GAME CONTEXT:")
    print("-"*40)
    print(accuracy_by_context.to_string(index=False))

    print("\nACCURACY BY GAME COMPLEXITY:")
    print("-"*40)
    print(accuracy_by_complexity.to_string(index=False))

    # Generate plots
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(accuracy_by_context)
    generate_confusion_matrices(merged_data)

    # Save results
    accuracy_by_context.to_csv('accuracy_by_game_context.csv', index=False)
    accuracy_by_complexity.to_csv('accuracy_by_game_complexity.csv', index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: accuracy_by_game_context.csv, accuracy_by_game_complexity.csv")
    print(f"Visualizations saved to: accuracy_by_game_type.png, confusion_matrices.png")

if __name__ == "__main__":
    analyze_game_type_performance()
