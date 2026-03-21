import numpy as np
import random
from typing import List, Tuple

def adaptive_mutation_rate(current_brier: float, target_brier: float, generation: int, max_generations: int) -> float:
    """
    Dynamically adjust mutation rate based on evolution progress.
    
    Parameters:
    current_brier (float): Current model's Brier score
    target_brier (float): Target Brier score
    generation (int): Current generation number
    max_generations (int): Maximum number of generations
    
    Returns:
    float: Adaptive mutation rate (0.0 to 1.0)
    """
    # Base mutation rate
    base_rate = 0.05
    
    # Progress-based adjustment
    progress = (target_brier - current_brier) / (target_brier - 0.15)  # 0.15 is theoretical minimum
    
    # Generation-based decay
    gen_factor = 1.0 - (generation / max_generations) * 0.5
    
    # Combine factors with safety clamps
    mutation_rate = max(0.01, min(0.3, base_rate * (1 + progress) * gen_factor))
    
    return mutation_rate

def generate_feature_mutations(feature_names: List[str], mutation_rate: float) -> List[Tuple[str, str, float]]:
    """
    Generate feature mutations with adaptive intensity.
    
    Parameters:
    feature_names (List[str]): List of available feature names
    mutation_rate (float): Current mutation rate
    
    Returns:
    List[Tuple[str, str, float]]: List of mutations (feature, operation, magnitude)
    """
    mutations = []
    num_mutations = int(len(feature_names) * mutation_rate * 2)  # Scale with feature count
    
    for _ in range(num_mutations):
        feature = random.choice(feature_names)
        operation = random.choice(['scale', 'shift', 'combine', 'transform'])
        
        if operation == 'scale':
            magnitude = np.random.uniform(0.8, 1.2)
            mutations.append((feature, 'scale', magnitude))
        elif operation == 'shift':
            magnitude = np.random.uniform(-5, 5)
            mutations.append((feature, 'shift', magnitude))
        elif operation == 'combine':
            magnitude = np.random.uniform(0.5, 2.0)
            other_feature = random.choice(feature_names)
            mutations.append((feature, f'combine_{other_feature}', magnitude))
        elif operation == 'transform':
            transform_type = random.choice(['log', 'sqrt', 'exp', 'square'])
            mutations.append((feature, f'transform_{transform_type}', 1.0))
    
    return mutations

def apply_feature_engineering_boosts(games: pd.DataFrame, team_id: int, features: dict) -> dict:
    """
    Apply advanced feature engineering boosts to existing features.
    
    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    features (dict): Existing feature dictionary
    
    Returns:
    dict: Enhanced feature dictionary
    """
    enhanced_features = features.copy()
    
    # Advanced momentum features
    if 'weighted_win_streak' in features:
        enhanced_features['momentum_power'] = features['weighted_win_streak'] ** 1.3
        enhanced_features['momentum_log'] = np.log1p(features['weighted_win_streak'])
    
    # Advanced rest features
    if 'rest_quality_score' in features:
        enhanced_features['rest_penalty'] = 1.0 if features['rest_quality_score'] < 0.8 else 0.0
        enhanced_features['travel_impact'] = features['travel_distance'] / 1000.0
    
    # Advanced pace features
    if 'pace_adjusted_points' in features:
        enhanced_features['pace_volatility'] = features['pace_adjusted_points'] * features.get('margin_volatility', 1.0)
    
    # Advanced opponent features
    if 'opponent_strength' in features:
        enhanced_features['strength_interaction'] = features['opponent_strength'] * features.get('weighted_win_streak', 1.0)
    
    return enhanced_features

def evolutionary_feature_selection(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                                   mutation_rate: float, generations: int = 10) -> List[str]:
    """
    Perform evolutionary feature selection to improve model performance.
    
    Parameters:
    X (np.ndarray): Feature matrix
    y (np.ndarray): Target labels
    feature_names (List[str]): List of feature names
    mutation_rate (float): Mutation rate for evolution
    generations (int): Number of evolutionary generations
    
    Returns:
    List[str]: Selected feature subset
    """
    n_features = len(feature_names)
    population_size = max(10, int(n_features * 0.3))
    
    # Initialize population (binary masks)
    population = np.random.randint(0, 2, (population_size, n_features))
    
    for gen in range(generations):
        # Evaluate fitness (using a simple proxy for Brier score)
        fitness_scores = []
        for individual in population:
            selected_features = feature_names[individual == 1]
            if len(selected_features) == 0:
                fitness_scores.append(1.0)  # Penalize empty selection
                continue
            
            # Use a simple model for evaluation
            X_selected = X[:, individual == 1]
            if X_selected.shape[1] == 0:
                fitness_scores.append(1.0)
                continue
            
            # Simple evaluation (lower is better)
            score = np.mean((X_selected.mean(axis=1) - y) ** 2)
            fitness_scores.append(score)
        
        # Normalize fitness
        fitness_scores = np.array(fitness_scores)
        fitness_scores = 1 / (1 + fitness_scores)  # Convert to maximization problem
        
        # Selection (tournament selection)
        new_population = []
        for _ in range(population_size):
            tournament = np.random.choice(population_size, 3, replace=False)
            winner = population[tournament[np.argmax(fitness_scores[tournament])]]
            new_population.append(winner.copy())
        
        # Crossover
        for i in range(0, population_size, 2):
            if i + 1 < population_size:
                if random.random() < 0.7:
                    crossover_point = np.random.randint(1, n_features - 1)
                    new_population[i][:crossover_point], new_population[i+1][:crossover_point] = \
                        new_population[i+1][:crossover_point], new_population[i][:crossover_point]
        
        # Mutation
        for i in range(population_size):
            if random.random() < mutation_rate:
                mutation_point = np.random.randint(0, n_features)
                new_population[i][mutation_point] = 1 - new_population[i][mutation_point]
        
        population = np.array(new_population)
    
    # Return best individual
    best_idx = np.argmax(fitness_scores)
    best_features = feature_names[population[best_idx] == 1]
    
    return best_features

def generate_evolutionary_boosts(games: pd.DataFrame, team_id: int, 
                                 current_brier: float, target_brier: float, 
                                 generation: int, max_generations: int) -> dict:
    """
    Generate comprehensive evolutionary boosts for feature engineering.
    
    Parameters:
    games (pd.DataFrame): Game data
    team_id (int): Team ID
    current_brier (float): Current model's Brier score
    target_brier (float): Target Brier score
    generation (int): Current generation number
    max_generations (int): Maximum number of generations
    
    Returns:
    dict: Dictionary of evolutionary boost parameters
    """
    boosts = {}
    
    # Adaptive mutation rate
    boosts['mutation_rate'] = adaptive_mutation_rate(current_brier, target_brier, generation, max_generations)
    
    # Feature mutations
    feature_names = ['weighted_win_streak', 'margin_trend', 'avg_margin', 
                     'margin_volatility', 'rest_quality_score', 'travel_distance',
                     'timezone_adjustment', 'opponent_strength', 'pace_adjusted_points']
    boosts['feature_mutations'] = generate_feature_mutations(feature_names, boosts['mutation_rate'])
    
    # Feature engineering
    base_features = compute_momentum_features(games, team_id)
    base_features.update(compute_rest_impact_features(games, team_id, games.iloc[0]))
    boosts['enhanced_features'] = apply_feature_engineering_boosts(games, team_id, base_features)
    
    # Evolutionary feature selection
    # (This would require actual model training data - placeholder for now)
    # X, y, feature_names = prepare_training_data(games)
    # boosts['selected_features'] = evolutionary_feature_selection(X, y, feature_names, 
    #                                                               boosts['mutation_rate'])
    
    return boosts

# Example usage:
# games = pd.DataFrame({...})  # Your game data
# team_id = 1
# current_brier = 0.22
# target_brier = 0.20
# generation = 220
# max_generations = 300

# boosts = generate_evolutionary_boosts(games, team_id, current_brier, target_brier, 
#                                       generation, max_generations)
# print(f"Adaptive mutation rate: {boosts['mutation_rate']:.4f}")
# print(f"Feature mutations: {boosts['feature_mutations']}")
# print(f"Enhanced features: {boosts['enhanced_features']}")
