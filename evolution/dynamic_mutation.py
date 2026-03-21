import numpy as np

class DynamicMutationRate:
    """
    Adaptive mutation rate that decreases over generations
    Early generations: high exploration (0.1)
    Late generations: fine-tuning (0.01)
    """
    def __init__(self, initial_rate=0.1, final_rate=0.01, total_generations=300):
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.total_generations = total_generations

    def get_rate(self, current_generation):
        """
        Calculate mutation rate using exponential decay
        rate = initial * (final/initial)^(generation/total)
        """
        if current_generation >= self.total_generations:
            return self.final_rate
        
        decay_factor = (self.final_rate / self.initial_rate) ** (current_generation / self.total_generations)
        return self.initial_rate * decay_factor

def apply_dynamic_mutation(individual, current_generation, total_generations):
    """
    Apply mutation with dynamic rate
    """
    dynamic_rate = DynamicMutationRate(total_generations=total_generations).get_rate(current_generation)
    
    # Only mutate if random value is below dynamic rate
    if np.random.random() < dynamic_rate:
        individual.mutate(rate=dynamic_rate)
    
    return individual
