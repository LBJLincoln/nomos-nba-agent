import os
import uuid
import psycopg2
import json
from datetime import datetime

# Experiment parameters to test
EXPERIMENT_PARAMS = {
    'ft_transformer': {
        'model': 'FT_Transformer',
        'params': {'num_layers': 4, 'd_model': 256, 'n_head': 8}
    },
    'node': {
        'model': 'NODE',
        'params': {'num_layers': 6, 'hidden_dim': 128}
    },
    'saint': {
        'model': 'SAINT',
        'params': {'num_layers': 3, 'd_model': 512}
    },
    'mc_dropout_rnn': {
        'model': 'MC_Dropout_RNN',
        'params': {'hidden_size': 256, 'dropout_rate': 0.3, 'n_layers': 2}
    }
}

def submit_experiment(experiment_type):
    """Submit GPU experiment to Supabase"""
    experiment_id = f'exp_adam_{uuid.uuid4().hex[:8]}'
    params = EXPERIMENT_PARAMS.get(experiment_type)
    
    if not params:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    conn = None
    try:
        # Connect to Supabase
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        cursor = conn.cursor()
        
        # Insert experiment
        cursor.execute("""
            INSERT INTO nba_experiments (
                experiment_id, agent_name, experiment_type, 
                params, status, target_space, priority, 
                baseline_brier, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            experiment_id,
            'adam_strategist',
            experiment_type,
            json.dumps(params),
            'pending',
            'gpu',
            8,
            0.2205,
            datetime.now()
        ))
        
        conn.commit()
        cursor.close()
        print(f"Submitted experiment: {experiment_id} ({experiment_type})")
        
    except Exception as e:
        print(f"Error submitting experiment: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    # Submit one experiment - you can modify this to test different models
    submit_experiment('ft_transformer')
