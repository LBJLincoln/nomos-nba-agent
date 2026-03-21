import os
import uuid
import psycopg2
import json
from datetime import datetime

# Experiment parameters to test
EXPERIMENT_PARAMS = {
    "ft_transformer": {
        "model": "FT_Transformer",
        "params": {
            "n_head": 8,
            "n_layer": 6,
            "dropout": 0.1,
            "ffn_dim": 2048
        }
    },
    "node": {
        "model": "NODE",
        "params": {
            "n_layer": 10,
            "dropout": 0.2,
            "n_head": 8
        }
    },
    "saint": {
        "model": "SAINT",
        "params": {
            "n_layer": 8,
            "dropout": 0.15,
            "n_head": 6
        }
    },
    "mc_dropout_rnn": {
        "model": "MC_Dropout_RNN",
        "params": {
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True
        }
    }
}

def submit_experiment(experiment_type: str):
    """Submit GPU experiment to Supabase"""
    # Generate unique experiment ID
    experiment_id = f"exp_adam_{uuid.uuid4().hex[:8]}"
    
    # Get experiment parameters
    params = EXPERIMENT_PARAMS.get(experiment_type)
    if not params:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Database connection
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL not set in environment")
    
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # Insert experiment record
    query = """
    INSERT INTO nba_experiments (
        experiment_id, agent_name, experiment_type, params, 
        status, target_space, priority, baseline_brier, 
        created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    data = (
        experiment_id,
        "adam_strategist",
        experiment_type,
        json.dumps(params),
        "pending",
        "gpu",
        8,
        0.2205,
        datetime.now()
    )
    
    cursor.execute(query, data)
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Submitted experiment: {experiment_id}")
    print(f"Type: {experiment_type}")
    print(f"Model: {params['model']}")
    print(f"Params: {params['params']}")

if __name__ == "__main__":
    # Submit all experiments
    for exp_type in EXPERIMENT_PARAMS.keys():
        submit_experiment(exp_type)
