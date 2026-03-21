import os
import uuid
import psycopg2
import json
from datetime import datetime

def submit_gpu_experiment():
    # Generate experiment ID
    experiment_id = f'exp_adam_{uuid.uuid4().hex[:8]}'
    
    # Define experiment parameters
    params = {
        "model": "ft_transformer",
        "architecture": {
            "layers": 4,
            "hidden_dim": 256,
            "dropout": 0.3,
            "learning_rate": 0.001
        },
        "training": {
            "batch_size": 32,
            "epochs": 50,
            "early_stopping": True
        }
    }
    
    # Connect to Supabase
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # Insert experiment
    insert_query = """
    INSERT INTO nba_experiments (
        experiment_id, agent_name, experiment_type, 
        params, status, target_space, priority, 
        baseline_brier, created_at
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    
    cursor.execute(insert_query, (
        experiment_id,
        'adam_strategist',
        'model_test',
        json.dumps(params),
        'pending',
        'gpu',
        8,
        0.2205,
        datetime.now()
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Experiment submitted: {experiment_id}")
    print(f"Params: {json.dumps(params, indent=2)}")

if __name__ == "__main__":
    submit_gpu_experiment()
