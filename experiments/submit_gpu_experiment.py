import os
import uuid
import psycopg2
import json
from datetime import datetime

def submit_gpu_experiment():
    # Generate experiment ID
    experiment_id = f'exp_adam_{uuid.uuid4().hex[:8]}'
    
    # Choose model to test
    model_to_test = 'ft_transformer'  # Options: ft_transformer, node, saint, mc_dropout_rnn
    
    # Build params payload
    params = {
        'model': model_to_test,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 50,
        'dropout_rate': 0.3,
        'graph_layers': 3,
        'transformer_heads': 4
    }
    
    # Database connection
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    conn = None
    try:
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
        print(f"Experiment submitted: {experiment_id}")
        
    except Exception as e:
        print(f"Error submitting experiment: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    submit_gpu_experiment()
