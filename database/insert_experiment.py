import os
import uuid
import psycopg2
import json

def insert_experiment():
    # Get the DATABASE_URL from the environment
    DATABASE_URL = os.environ.get('DATABASE_URL')

    # Parse the DATABASE_URL to get the connection parameters
    conn = psycopg2.connect(DATABASE_URL)

    # Create a cursor object
    cur = conn.cursor()

    # Generate a random experiment_id
    experiment_id = f'exp_adam_{uuid.uuid4().hex}'

    # Choose a random model from the list
    models = ['ft_transformer', 'node', 'saint', 'mc_dropout_rnn']
    params = {'model': models[0]}  # Choose the first model for now

    # Insert the experiment into the nba_experiments table
    cur.execute("""
        INSERT INTO nba_experiments (experiment_id, agent_name, experiment_type, params, status, target_space, priority, baseline_brier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        experiment_id,
        'adam_strategist',
        'model_test',
        json.dumps(params),
        'pending',
        'gpu',
        8,
        0.2205
    ))

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cur.close()
    conn.close()

    print(f'Experiment {experiment_id} inserted into the database')

# Call the function to insert the experiment
insert_experiment()
