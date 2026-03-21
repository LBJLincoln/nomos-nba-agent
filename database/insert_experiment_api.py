import os
import uuid
import requests
import json

def insert_experiment():
    # Get the DATABASE_URL from the environment
    DATABASE_URL = os.environ.get('DATABASE_URL')

    # Parse the DATABASE_URL to get the connection parameters
    url = f'{DATABASE_URL}/nba_experiments'

    # Generate a random experiment_id
    experiment_id = f'exp_adam_{uuid.uuid4().hex}'

    # Choose a random model from the list
    models = ['ft_transformer', 'node', 'saint', 'mc_dropout_rnn']
    params = {'model': models[0]}  # Choose the first model for now

    # Insert the experiment into the nba_experiments table
    data = {
        'experiment_id': experiment_id,
        'agent_name': 'adam_strategist',
        'experiment_type': 'model_test',
        'params': json.dumps(params),
        'status': 'pending',
        'target_space': 'gpu',
        'priority': 8,
        'baseline_brier': 0.2205
    }

    # Set the API key in the headers
    headers = {
        'Authorization': f'Bearer {os.environ.get("SUPABASE_KEY")}',
        'Content-Type': 'application/json'
    }

    # Send the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 201:
        print(f'Experiment {experiment_id} inserted into the database')
    else:
        print(f'Failed to insert experiment: {response.text}')

# Call the function to insert the experiment
insert_experiment()
