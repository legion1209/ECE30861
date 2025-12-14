import os
import boto3
from typing import Any, Mapping
from decimal import Decimal
import time

# Initialize DynamoDB client outside the functions for Lambda performance
dynamodb = boto3.resource('dynamodb')

# Table name is pulled from environment variables set by the SAM template
TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')

dynamodb = boto3.resource('dynamodb')

table = dynamodb.Table(TABLE_NAME)
print(f"Database Service Connected to: {TABLE_NAME}")

# --- 1. Used by API Lambda (Job Submission) ---
def create_artifact(artifact_id: str, url: str) -> None:
    """
    Creates the initial artifact record with PENDING status.
    Called by the API Lambda (backend-api/handler.py).
    """
    current_timestamp = Decimal(str(time.time()))
    
    table.put_item(
        Item={
            'id': artifact_id, # Primary Key
            'url': url,
            'status': 'PENDING',
            'timestamp': current_timestamp 
        }
    )

# --- 2. Used by Worker Lambda (Job Completion) ---
def update_rating(artifact_id: str, rating_data: Mapping[str, Any], status: str = "COMPLETE") -> None:
    """
    Updates the artifact record with the final rating data.
    Called by the Worker Lambda (scoring-worker/handler.py via scoring.py).
    """
    # Note: '#s' is an alias for the reserved keyword 'status'
    table.update_item(
        Key={'id': artifact_id},
        UpdateExpression="SET ModelRating = :r, #s = :c",
        ExpressionAttributeNames={'#s': 'status'},
        ExpressionAttributeValues={
            ':r': rating_data,
            ':c': status
        }
    )

# --- 3. Used by API Lambda (Status Polling) ---
def get_artifact_rating(artifact_id: str) -> dict[str, Any] | None:
    """
    Retrieves the artifact status and rating data.
    Called by the API Lambda (backend-api/handler.py).
    """
    response = table.get_item(
        Key={'id': artifact_id}
    )
    return response.get('Item')

def update_database(artifact_id, status, scores=None):
    """
    Update the status and scores of an artifact in DynamoDB.

    :param artifact_id: The ID of the artifact.
    :param status: The status string (e.g., 'COMPLETE' or 'FAILED').
    :param scores: The score object (containing attributes like net_score, ramp_up_time, etc.).
    """
    
    update_expression = "SET #s = :status"
    expression_values = {':status': status}
    expression_names = {'#s': 'status'} 

    # If scores are provided, write them to the database as well
    # Note: DynamoDB does not support float types; they must be converted to Decimal.
    if scores:
        # Assume 'scores' is an object; convert its attributes to a dictionary.
        # If 'scores' is already a dict, .__dict__ is not needed.
        score_data = scores if isinstance(scores, dict) else scores.__dict__
        
        # Flatten the scores or store them as a Map. Here, we store them in the 'ModelRating' field.
        # You need to adjust this structure based on the OpenAPI Spec response format.
        rating_map = {}
        for k, v in score_data.items():
            if isinstance(v, float):
                # Convert float to Decimal to satisfy DynamoDB requirements
                rating_map[k] = Decimal(str(v))
            else:
                rating_map[k] = v
        
        update_expression += ", ModelRating = :r"
        expression_values[':r'] = rating_map

    try:
        table.update_item(
            Key={'id': artifact_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_names,
            ExpressionAttributeValues=expression_values
        )
        print(f"Successfully updated artifact {artifact_id} to status {status}")
    except Exception as e:
        print(f"Error updating database: {str(e)}")
        raise e

__all__ = ["create_artifact", "update_rating", "get_artifact_rating"]