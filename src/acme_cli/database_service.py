import os
import boto3
from typing import Any, Mapping

# Initialize DynamoDB client outside the functions for Lambda performance
dynamodb = boto3.resource('dynamodb')

# Table name is pulled from environment variables set by the SAM template
ARTIFACTS_TABLE_NAME = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')

# --- Helper to get the table instance ---
def get_table():
    """Returns the DynamoDB table instance."""
    return dynamodb.Table(ARTIFACTS_TABLE_NAME)

# --- 1. Used by API Lambda (Job Submission) ---
def create_artifact(artifact_id: str, url: str) -> None:
    """
    Creates the initial artifact record with PENDING status.
    Called by the API Lambda (backend-api/handler.py).
    """
    table = get_table()
    table.put_item(
        Item={
            'id': artifact_id, # Primary Key
            'url': url,
            'status': 'PENDING',
            'timestamp': boto3.dynamodb.types.Decimal(os.time()) # Record submission time
        }
    )

# --- 2. Used by Worker Lambda (Job Completion) ---
def update_rating(artifact_id: str, rating_data: Mapping[str, Any], status: str = "COMPLETE") -> None:
    """
    Updates the artifact record with the final rating data.
    Called by the Worker Lambda (scoring-worker/handler.py via scoring.py).
    """
    table = get_table()
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
    table = get_table()
    response = table.get_item(
        Key={'id': artifact_id}
    )
    return response.get('Item')

__all__ = ["create_artifact", "update_rating", "get_artifact_rating"]