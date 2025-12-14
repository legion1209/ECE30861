# scoring-worker/handler.py
import json
import os
import boto3
import logging

from src.acme_cli.runner import score_artifact_for_worker
from src.acme_cli.database_service import update_database

# Initialize AWS clients outside the handler for better performance
s3_client = boto3.client('s3')

# Environment variables will be passed from the SAM template
S3_BUCKET = os.environ.get('S3_ARTIFACT_BUCKET') 
LOGGER = logging.getLogger("acme_cli")
LOGGER.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Entry point for the SQS-triggered Worker Lambda.
    """
    for record in event['Records']:
        try:
            # Parse the SQS message body (sent by the API Lambda)
            body = json.loads(record['body'])
            artifact_id = body.get('artifact_id')
            url = body.get('url')

            LOGGER.info(f"Scoring {artifact_id} via runner logic...")

            scores = score_artifact_for_worker(url)

            if scores.net_score < 0.5:
                status = "FAILED"
            else:
                status = "COMPLETE"

            # Update db
            update_database(artifact_id, status, scores)
            LOGGER.info(f"Success! Score: {scores.net_score}")

        except Exception as e:
            LOGGER.error("Failed to process SQS record for %s: %s", artifact_id, str(e))
            # Re-raise the exception to allow SQS to handle retries
            raise e 
    
    return {'statusCode': 200}