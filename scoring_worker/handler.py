# scoring-worker/handler.py
import json
import os
import boto3
import logging

# Assuming your refactored logic is here:
# (phase2/src/acme_cli/scoring.py -> calculate_and_report_score)
from src.acme_cli.scoring import calculate_and_report_score
from src.acme_cli.runner import run_score
from src.acme_cli.database_service import update_database

# Initialize AWS clients outside the handler for better performance
s3_client = boto3.client('s3')

# Environment variables will be passed from the SAM template
S3_BUCKET = os.environ.get('S3_ARTIFACT_BUCKET') 
LOGGER = logging.getLogger("acme_cli")
LOGGER.setLevel(logging.INFO)

def calculate_and_report_score(local_path, artifact_id):
    # Calculate netscore
    scores = run_score(local_path) 
    
    # Run threshold check
    if scores.net_score < 0.5:
        status = "FAILED"
        LOGGER.info(f"Artifact {artifact_id} rejected due to low score: {scores.net_score}")
    else:
        status = "COMPLETE"
        
    # Update DB
    update_database(artifact_id, status, scores)

def lambda_handler(event, context):
    """
    Entry point for the SQS-triggered Worker Lambda.
    """
    for record in event['Records']:
        try:
            # 1. Parse the SQS message body (sent by the API Lambda)
            message_body = json.loads(record['body'])
            artifact_id = message_body['artifact_id']
            s3_key = message_body['s3_key']
            
            # 2. Define local path in the Lambda's temporary storage
            local_path = f"/tmp/{artifact_id}.zip" 

            LOGGER.info("Processing artifact_id: %s from s3://%s/%s", artifact_id, S3_BUCKET, s3_key)

            # 3. Download artifact from S3
            s3_client.download_file(S3_BUCKET, s3_key, local_path)

            # 4. Execute the core scoring logic and report the result
            # This function handles calculating the score AND updating the DynamoDB status.
            calculate_and_report_score(local_path, artifact_id)
            
            # 5. Clean up local file (good practice)
            os.remove(local_path)

        except Exception as e:
            LOGGER.error("Failed to process SQS record for %s: %s", artifact_id, str(e))
            # Re-raise the exception to allow SQS to handle retries
            raise e 
    
    return {'statusCode': 200}