# scoring-worker/handler.py
import json
import os
import boto3
import logging

from src.acme_cli.runner import score_artifact_for_worker
from src.acme_cli.database_service import update_database, get_artifact_rating

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
    os.chdir('/tmp')

    for record in event['Records']:
        artifact_id = "unknown"
        try:
            body = json.loads(record['body'])
            artifact_id = body.get('artifact_id')
            url = body.get('url')

            if not url and artifact_id:
                artifact_data = get_artifact_rating(artifact_id) 
                if artifact_data:
                    url = artifact_data.get('url')

            if not url:
                LOGGER.error("FATAL: URL not found for artifact: %s", artifact_id)
                update_database(artifact_id, "FAILED", None) 
                continue 

            LOGGER.info(f"Scoring {artifact_id} via runner logic...")

            scores_dict = score_artifact_for_worker(url)
            
            net_score = scores_dict.get('net_score', 0.0)

            if net_score < 0.5:
                status = "FAILED"
            else:
                status = "COMPLETE"

            # 3. Update db 
            update_database(artifact_id, status, scores_dict)
            LOGGER.info(f"Success! Score: {net_score}")

        except Exception as e:
            LOGGER.error("Failed to process SQS record for %s: %s", artifact_id, str(e))
            raise e 
    
    return {'statusCode': 200}