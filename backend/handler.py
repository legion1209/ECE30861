# backend-api/handler.py
import json
import os
import uuid
import boto3
import urllib.request
import logging

from src.acme_cli.database_service import create_artifact, get_artifact_rating 

s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs')

S3_BUCKET = os.environ.get('S3_ARTIFACT_BUCKET')
SQS_URL = os.environ.get('SQS_QUEUE_URL')

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

def lambda_handler(event, context):
    """Router for API Gateway events."""
    http_method = event.get('httpMethod')
    path = event.get('path')
    
    if http_method == 'POST' and path == '/artifact/model':
        return handle_post_artifact(event)
    elif http_method == 'GET' and '/artifact/model/' in path and '/rate' in path:
        return handle_get_rating(event)

    return {'statusCode': 404, 'body': json.dumps({'error': 'Not found'})}

# --- POST /artifact/model (Job Submission) ---
def handle_post_artifact(event):
    try:
        body = json.loads(event['body'])
        source_url = body.get('url')
        if not source_url:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing source URL'})}

        artifact_id = str(uuid.uuid4())
        s3_key = f"sources/{artifact_id}.zip"

        # 1. Download and Upload to S3
        with urllib.request.urlopen(source_url) as response:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=response.read(),
                ContentType=response.info().get_content_type()
            )

        # 2. Create PENDING record in DynamoDB
        create_artifact(artifact_id, source_url)

        # 3. Send message to SQS (The ASYNCHRONOUS link)
        sqs_client.send_message(
            QueueUrl=SQS_URL,
            MessageBody=json.dumps({
                'artifact_id': artifact_id,
                's3_bucket': S3_BUCKET,
                's3_key': s3_key
            })
        )
        
        # 4. Return 202 Accepted response immediately
        return {
            'statusCode': 202,
            'body': json.dumps({'id': artifact_id, 'status': 'Evaluation accepted and deferred.'})
        }
    except Exception as e:
        LOGGER.error("API Error: %s", str(e), exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})}

# --- GET /artifact/model/{id}/rate (Polling) ---
def handle_get_rating(event):
    try:
        # Extract ID from path parameters
        artifact_id = event['pathParameters']['id']
        
        # Look up status and rating in DynamoDB
        artifact_data = get_artifact_rating(artifact_id) 
        
        if not artifact_data:
            return {'statusCode': 404, 'body': json.dumps({'error': 'Artifact not found'})}
            
        status = artifact_data.get('status', 'PENDING')
        
        if status == 'COMPLETE':
            # Return the full ModelRating object
            return {'statusCode': 200, 'body': json.dumps(artifact_data.get('ModelRating'))}
        
        # If pending or failed, return status only
        return {'statusCode': 204, 'headers': {'X-Status': status}, 'body': ''}
        
    except Exception as e:
        LOGGER.error("Polling Error: %s", str(e), exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})}