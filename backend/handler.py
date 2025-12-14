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

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Authorization"
}

def lambda_handler(event, context):
    """Router for API Gateway events."""
    # 1. First step: Extract HTTP method and path (Must be done first)
    http_method = event.get('httpMethod')
    path = event.get('path')
    
    LOGGER.info(f"Received {http_method} request for {path}")

    if http_method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': CORS_HEADERS,
            'body': ''
        }
    response = None

    # 2. Routing logic
    try:
        # [A] Login authentication (PUT /authenticate)
        if http_method == 'PUT' and path == '/authenticate':
            response = handle_authenticate(event)
        
        # [B] Upload Artifact (Supports model, dataset, code)
        # Logic: It is a POST method, and the path starts with /artifact/, but excludes other sub-paths (like /byRegEx)
        # Example: /artifact/model, /artifact/dataset
        if http_method == 'POST' and '/artifact/' in path:
            # Simple path parsing to get type. Example: path="/Prod/artifact/model/id" -> parts=['', 'artifact', 'model']
            parts = path.strip('/').split('/')
            if len(parts) == 2:
                artifact_type = parts[1] # Gets 'model', 'dataset', or 'code'
                response = handle_post_artifact(event, artifact_type)

        # [C] Check rating (GET /artifact/model/{id}/rate)
        if http_method == 'GET' and '/artifact/model/' in path and path.endswith('/rate'):
            response = handle_get_rating(event)

        # [D] Download Artifact (GET /artifact/{type}/{id}) - (Added based on your requirements)
        if http_method == 'GET' and '/artifact/' in path:
            response = handle_download_artifact(event)

        # If no route matches
        if 'headers' not in response:
            response['headers'] = {}

    except Exception as e:
        LOGGER.error("Unhandled Error: %s", str(e), exc_info=True)
        response = {
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})
        }
    
    response['headers'].update(CORS_HEADERS)
    
    return response

def handle_authenticate(event):
    # Implement authentication logic here
    # Return a dummy token for demonstration purposes
    try:
        body = json.loads(event['body'])
        if (body.get('user', {}).get('name') == "ece30861defaultadminuser" and 
            body.get('secret', {}).get('password') == "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE artifacts;"):
            return {
                'statusCode': 200,
                'body': json.dumps("some-valid-token-string") # Return token
            }
        else:
            return {'statusCode': 401, 'body': json.dumps({'error': 'Invalid credentials'})}
    except Exception as e:
        return {'statusCode': 400, 'body': json.dumps({'error': str(e)})}

# --- POST /artifact/model (Job Submission) ---
def handle_post_artifact(event, artifact_type):
    try:
        body = json.loads(event['body'])
        source_url = body.get('url')
        if not source_url:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing source URL'})}

        artifact_id = str(uuid.uuid4())
        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"

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
        if artifact_type == 'model':
            sqs_client.send_message(
                QueueUrl=SQS_URL,
                MessageBody=json.dumps({
                    'artifact_id': artifact_id,
                    's3_bucket': S3_BUCKET,
                    's3_key': s3_key,
                    'type': artifact_type
                })
            )
        
        # 4. Return 201 Accepted response immediately
        return {
            'statusCode': 201,
            'body': json.dumps({
                'metadata': {'id': artifact_id, 'name': 'unknown', 'type': artifact_type},
                'data': {'url': source_url}
            })
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
    
def handle_download_artifact(event):
    # Check if artifact exists and is complete
    # item = db.get_item(artifact_id)
    # if not item or item.status == 'FAILED': return 404...
    
    # generate S3 Presigned URL for download
    try:
        # path: Prod/artifacts/{type}/{id}
        path = event.get('path')
        parts = path.strip('/').split('/')

        if len(parts) >= 3:
            artifact_type = parts[1]
            artifact_id = parts[2]
        else:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}
        
        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
        
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600 # URL expires in 1 hour
        )
        artifact_name = url.strip('/').split('/')[-2]
        return {
            'statusCode': 200,
            'body': json.dumps({'metadata': {'name': artifact_name, 'type': artifact_type, 'id': artifact_id}, 'data': {'url': url}}) # url for download
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

def handle_search_artifacts(event):
    # Return empty list for now
    return {
        'statusCode': 200,
        'body': json.dumps([])
    }