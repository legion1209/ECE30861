# backend-api/handler.py
import json
import os
import uuid
import boto3
import logging
import re
from decimal import Decimal
from boto3.dynamodb.conditions import Attr

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

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

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
        elif http_method == 'POST' and '/artifact/' in path:
            # Simple path parsing to get type. Example: path="/Prod/artifact/model/id" -> parts=['', 'artifact', 'model']
            artifact_type = path.strip('/').split('/')[-1]
            response = handle_post_artifact(event, artifact_type)
            # Gets 'model', 'dataset', or 'code'

        # [C] Check rating (GET /artifact/model/{id}/rate)
        elif http_method == 'GET' and '/artifact/model/' in path and path.endswith('/rate'):
            response = handle_get_rating(event)

        # [D] Lineage (GET /artifact/{type}/{id}/lineage)
        elif http_method == 'GET' and '/artifact/model/' in path and path.endswith('/lineage'):
            parts = path.strip('/').split('/')
            current_id = parts[-2] 
            
            response = {
                'statusCode': 200,
                'body': json.dumps({
                    'nodes': [{'artifact_id': current_id, 'name': 'Current Artifact', 'source': 'model', 'metadata': {}}],
                    'edges': []
                })
            }

        # [E] License Check (GET /artifact/{type}/{id}/license)
        elif http_method == 'POST' and '/artifact/model/' in path and path.endswith('/license-check'):
            response = {
                'statusCode': 200, 
                'body': json.dumps(True)
            }

        # [F] Download Artifact (GET /artifact/{type}/{id}) - (Added based on your requirements)
        elif http_method == 'GET' and '/artifact/' in path:
            response = handle_download_artifact(event)

        # [G] Delete Artifact (DELETE /artifact/{type}/{id})
        elif http_method == 'DELETE' and '/artifact/' in path:
            response = handle_delete_artifact(event)

        # [H] Search by RegEx (POST /artifact/byRegEx)
        elif http_method == 'POST' and path.endswith('/byRegEx'):
            response = handle_search_by_regex(event) 

        # [I] List/Search Artifacts (POST /artifacts)
        elif http_method == 'POST' and path.endswith('/artifacts'):
            response = handle_search_artifacts(event)

        # If no route matches
        else:
            response = {'statusCode': 404, 'body': json.dumps({'error': f'No route for {http_method} {path}'})}

    except Exception as e:
        LOGGER.error("Unhandled Error: %s", str(e), exc_info=True)
        response = {
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})
        }
    if 'headers' not in response:
        response['headers'] = {}
    response['headers'].update(CORS_HEADERS)
    
    return response

def handle_authenticate(event):
    # Implement authentication logic here
    # Return a dummy token for demonstration purposes
    try:
        body = json.loads(event['body'])
        if (body.get('user', {}).get('name') == "ece30861defaultadminuser" and 
            body.get('secret', {}).get('password') == "correcthorsebatterystaple123(!__+@**(A'" + '"`;DROP TABLE artifacts;'):
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

        # Create PENDING record in DynamoDB
        create_artifact(artifact_id, source_url)

        # Send to SQS only if configured (skipped in local mode)
        if SQS_URL and artifact_type == 'model':
            s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
            sqs_client.send_message(
                QueueUrl=SQS_URL,
                MessageBody=json.dumps({
                    'artifact_id': artifact_id,
                    's3_key': s3_key,
                    'type': artifact_type,
                    'url': source_url
                })
            )
        elif artifact_type == 'model':
            # LOCAL MODE: score inline (no SQS/S3)
            LOGGER.info("LOCAL MODE: scoring inline for %s", artifact_id)
            _score_inline(artifact_id, source_url)
        
        # Return 201 Accepted response immediately
        return {
            'statusCode': 201,
            'body': json.dumps({
                'metadata': {'id': artifact_id, 'name': source_url.split('/')[-1], 'type': artifact_type},
                'data': {'url': source_url}
            })
        }
    except Exception as e:
        LOGGER.error("API Error: %s", str(e), exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})}

def _score_inline(artifact_id: str, url: str):
    """Run scoring synchronously in local mode (no SQS worker needed)."""
    import threading
    from src.acme_cli.runner import score_artifact_for_worker
    from src.acme_cli.database_service import update_database

    def _run():
        try:
            scores = score_artifact_for_worker(url)
            net_score = scores.get('net_score', 0.0)
            status = "COMPLETE" if net_score >= 0.5 else "FAILED"
            update_database(artifact_id, status, scores)
            LOGGER.info("Inline score done for %s: net_score=%.2f status=%s", artifact_id, net_score, status)
        except Exception as exc:
            LOGGER.error("Inline scoring failed for %s: %s", artifact_id, exc)
            update_database(artifact_id, "FAILED", None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

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
            return {'statusCode': 200, 'body': json.dumps(artifact_data.get('ModelRating'), cls=DecimalEncoder)}
        
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
        artifact_data = get_artifact_rating(artifact_id)

        if len(parts) >= 3:
            artifact_type = parts[1]
            artifact_id = parts[2]
        else:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}
        source_url = artifact_data.get('url', '')
        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
        
        if S3_BUCKET:
            s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
            download_url = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': S3_BUCKET, 'Key': s3_key},
                ExpiresIn=3600
            )
        else:
            download_url = source_url

        name = source_url.rstrip('/').split('/')[-1]
        return {
            'statusCode': 200,
            'body': json.dumps({
                'meta': {'name': name},
                'metadata': {'type': artifact_type, 'id': artifact_id},
                'data': {'url': download_url}
            })
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

def handle_search_artifacts(event):
    """
    Handle POST /artifacts to list or search packages.
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)

        response = table.scan()
        items = response.get('Items', [])

        results = []
        for item in items:
            results.append({
                'id': item.get('id'),
                'name': item.get('name', item.get('url', 'unknown').rstrip('/').split('/')[-1]),
                'Version': '1.0.0',
                'Type': item.get('type', 'model'),
                'type': item.get('type', 'model'),
                'status': item.get('status', 'PENDING'),
            })

        return {'statusCode': 200, 'body': json.dumps(results)}

    except Exception as e:
        LOGGER.error(f"Search Error: {str(e)}", exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    
def handle_search_by_regex(event):
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)
        
        body = json.loads(event['body'])
        regex = body.get('regex', '')
        
        response = table.scan()
        items = response.get('Items', [])
        
        matches = []
        pattern = re.compile(regex)
        
        for item in items:
            name = item.get('name', item.get('url', '').rstrip('/').split('/')[-1])
            if pattern.search(name):
                matches.append({
                    'id': item.get('id'),
                    'name': name,
                    'Version': '1.0.0',
                    'Type': item.get('type'),
                    'type': item.get('type')
                })
                
        return {'statusCode': 200, 'body': json.dumps(matches)}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}
    
def handle_delete_artifact(event):
    try:
        # Extract ID from path parameters
        path = event.get('path')
        parts = path.strip('/').split('/')

        if len(parts) >= 3:
            artifact_type = parts[1]
            artifact_id = parts[2]
        else:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        # Delete from S3
        if S3_BUCKET:
            s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
            s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)

        # Delete from DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)
        table.delete_item(Key={'id': artifact_id})

        return {'statusCode': 200, 'body': json.dumps({'message': 'Artifact deleted successfully'})}
    except Exception as e:
        LOGGER.error("Delete Error: %s", str(e), exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': f"Internal error: {str(e)}"})}