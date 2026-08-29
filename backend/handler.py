# backend-api/handler.py
import json
import os
import uuid
import boto3
import logging
import re
from decimal import Decimal

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
        
         # [B] Check rating (GET /artifact/model/{id}/rate)
        elif http_method == 'GET' and '/artifact/model/' in path and path.endswith('/rate'):
            response = handle_get_rating(event)

        # [C] Lineage (GET /artifact/{type}/{id}/lineage)
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

        # [D] License Check (POST /artifact/{type}/{id}/license-check)
        elif http_method == 'POST' and '/artifact/model/' in path and path.endswith('/license-check'):
            response = {
                'statusCode': 200, 
                'body': json.dumps(True)
            }

        # [E] Search by RegEx (POST /artifact/byRegEx)
        elif http_method == 'POST' and path.endswith('/byRegEx'):
            response = handle_search_by_regex(event) 

        # [F] Search by exact name (GET /artifact/byName/{name})
        elif http_method == 'GET' and '/artifact/byName/' in path:
            response = handle_search_by_name(event)

        # [G] Upload Artifact (Supports model, dataset, code)
        elif http_method == 'POST' and path.rstrip('/').split('/')[-1] in {'model', 'dataset', 'code'}:
            artifact_type = path.rstrip('/').split('/')[-1]
            response = handle_post_artifact(event, artifact_type)

        # [H] Download Artifact (GET /artifact/{type}/{id})
        elif http_method == 'GET' and '/artifact/' in path:
            response = handle_download_artifact(event)

        # [I] Delete Artifact (DELETE /artifact/{type}/{id})
        elif http_method == 'DELETE' and '/artifact/' in path:
            response = handle_delete_artifact(event)

        # [J] List/Search Artifacts (POST /artifacts)
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

def _artifact_parts(path):
    parts = path.strip('/').split('/')
    try:
        artifact_index = parts.index('artifact')
    except ValueError:
        return None, None

    if len(parts) <= artifact_index + 2:
        return None, None

    return parts[artifact_index + 1], parts[artifact_index + 2]

def _metadata_from_item(item):
    source_url = item.get('url', '')
    return {
        'id': item.get('id'),
        'name': item.get('name') or source_url.rstrip('/').split('/')[-1],
        'Version': '1.0.0',
        'Type': item.get('type', 'model'),
        'type': item.get('type', 'model'),
        'status': item.get('status', 'PENDING'),
    }

# --- POST /artifact/model (Job Submission) ---
def handle_post_artifact(event, artifact_type):
    try:
        body = json.loads(event['body'])
        source_url = body.get('url')
        if not source_url:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Missing source URL'})}

        artifact_id = str(uuid.uuid4())
        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"

        name = source_url.rstrip('/').split('/')[-1]

        # Create PENDING record in DynamoDB
        create_artifact(artifact_id, source_url, artifact_type, name)

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
                'metadata': {'id': artifact_id, 'name': name, 'type': artifact_type},
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
        path = event.get('path')
        artifact_type, artifact_id = _artifact_parts(path)
        if not artifact_type or not artifact_id:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        artifact_data = get_artifact_rating(artifact_id)
        if not artifact_data:
            return {'statusCode': 404, 'body': json.dumps({'error': 'Artifact not found'})}

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
                'metadata': {'type': artifact_type, 'id': artifact_id, 'name': name},
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

        queries = []
        if event.get('body'):
            parsed_body = json.loads(event['body'])
            queries = parsed_body if isinstance(parsed_body, list) else [parsed_body]

        response = table.scan()
        items = response.get('Items', [])

        results = []
        for item in items:
            metadata = _metadata_from_item(item)
            if queries and not any(_matches_artifact_query(metadata, query) for query in queries):
                continue
            results.append(metadata)

        return {'statusCode': 200, 'body': json.dumps(results)}
    
    except Exception as e:
            LOGGER.error(f"Search Error: {str(e)}", exc_info=True)
            return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

def _matches_artifact_query(metadata, query):
    name_filter = query.get('name', '*') if isinstance(query, dict) else '*'
    type_filter = query.get('types') if isinstance(query, dict) else None

    if type_filter and metadata.get('type') not in type_filter:
        return False

    if not name_filter or name_filter == '*':
        return True

    return metadata.get('name') == name_filter
    
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
            metadata = _metadata_from_item(item)
            name = metadata.get('name', '')
            if pattern.search(name):
                matches.append(metadata)
                
        return {'statusCode': 200, 'body': json.dumps(matches)}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

def handle_search_by_name(event):
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)
        name = path_unquote(event.get('path', '').rstrip('/').split('/')[-1])

        response = table.scan()
        matches = [
            metadata for metadata in (_metadata_from_item(item) for item in response.get('Items', []))
            if metadata.get('name') == name
        ]

        return {'statusCode': 200, 'body': json.dumps(matches)}
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

def path_unquote(value):
    from urllib.parse import unquote
    return unquote(value)
    
def handle_delete_artifact(event):
    try:
        # Extract ID from path parameters
        path = event.get('path')
        artifact_type, artifact_id = _artifact_parts(path)
        if not artifact_type or not artifact_id:
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