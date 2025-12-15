# backend-api/handler.py
import json
import os
import uuid
import boto3
import urllib.request
import logging
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

def add_cors(resp: dict) -> dict:
    headers = resp.get('headers', {})
    headers.update(CORS_HEADERS)
    resp['headers'] = headers
    return resp

class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)

def lambda_handler(event, context):
    """Router for API Gateway events."""
    http_method = event.get('httpMethod')
    path = event.get('path') or ""

    LOGGER.info(f"Received {http_method} request for {path}")

    if http_method == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': CORS_HEADERS,
            'body': ''
        }

    try:
        # [A] Login authentication (PUT /authenticate)
        if http_method == 'PUT' and path.endswith('/authenticate'):
            return add_cors(handle_authenticate(event))

        # [B] Regex search (POST /artifact/byRegEx)
        if http_method == 'POST' and path.endswith('/byRegEx'):
            return add_cors(handle_search_by_regex(event))

        # [C] List/Search artifacts (POST /artifact?offset=1)
        # Template maps POST /artifact here; UI calls /artifact?offset=1
        if http_method == 'POST' and path.rstrip('/').endswith('/artifact'):
            return add_cors(handle_search_artifacts(event))

        # [D] Upload Artifact (POST /artifact/{type})
        if (
            http_method == 'POST'
            and '/artifact/' in path
            and not path.endswith('/byRegEx')
            and not path.rstrip('/').endswith('/artifact')
            and not path.endswith('/license-check')
        ):
            parts = path.strip('/').split('/')
            if 'artifact' in parts:
                idx = parts.index('artifact')
                if len(parts) > idx + 1:
                    artifact_type = parts[idx + 1]  # 'model', 'dataset', or 'code'
                    return add_cors(handle_post_artifact(event, artifact_type))

        # [E] Check rating (GET /artifact/model/{id}/rate)
        if http_method == 'GET' and '/artifact/model/' in path and path.endswith('/rate'):
            return add_cors(handle_get_rating(event))

        # [F] Lineage (GET /artifact/model/{id}/lineage)
        if http_method == 'GET' and '/artifact/model/' in path and path.endswith('/lineage'):
            parts = path.strip('/').split('/')
            current_id = parts[-2] if len(parts) >= 2 else "unknown"
            current_type = parts[-3] if len(parts) >= 3 else "model"

            return add_cors({
                'statusCode': 200,
                'body': json.dumps({
                    'nodes': [
                        {
                            'artifact_id': current_id,
                            'name': 'Current Artifact',
                            'source': current_type,
                            'metadata': {}
                        }
                    ],
                    'edges': []
                })
            })

        # [G] License Check (POST /artifact/model/{id}/license-check)
        if http_method == 'POST' and '/artifact/model/' in path and path.endswith('/license-check'):
            return add_cors({
                'statusCode': 200,
                'body': json.dumps(True)
            })

        # [H] Download Artifact (GET /artifact/{type}/{id})
        if http_method == 'GET' and '/artifact/' in path:
            return add_cors(handle_download_artifact(event))

        # [I] Delete Artifact (DELETE /artifact/{type}/{id})
        if http_method == 'DELETE' and '/artifact/' in path:
            return add_cors(handle_delete_artifact(event))

        # No route matched
        return add_cors({
            'statusCode': 404,
            'body': json.dumps({'error': 'Not found'})
        })

    except Exception as e:
        LOGGER.error("Unhandled Error: %s", str(e), exc_info=True)
        return add_cors({
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})
        })

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
                    'type': artifact_type,
                    'url': source_url
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
        artifact_id = event['pathParameters']['id']

        artifact_data = get_artifact_rating(artifact_id)

        if not artifact_data:
            return {'statusCode': 404, 'body': json.dumps({'error': 'Artifact not found'})}

        status = artifact_data.get('status', 'PENDING')

        if status == 'COMPLETE':
            # Return just the ModelRating object (what frontend expects)
            rating = artifact_data.get('ModelRating')
            return {'statusCode': 200, 'body': json.dumps(rating, cls=DecimalEncoder)}

        # If pending or failed, return status info
        return {
            'statusCode': 200,
            'body': json.dumps({'status': status})
        }

    except Exception as e:
        LOGGER.error("Polling Error: %s", str(e), exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal error: {str(e)}", 'status': 'FAILED'})
        }

def handle_download_artifact(event):
    try:
        path = event.get('path') or ""
        parts = path.strip('/').split('/')

        if 'artifact' not in parts:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        idx = parts.index('artifact')
        if len(parts) <= idx + 2:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        artifact_type = parts[idx + 1]
        artifact_id = parts[idx + 2]

        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"

        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET, 'Key': s3_key},
            ExpiresIn=3600  # URL expires in 1 hour
        )

        artifact_name = artifact_id

        return {
            'statusCode': 200,
            'body': json.dumps({
                'meta': {'name': artifact_name},
                'metadata': {'type': artifact_type, 'id': artifact_id},
                'data': {'url': url}
            })
        }
    except Exception as e:
        LOGGER.error("Download Error: %s", str(e), exc_info=True)
        return {'statusCode': 500, 'body': json.dumps({'error': str(e)})}

from boto3.dynamodb.conditions import Attr

def handle_search_artifacts(event):
    """
    Handle POST /artifact?offset=1 to list or search artifacts.

    Request body (from UI):
      [ { "name": "pattern-or-*", "types": ["model","dataset","code"]? } ]

    Response (for UI Search):
      [
        { "id": "...", "name": "...", "type": "model" },
        ...
      ]
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)

        body = []
        if event.get('body'):
            try:
                body = json.loads(event['body'])
            except Exception:
                body = []

        # The spec uses an array of queries; take the first one.
        if isinstance(body, list) and body:
            query = body[0]
        elif isinstance(body, dict):
            query = body
        else:
            query = {}

        name_filter = (query.get('name') or '*').strip()
        type_filter_list = query.get('types') or []  # list of 'model' | 'dataset' | 'code'

        # Scan table (OK for class-size datasets)
        resp = table.scan()
        items = resp.get('Items', [])

        results = []
        for item in items:
            art_id = item.get('id')
            if not art_id:
                continue

            # Determine type
            art_type = item.get('type', 'model')
            if type_filter_list and art_type not in type_filter_list:
                continue

            # Determine name: prefer stored name, else derive from URL, else 'unknown'
            name_val = item.get('name')
            if not name_val:
                url = item.get('url', '')
                if isinstance(url, str) and url:
                    name_val = url.rstrip('/').split('/')[-1] or 'unknown'
                else:
                    name_val = 'unknown'

            # Apply name filter ("*" means all; otherwise substring match)
            if name_filter != '*' and name_filter.lower() not in str(name_val).lower():
                continue

            results.append({
                'id': art_id,
                'name': name_val,
                'type': art_type,
            })

        return {
            'statusCode': 200,
            'body': json.dumps(results, cls=DecimalEncoder)
        }

    except Exception as e:
        LOGGER.error(f"Search Error: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
    
import re

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
            name = item.get('name', '')
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
        path = event.get('path') or ""
        parts = path.strip('/').split('/')

        if 'artifact' not in parts:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        idx = parts.index('artifact')
        if len(parts) <= idx + 2:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Invalid path'})}

        artifact_type = parts[idx + 1]
        artifact_id = parts[idx + 2]

        # Delete from S3
        s3_key = f"sources/{artifact_type}/{artifact_id}.zip"
        s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)

        # Delete from DynamoDB
        dynamodb = boto3.resource('dynamodb')
        table_name = os.environ.get('DYNAMODB_TABLE_NAME', 'ECE461Artifacts')
        table = dynamodb.Table(table_name)
        table.delete_item(Key={'id': artifact_id})

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Artifact deleted successfully'})
        }
    except Exception as e:
        LOGGER.error("Delete Error: %s", str(e), exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Internal error: {str(e)}"})
        }