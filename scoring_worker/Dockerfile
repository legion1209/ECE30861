# scoring-worker/Dockerfile
# Use the official AWS Lambda Python Base Image (optimized for Lambda)
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory to where Lambda expects the code
WORKDIR /var/task

# Copy your core application logic (src/) and the worker entry point and dependencies
# The Docker build MUST be run from the 'phase2' root directory for these paths to work.
COPY ./src /var/task/src
COPY ./scoring-worker/handler.py /var/task/
COPY ./scoring-worker/requirements.txt /var/task/

# Install Python dependencies. 
RUN pip install -r /var/task/requirements.txt --target /var/task

# Set the entry point for the Lambda function.
# This points to the handler.py file and the function name lambda_handler.
CMD [ "handler.lambda_handler" ]