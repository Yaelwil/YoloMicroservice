import time
from pathlib import Path
from detect import run  # Ensure detect.py is in your working directory and contains the run function
import yaml
from loguru import logger
import os
import boto3
import requests
import json
from json_praising import PraisingJSON
from dynamodb_save import store_results_in_dynamodb

# Load environment variables
images_bucket = os.environ["BUCKET_NAME"]
queue_name = os.environ["YOLO_QUEUE_NAME"]
region = os.environ["REGION"]
dynamodb_table_name = os.environ["DYNAMODB_TABLE_NAME"]
alb_url = os.environ["ALB_URL"]

# Initialize SQS client, S3 client, and DynamoDB resource
sqs_client = boto3.client('sqs', region_name=region)
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb', region_name=region)
dynamodb_table = dynamodb.Table(dynamodb_table_name)
# Load class names from coco128.yaml
with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def consume():
    logger.debug('Queue is running')
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message_body = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'Raw message body: {message_body}')

            try:
                # Split the message body by commas and extract the relevant fields
                chat_id, predict, full_s3_path, img_name = message_body.split(', ')
            except ValueError as e:
                logger.error(f"Error parsing message body: {e}")
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                continue

            logger.info(f'Prediction: {prediction_id}. Start processing')

            if not img_name or not chat_id or not full_s3_path:
                logger.error('Invalid message format: chat_id, file_name, or s3_key missing')
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                continue

            logger.info(f'img_name received: {img_name}')
            s3.download_file(images_bucket, full_s3_path, img_name)
            original_img_path = img_name
            logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

            logger.info(f'chat_id: {chat_id}, predict: {predict}, full_s3_path: {full_s3_path}, img_name: {img_name}, original_img_path: {original_img_path}')

            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            # This is the path for the predicted image with labels
            predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')
            logger.info(f'predicted_img_path: {predicted_img_path}')

            # Combine the base name and the file extension
            base_name, file_extension = os.path.splitext(os.path.basename(original_img_path))
            new_file_name = f"{base_name}-predict{file_extension}"
            s3_predicted_directory_path = 'predicted_photos/'
            full_name_s3 = s3_predicted_directory_path + new_file_name

            # Upload the predicted image to S3
            s3.upload_file(str(predicted_img_path), images_bucket, full_name_s3)
            logger.info(f'prediction: {new_file_name}. was uploaded to s3 successfully')

            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
            labels = []
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': float(l[1]),
                        'cy': float(l[2]),
                        'width': float(l[3]),
                        'height': float(l[4]),
                    } for l in labels]

            logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

            prediction_summary = {
                'prediction_id': prediction_id,
                'original_img_path': original_img_path,
                'predicted_img_path': str(predicted_img_path),
                'labels': labels,
                'time': time.time()
            }

            # Define the path where you want to save the JSON file
            json_file_path = f'{base_name}.json'
            logger.info(f'Creating JSON file at path: {json_file_path}')

            # Write the prediction summary to a JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(prediction_summary, json_file)
            logger.info(f'JSON file created successfully: {json_file_path}')

            # Upload json file to S3
            json_folder_path = "json"
            json_full_path = f'{json_folder_path}/{json_file_path}'
            logger.info(f'Uploading JSON file to S3: {json_full_path}')
            try:
                s3.upload_file(json_file_path, images_bucket, json_full_path)
                logger.info(f'JSON file uploaded successfully to S3: {json_full_path}')
            except Exception as e:
                logger.error(f'Failed to upload JSON file to S3: {e}')

            logger.info(f'json_file_path: {json_file_path}')
            # Create a PraisingJSON object and process the image
            JSON_praising = PraisingJSON(json_file_path)
            formatted_json_file = JSON_praising.process_prediction_results(json_file_path)
            logger.info(f'formatted_json_file: {formatted_json_file}')

            # Store the results in DynamoDB
            if formatted_json_file:
                store_results_in_dynamodb(prediction_id, formatted_json_file, dynamodb_table_name, region, chat_id)
            else:
                logger.error('Formatted results are invalid or empty')
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)

            # Perform a POST request to the /results_predict endpoint
            try:
                if prediction_id:
                    response = requests.post(f"{alb_url}/results_predict?predictionId={prediction_id}")
                    if response.status_code == 200:
                        logger.info("Successfully sent results to the endpoint")
                        sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                        logger.info('Deleted the job from the queue')
                    else:
                        logger.error(f"Failed to send results. Status code: {response.status_code}")
                else:
                    logger.error("Empty or missing predictionId")
            except requests.exceptions.RequestException as e:
                logger.error(f"Error during POST request: {e}")


if __name__ == "__main__":
    consume()
