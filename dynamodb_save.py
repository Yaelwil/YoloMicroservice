from datetime import datetime
import boto3
from loguru import logger


def store_results_in_dynamodb(prediction_id, formatted_json_file, dynamodb_table_name, region, chat_id):
    """
    Stores the formatted prediction results in DynamoDB.

    Parameters:
        prediction_id (str): Unique identifier for the prediction.
        formatted_json_file (str): Formatted results as a JSON string.
        dynamodb_table_name (str): Name of the DynamoDB table.
        region (str): AWS region where the DynamoDB table resides.
        chat_id (str): The chat ID associated with the prediction.
    """
    # Create a boto3 session with the specified region
    session = boto3.Session(region_name=region)
    dynamodb = session.resource('dynamodb')

    table = dynamodb.Table(dynamodb_table_name)

    try:
        # Log the values to be stored
        logger.debug(f'Prediction ID: {prediction_id}')
        logger.debug(f'Chat ID: {chat_id}')
        logger.debug(f'Results: {formatted_json_file}')

        item = {
            'prediction_id': prediction_id,
            'chat_id': chat_id,
            'results': formatted_json_file,
            'timestamp': datetime.utcnow().isoformat()
        }
        table.put_item(Item=item)
        logger.info(f'Successfully stored results in DynamoDB with prediction_id: {prediction_id}')
    except Exception as e:
        logger.error(f'Error storing results in DynamoDB: {e}')

