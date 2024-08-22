from datetime import datetime
import os
import boto3
import json


class PraisingJSON:

    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.s3 = boto3.client('s3')

    def process_prediction_results(self, json_file_path):
        """
        Process the response from YOLO to the format required in the project.

        Parameters:
            json_file_path (str): Path to the JSON file containing prediction results.

        Returns:
            processed_results (list): Processed prediction to the format required in the project.
        """
        # Load the prediction results from the JSON file
        with open(json_file_path, "r") as json_file:
            prediction_results = json.load(json_file)

        # Assuming prediction_results is a JSON object
        # Extract relevant information
        labels = prediction_results.get('labels', [])

        # Count occurrences of each object
        object_count = {}
        for label_info in labels:
            label = label_info['class']
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1

        # Format the results as a list of dictionaries
        processed_results = [{'class': label, 'count': count} for label, count in object_count.items()]

        if processed_results:
            # Convert the processed results to a list of strings
            formatted_results = []
            for result in processed_results:
                if 'class' in result:  # Ensure keys are present
                    formatted_result = f"Object: {result['class']} Count: {result['count']}"
                    formatted_results.append(formatted_result)
                else:
                    return  # Exit function if the format is invalid

            # Join the formatted results into a single string
            processed_results_message = "Prediction results:\n" + "\n".join(formatted_results)

            # Send the message to the Telegram end-user using the obtained chat ID
            return processed_results_message

        else:
            return
