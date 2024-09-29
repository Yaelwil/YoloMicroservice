import boto3
import os

# region_name = os.environ["REGION"]
region_name = os.environ["REGION"]

def get_cert(prefix):
    # Create a Secrets Manager client
    client = boto3.client('secretsmanager', region_name=region_name)

    try:
        # Initialize an empty list to hold matching secrets
        matching_secrets = []

        # List all secrets with pagination
        paginator = client.get_paginator('list_secrets')
        for page in paginator.paginate():
            for secret in page['SecretList']:
                if secret['Name'].startswith(prefix):
                    matching_secrets.append(secret['Name'])

        # If any matching secrets are found, retrieve and return the first one
        if matching_secrets:
            secret_name = matching_secrets[0]
            secret_value = client.get_secret_value(SecretId=secret_name)
            return secret_value['SecretString']
        else:
            print(f"No secrets found with prefix {prefix}")
            return None
    except Exception as e:
        print(f"Error retrieving secrets: {str(e)}")
        return None
