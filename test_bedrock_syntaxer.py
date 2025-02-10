import csv
import json

from bedrock_syntaxer import bedrock_syntaxer

ACCOUNT_ID = "961341532715"
REGION = "us-east-1"
MODEL_NAME = "claude-3-5-sonnet-20241022-v2:0"
MODEL_ID = (
    f"arn:aws:bedrock:{REGION}:{ACCOUNT_ID}:inference-profile/us.anthropic.{MODEL_NAME}"
)


def test_bedrock_syntaxer_json():
    prompt = (
        "Write the definition of 3 Filipino word in JSON format. "
        "The words are 'kamusta', 'salamat', and 'paalam'."
        "Use the keys 'word' and 'definition'."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    response = bedrock_syntaxer(
        model_id=MODEL_ID,
        messages=messages,
        syntax="json",
        region_name=REGION,
    )

    # Check that the response is in JSON format
    response_dict = json.loads(response)

    print(response_dict)


def test_bedrock_syntaxer_csv():
    prompt = (
        "Write the definition of 3 Filipino word in CSV format. "
        "The words are 'kamusta', 'salamat', and 'paalam'."
        "Use the keys 'word' and 'definition'."
    )
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    response = bedrock_syntaxer(
        model_id=MODEL_ID,
        messages=messages,
        syntax="csv",
        region_name=REGION,
    )

    # Check that the response is in CSV format
    response_csv = csv.reader(response.splitlines())

    for row in response_csv:
        print(row)
