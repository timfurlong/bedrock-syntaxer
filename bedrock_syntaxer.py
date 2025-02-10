"""
A utility to force bedrock to apply the given syntax to it's response
"""

import boto3
from typing import Optional


def bedrock_syntaxer(
    model_id: str,
    messages: list,
    syntax: str,
    inference_config: dict = {},
    region_name: Optional[str] = None,
) -> str:
    """
    A utility to force bedrock to apply the given syntax to it's response

    Parameters:
        model_id (str): The model ID of the bedrock model
        messages (list): The messages to send to bedrock
        syntax (str): The syntax to apply to the response
        inference_config (dict): The inference configuration to use
        region_name (str): The region name of the bedrock model

    Returns:
        str: The response from bedrock with the syntax applied
    """

    # Make sure that the last message is a user message
    if messages[-1]["role"] != "user":
        raise ValueError("The last message must be a user message")

    # Make sure that stopSequence isn't in the inference config
    if "stopSequence" in inference_config:
        raise ValueError("stopSequence cannot be in the inference config")

    bedrock_runtime = boto3.client("bedrock-runtime", region_name=region_name)

    # Seed the LLM to get it return in the format we want
    seed = f"""```{syntax}"""
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"text": seed},
            ],
        }
    )

    # add the stop sequence to the inference config
    inference_config["stopSequences"] = ["\n```"]

    # Get the prediction from the LLM
    response = bedrock_runtime.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config,
    )
    response_text = seed + response["output"]["message"]["content"][0]["text"]

    # Get the text between the backticks
    response_text = response_text.split(f"```{syntax}\n")[1].split("\n```")[0]

    return response_text
