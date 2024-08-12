# https://github.com/anthropics/anthropic-cookbook/blob/main/misc/building_moderation_filter.ipynb

from anthropic import Anthropic
import time, pandas as pd
from datetime import date

client = Anthropic()
MODEL_NAME = "claude-3-haiku-20240307"
client.api_key = 'ENTER API KEY'

# def moderate_text(user_text):
guidelines = '''BLOCK CATEGORY:
                - Promoting violence, illegal activities, or hate speech
                - Explicit sexual content
                - Harmful misinformation or conspiracy theories

                ALLOW CATEGORY:
                - Most other content is allowed, as long as it is not explicitly disallowed
            '''
prompt_template = """
You are a content moderation expert tasked with categorizing user-generated text based on the following guidelines:

{guidelines}

Here is the user-generated text to categorize:
<user_text>{user_text}</user_text>

Based on the guidelines above, classify this text as either ALLOW or BLOCK. Return nothing else.
"""

def run_anthropic_ME(text_list):
    responses = []
    parsed_response = []
    try:
        for text in text_list:
            assert type(text) == str
            assert len(text) > 0
    except AssertionError as ae:
        return []
    
    for text in text_list:
        response = me_caller(text)
        responses.append(response)
        if "API error" in response:
            parsed_response.append(None)
        else:
            parsed_response.append(1) if response == 'BLOCK' else parsed_response.append(0)

    return responses, parsed_response, [(date.today(), MODEL_NAME)]*len(responses)


def me_caller(text, sleep_time=1, retry=5):
    try:
        prompt = prompt_template.format(user_text=text, guidelines=guidelines)

        # Send the prompt to Claude and get the response
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text
    except Exception as e:
        time.sleep(sleep_time)
        return me_caller(text, sleep_time=sleep_time+1, retry=retry-1) if retry > 0 else f"API error: {e}"
    
    return response
    
    
if __name__ == "__main__":
    start = time.time()
    data = pd.read_csv("Data/Movies/TMDB_with_ME.csv").head(1)
    responses = run_anthropic_ME(data["plots"].tolist())
    print(responses)
    data["test_responses"] = responses[0]
    end = time.time()
    print("Elapsed time", end-start)
    print(data)
    # data.to_csv("test_data.csv", index=False)