
import requests, json, time
from datetime import date
url = "https://language.googleapis.com/v1/documents:moderateText?key=ENTER_API_KEY"


def me_caller(text:str):
    data = {
            "document": {
                        "type": "PLAIN_TEXT",
                        "language": "en",
                        "content": text
                        }
            }
    try:
        response = requests.post(url, data=json.dumps(data))
        if response.status_code != 200:
            time.sleep(5)
            response = requests.post(url, data=json.dumps(data))
    except Exception as e:
        return f"API error: {e}"

    try:
        return conv_response(response.json()["moderationCategories"])
    except Exception as e:
        return f"API error: {e}"
    

def conv_response(response_list: dict):
    response = {}
    for response_dict in response_list:
        response[response_dict["name"]] = response_dict["confidence"]
    
    return response


def run_google_ME(text_list:list):

    responses = []
    try:
        for text in text_list:
            assert type(text) == str 
            assert len(text) > 0
    except AssertionError as ae:
        print(ae)
        return []
    
    for text in text_list:
        responses.append(me_caller(text))
        
    return responses, [(date.today(), 'Unknown Model')]*len(responses)
