from googleapiclient import discovery
import time
from datetime import date

API_KEY = 'ENTER API KEY'
MODEL = 'v1alpha1'
def me_caller(text, client, sleep_time=1, retry=5):
    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': {'TOXICITY': {}}
    }
    try:
        response = client.comments().analyze(body=analyze_request).execute()
    except Exception as e:
        time.sleep(sleep_time)
        return me_caller(text, client, sleep_time=sleep_time+1, retry=retry-1) if retry > 0 else f"API error: {e}"
    
    return response

def run_Perspective_ME(text_list):
    client = discovery.build(
            "commentanalyzer",
            MODEL,
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
            )
    responses = []
    
    try:
        for text in text_list:
            assert type(text) == str 
            assert len(text) > 0
    except AssertionError as ae:
        print(ae)
        return []
    
    for text in text_list:
        responses.append(me_caller(text, client))
        
    return responses, [(date.today(), MODEL)]*len(responses)

