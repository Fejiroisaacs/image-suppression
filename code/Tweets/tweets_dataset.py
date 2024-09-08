# https://arxiv.org/pdf/2010.12421 -- dataset from paper
import pandas as pd
import sys
import os
import ast
from sklearn.metrics import accuracy_score
# https://www.geeksforgeeks.org/python-import-from-parent-directory/
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)
# adding the parent directory to 
# the sys.path.
sys.path.append(parent)
from run_openai_ME import run_me_caller
from run_llama_ME import run_OctoAI_ME
from run_perspective_ME import run_Perspective_ME
from run_anthropic_ME import run_anthropic_ME
from run_google_ME import run_google_ME
from main import conv_openAI_ME_data
import time
from main import get_terms_mapping, openAI_analysis, ME_score_analysis


def add_identity_tags():
    subsets = ["hate", "offensive"]
    for subset in subsets:
        twitter_data = pd.read_csv(f"Data/Tweets/{subset}_twitter_data_with_ME.csv")
        tweets = twitter_data["Text"].tolist()

        count = 0
        tweet_identities = []
        tweet_subidentities = []
        wiki_terms = get_terms_mapping()
        
        for tweet in tweets:
            tweet_identity = set()
            tweet_subidentity = set()
            tweet = tweet.replace(".", " ").replace(",", " ").replace("#", " ").replace("-", " ").replace("'", " ").replace("(", " ").replace(")", " ")
            for key in wiki_terms.keys():
                if f" {key.lower()} " in tweet.lower() \
                    or f" {key.lower()}s " in tweet.lower()\
                    or f" {key.lower()}es " in tweet.lower():
                    count += 1
                    tweet_identity.add(wiki_terms[key][0].lower())
                    tweet_subidentity.add(wiki_terms[key][1].lower())
                    
            tweet_identities.append(tweet_identity)
            tweet_subidentities.append(tweet_subidentity)
        
        print(sum([1 for id in tweet_identities if len(id) != 0]), len(tweets))
        twitter_data["Big_identity"] = [";".join(tweetid) for tweetid in tweet_identities]
        twitter_data["Sub_Identities"] = [";".join(tweetid) for tweetid in tweet_subidentities]

        # twitter_data = twitter_data[twitter_data.Big_identity != ""]
        twitter_data.to_csv(f"Data/Tweets/{subset}_twitter_data_with_ME.csv", index=False)

    
def get_ME_responses() -> None:
    files = ["hate", "offensive"]
    
    for file in files:
        twitter_data = pd.read_csv(f"Data/Tweets/{file}_twitter_data.csv")
        # openAI call
        twitter_data = twitter_data.dropna().reset_index()
        print(f"Starting ME call for {file}_twitter_data...")
        OpenAI_ME_responses = run_me_caller(twitter_data, "Text")
        twitter_data["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
        twitter_data["OpenAI_ME_bool"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
        twitter_data['OpenAI_data'] = OpenAI_ME_responses[1]
        twitter_data.to_csv(f"Data/Tweets/{file}_twitter_data_with_ME.csv", index=False)
        print(f"Finished ME call for {file}_twitter_data")
        
        # Octo AI call
        start = time.time()
        OctoAI_responses = run_OctoAI_ME(twitter_data['Text'].tolist())
        twitter_data["OctoAI_ME_responses"] = OctoAI_responses[0]
        twitter_data["OctoAI_ME_bool"] = OctoAI_responses[1]
        twitter_data['OctoAI_data'] = OctoAI_responses[2]
        print("Elapsed time:", time.time() - start)
        twitter_data.to_csv(f"Data/Tweets/{file}_twitter_data_with_ME.csv", index=False)
        
        # perspective AI call
        start = time.time()
        perspective_responses = run_Perspective_ME(twitter_data['Text'].tolist())
        twitter_data["perspective_ME_responses"] = perspective_responses[0]
        twitter_data['Perspective_data'] = perspective_responses[1]
        print("Elapsed time:", time.time() - start)
        twitter_data.to_csv(f"Data/Tweets/{file}_twitter_data_with_ME.csv", index=False)    

        # Google ME call
        start = time.time()
        Google_responses = run_google_ME(twitter_data['Text'].tolist())
        twitter_data["Google_ME_responses"] = Google_responses[0]
        twitter_data['Google_data'] = Google_responses[1]
        print("Elapsed time:", time.time() - start)
        twitter_data.to_csv(f"Data/Tweets/{file}_twitter_data_with_ME.csv", index=False)    
        
        # Anthropic ME call
        start = time.time()
        Anthropic_responses = run_anthropic_ME(twitter_data['Text'].tolist())
        twitter_data["Anthropic_ME_responses"] = Anthropic_responses[0]
        twitter_data["Anthropic_ME_bool"] = Anthropic_responses[1]
        twitter_data['Anthropic_data'] = Anthropic_responses[2]
        print("Elapsed time:", time.time() - start)
        twitter_data.to_csv(f"Data/Tweets/{file}_twitter_data_with_ME.csv", index=False)
        

def make_data():
    files = ["test", "val", "train"]
    folders = ["hate", "offensive"]
    for folder in folders:
        texts = []
        toxicities = []
        for file in files:
            with open(f"Data/Tweets/{folder}/{file}_text.txt", encoding="utf8") as text:
                text = text.read().split("\n")[:-1] # last element is empty
                texts.extend(text)
                
            with open(f"Data/Tweets/{folder}/{file}_labels.txt", encoding="utf8") as labels:
                labels = labels.read().split("\n")[:-1]
                toxicities.extend(labels)

        df = pd.DataFrame({"Text": texts, "Toxicity": toxicities})
        df.to_csv(f"Data/Tweets/{folder}_twitter_data.csv", index=False)


def run_tweets_audit():
    make_data()
    add_identity_tags()
    get_ME_responses()
    
    ME_score_analysis(identity_type="big", data_type="Tweets", file="hate_twitter_data_with_ME.csv", ME="Google", ex="hate_")
    ME_score_analysis(identity_type="small", data_type="Tweets", file="hate_twitter_data_with_ME.csv", ME="Google", ex="hate_")
    ME_score_analysis(identity_type="small", data_type="Tweets", file="hate_twitter_data_with_ME.csv", ME="PerspectiveAI", ex="hate_")
    ME_score_analysis(identity_type="big", data_type="Tweets", file="hate_twitter_data_with_ME.csv", ME="PerspectiveAI", ex="hate_")
    
    ME_score_analysis(identity_type="big", data_type="Tweets", file="offensive_twitter_data_with_ME.csv", ME="Google", ex="offensive_")
    ME_score_analysis(identity_type="small", data_type="Tweets", file="offensive_twitter_data_with_ME.csv", ME="Google", ex="offensive_")
    ME_score_analysis(identity_type="small", data_type="Tweets", file="offensive_twitter_data_with_ME.csv", ME="PerspectiveAI", ex="offensive_")
    ME_score_analysis(identity_type="big", data_type="Tweets", file="offensive_twitter_data_with_ME.csv", ME="PerspectiveAI", ex="offensive_")
    
    openAI_analysis(identity_type="Big_identity", data_type="Tweets",
                    file="hate_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="hate")
    openAI_analysis(identity_type="Sub_Identities", data_type="Tweets",
                    file="hate_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="hate")
    
    openAI_analysis(identity_type="Big_identity", data_type="Tweets",
                    file="offensive_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="offensive")
    openAI_analysis(identity_type="Sub_Identities", data_type="Tweets",
                    file="offensive_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="offensive")
    
    openAI_analysis(identity_type="Big_identity", data_type="Tweets",
                    file="hate_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="hate", ME="OctoAI")
    openAI_analysis(identity_type="Sub_Identities", data_type="Tweets",
                    file="hate_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="hate", ME="OctoAI")
    
    openAI_analysis(identity_type="Big_identity", data_type="Tweets",
                    file="offensive_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="offensive", ME="OctoAI")
    openAI_analysis(identity_type="Sub_Identities", data_type="Tweets",
                    file="offensive_twitter_data_with_ME.csv", toxic_col="Toxicity", ex="offensive", ME="OctoAI")
        
        
if __name__ == "__main__":
    run_tweets_audit()