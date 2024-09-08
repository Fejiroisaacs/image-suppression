"""
    Authors: Fejiro Anigboro + Charlie Crawford
"""
import pandas as pd
import sys, time, os, ast
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
from main import conv_openAI_ME_data
from run_openai_ME import run_me_caller
from run_perspective_ME import run_Perspective_ME
from run_llama_ME import run_OctoAI_ME
from run_google_ME import run_google_ME
from run_anthropic_ME import run_anthropic_ME

def get_ME_responses() -> None:
    """Get ME responses from OpenAI and makes new csv with the results"""
    dataset = pd.read_csv("./Data/Dixon/Data_with_identities.csv")
    print("Starting ME call...")
    OpenAI_ME_responses = run_me_caller(dataset, "phrase")
    dataset["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
    dataset["OpenAI_ME_bool"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
    dataset['OpenAI_data'] = OpenAI_ME_responses[1]
    dataset.to_csv("./Data/Dixon/Data_with_ME.csv", index=False)
    
    # perspective AI call
    start = time.time()
    perspective_responses = run_Perspective_ME(dataset['phrase'].tolist())
    dataset["perspective_ME_responses"] = perspective_responses[0]
    dataset['Perspective_data'] = perspective_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Dixon/Data_with_ME.csv', index=False)
    
    # # llama AI call
    start = time.time()
    OctoAI_responses = run_OctoAI_ME(dataset['phrase'].tolist())
    dataset["OctoAI_ME_responses"] = OctoAI_responses[0]
    dataset["OctoAI_ME_bool"] = OctoAI_responses[1]
    dataset['OctoAI_data'] = OctoAI_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Dixon/Data_with_ME.csv', index=False)
    
    # Google ME call
    start = time.time()
    Google_responses = run_google_ME(dataset['phrase'].tolist())
    dataset["Google_ME_responses"] = Google_responses[0]
    dataset['Google_data'] = Google_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Dixon/Data_with_ME.csv', index=False)
    
    # Anthropic ME call
    start = time.time()
    Anthropic_responses = run_anthropic_ME(dataset['phrase'].tolist())
    dataset["Anthropic_ME_responses"] = Anthropic_responses[0]
    dataset["Anthropic_ME_bool"] = Anthropic_responses[1]
    dataset['Anthropic_data'] = Anthropic_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('./Data/Dixon/Data_with_ME.csv', index=False)
    

def update_data() -> None:
    """Reads Dixon data and makes relevant data for this project"""
    
    data = pd.read_csv("./Data/Dixon/en_sentence_templates.csv")
    # grouping identities into categories
    lesbian = ['lgbt-related', 'lesbian']
    gay = ['lgbt-related', 'gay']
    bisexual = ['lgbt-related', 'bisexual']
    homosexual = ['lgbt-related', 'homosexual']
    male = ['men', 'male']
    female = ['women', 'female']
    middle_eastern = ['Non-White', 'middle eastern']
    muslim = ['Non-Christian', 'muslim']
    jewish = ['Non-Christian', 'jewish']
    buddhist = ['Non-Christian', 'buddhist']
    sikh = ['Non-Christian', 'sikh']
    taoist = ['Non-Christian', 'taoist']
    trans = ['lgbt-related', 'trans', 'transgender', 'nonbinary']
    lgbtq = ['lgbt-related', 'lgbt', 'lgbtq', 'queer']
    straight = ['straight', 'heterosexual']
    black = ['Non-White', 'black', 'african', 'african american']
    white = ['white', 'european']
    latinx = ['Non-White', 'latinx', 'latina', 'latino', 'hispanic', 'mexican']
    asian = ['Non-White', 'asian', 'indian', 'chinese', 'japanese']
    christian = ['Christian', 'christian', 'catholic', 'protestant']
    disabled = ['disability', 'disabled', 'blind', 'deaf', 'paralyzed']

    identities = [lesbian,gay,bisexual,homosexual,male,female,
                middle_eastern,muslim,jewish,buddhist,sikh,
                taoist,trans,lgbtq,straight,black,white,latinx,
                asian,christian,disabled]

    id_dict = {}

    all_ids = []

    for identity in identities:
        id_dict.update({key: identity[0] for key in identity})
        for id in identity:
            all_ids.append(id)
    
    i=0
    data['id'] = ""
    for row in data['phrase']:
        for id in all_ids:
            if(row.find(id)!=-1):
                data.at[i,'id'] = id
        i+=1
    
    # id groupings
    data['id_group'] = ""
    for i, row in enumerate(data['id']):
        if len(row)>0:
            data.at[i, 'id_group'] = id_dict[row]
            
    data = data[data['id'] !=""]
    data.to_csv("./Data/Dixon/Data_with_identities.csv", index=False)


def run_kaggle_audit():
    update_data()
    get_ME_responses()


if __name__ == "__main__":
    run_kaggle_audit()