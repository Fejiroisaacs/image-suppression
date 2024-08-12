import pandas as pd, time, os, sys, matplotlib.pyplot as plt
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
from main import get_terms_mapping, conv_openAI_ME_data, ME_score_analysis, openAI_analysis
from run_openai_ME import run_me_caller
from run_perspective_ME import run_Perspective_ME
from run_llama_ME import run_OctoAI_ME
from run_google_ME import run_google_ME
from run_anthropic_ME import run_anthropic_ME

def main():
    cols = 'comment_text,male,toxicity,severe_toxicity,obscene,sexual_explicit,identity_attack,insult,threat,female,transgender,other_gender,heterosexual,homosexual_gay_or_lesbian,bisexual,other_sexual_orientation,christian,jewish,muslim,hindu,buddhist,atheist,other_religion,black,white,asian,latino,other_race_or_ethnicity,physical_disability,intellectual_or_learning_disability,psychiatric_or_mental_illness,other_disability'
    cols = cols.split(",")
    data = pd.read_csv("Data/CJadams/all_data.csv")[cols].dropna(subset=['comment_text'])
    print(data.shape)
    identity_cols = 'male,female,transgender,other_gender,heterosexual,homosexual_gay_or_lesbian,bisexual,other_sexual_orientation,christian,jewish,muslim,hindu,buddhist,atheist,other_religion,black,white,asian,latino,other_race_or_ethnicity,physical_disability,intellectual_or_learning_disability,psychiatric_or_mental_illness,other_disability'
    identity_cols = identity_cols.split(',') 
    data.dropna(subset=identity_cols, inplace=True)
    data.drop_duplicates(subset=['comment_text'], inplace=True)
    
    cols_to_change = cols.copy()
    cols_to_change.remove('comment_text')
    for col in cols_to_change:
        data[col] = data[col].apply(lambda x: 1 if x >= 0.5 else 0)
    print(data.shape)
    
    identities_dict = {
        'men' : ['male'],
        'women' : ['female'],
        'non-white' : ['black','asian','latino','other_race_or_ethnicity'],
        'white' : ['white'],
        'christian' : ['christian'],
        'non-christian' : ['jewish', 'muslim','hindu','buddhist','atheist','other_religion'],
        'lgbt-related' : ['homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation'],
        'straight' : ['heterosexual'],
        'disability' : ['physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness','other_disability'],
    }
    big_ids = []
    sub_ids = []
    
    for i in range(data.shape[0]):
        big_id = set()
        sub_id = set()
        for identity in identities_dict:
            for sub_identities in identities_dict[identity]:
                if data.iloc[i, :][sub_identities] == 1:
                    big_id.add(identity)
                    sub_id.add(sub_identities)
                    
        big_ids.append(";".join(big_id))
        sub_ids.append(";".join(sub_id))
        
    data["Big_identity"] = big_ids
    data["Sub_Identities"] = sub_ids
    
    data.to_csv("Data/CJadams/data.csv", index=False)


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=6)


def make_datasubset():
    data = pd.read_csv('Data/Cjadams/data_identity.csv').dropna(subset=['comment_text'])
    print(data.shape)
    no_identities_df = data.loc[pd.isnull(data[['Big_identity']]).any(axis=1)]
    print(no_identities_df.shape)
    data.dropna(subset=['Big_identity'], inplace=True)
    print(data.shape)
    
    data_subset = pd.concat([data, no_identities_df.sample(100000)])
    data_subset.to_csv('Data/Cjadams/data_identity_subset.csv')


def get_ME_responses() -> None:
    """Get ME responses from ME's and makes new csv with the results"""
    # dataset = pd.read_csv("Data/CJadams/data_identity_subset.csv").dropna(subset=['comment_text'])
    
    # # openAI call
    # print("Starting ME call...")
    # start = time.time()
    # OpenAI_ME_responses = run_me_caller(dataset, "comment_text")
    # dataset["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
    # dataset["OpenAI_ME_bool"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
    # dataset['OpenAI_data'] = OpenAI_ME_responses[1]
    # dataset.to_csv("Data/CJadams/data_with_ME.csv", index=False)
    # print("Elapsed time:", time.time() - start)
    
    # # perspective AI call
    # start = time.time()
    # perspective_responses = run_Perspective_ME(dataset["comment_text"].tolist())
    # dataset["perspective_ME_responses"] = perspective_responses[0]
    # dataset['Perspective_data'] = perspective_responses[1]
    # print("Elapsed time:", time.time() - start)
    # dataset.to_csv("Data/CJadams/data_with_ME.csv", index=False)
    

    dataset = pd.read_csv("Data/CJadams/data_with_ME.csv")
    # # Google ME call
    start = time.time()
    Google_responses = run_google_ME(dataset["comment_text"].tolist())
    dataset["Google_ME_responses"] = Google_responses[0]
    dataset['Google_data'] = Google_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv("Data/CJadams/data_with_ME.csv", index=False)
    
    # Anthropic ME call
    start = time.time()
    Anthropic_responses = run_anthropic_ME(dataset["comment_text"].tolist())
    dataset["Anthropic_ME_responses"] = Anthropic_responses[0]
    dataset["Anthropic_ME_bool"] = Anthropic_responses[1]
    dataset['Anthropic_data'] = Anthropic_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('Data/CJadams/data_with_ME.csv', index=False)
       
    # # llama AI call
    # start = time.time()
    # OctoAI_responses = run_OctoAI_ME(dataset["comment_text"].tolist())
    # dataset["OctoAI_ME_responses"] = OctoAI_responses[0]
    # dataset["OctoAI_ME_bool"] = OctoAI_responses[1]
    # dataset['OctoAI_data'] = OctoAI_responses[2]
    # print("Elapsed time:", time.time() - start)
    # dataset.to_csv("Data/CJadams/data_with_ME.csv", index=False)
    
    
if __name__ == "__main__":
    # main()
    # make_datasubset()
    # get_ME_responses()
    pass