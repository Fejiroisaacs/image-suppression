import pandas as pd, os, sys, time, matplotlib.pyplot as plt
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


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=6)


def add_identity():
    wiki_terms = get_terms_mapping()
    
    data = pd.read_csv("Data/Stormfront/stormfront_data.csv")
    prompts = data["Sentence"].to_list()
    count = 0
    prompt_identities = []
    prompt_subidentities = []
    
    for prompt in prompts:
        prompt_identity = set()
        prompt_subidentity = set()
        prompt = prompt.replace(".", " ").replace(",", " ").replace("#", " ").replace("-", " ").replace("'", " ").replace("(", " ").replace(")", " ")
        for key in wiki_terms.keys():
            if f" {key.lower()} " in prompt.lower() \
                or f" {key.lower()}s " in prompt.lower()\
                or f" {key.lower()}es " in prompt.lower():
                count += 1
                prompt_identity.add(wiki_terms[key][0].lower())
                prompt_subidentity.add(wiki_terms[key][1].lower())
                
        prompt_identities.append(prompt_identity)
        prompt_subidentities.append(prompt_subidentity)
        
    print(sum([1 for id in prompt_identities if len(id) != 0]), len(prompts))
    data["Big_identity"] = [";".join(promptid) for promptid in prompt_identities]
    data["Sub_Identities"] = [";".join(promptid) for promptid in prompt_subidentities]
    
    data.sort_values(by=["Sub_Identities"], inplace=True, ascending=False)
    data.to_csv("Data/Stormfront/stormfront_data_identity.csv", index=False)
    
    all_ids = {}
    for identity in prompt_identities:
        for sub_id in identity:
            if sub_id not in all_ids.keys():
                all_ids[sub_id] = 1
            else:
                all_ids[sub_id] += 1
    
    if not os.path.exists("plots/Stormfront/"):
        os.makedirs("plots/Stormfront/")
    plt.bar(all_ids.keys(), all_ids.values(), color='maroon', width=0.8)
    plt.xticks(rotation=90)
    addlabels(list(all_ids.keys()), list(all_ids.values()))
    plt.ylabel("Sentences count")
    plt.xlabel("Identity")
    plt.savefig("plots/Stormfront/big_identity_dist.png", bbox_inches='tight', pad_inches=0.5)
    plt.show()


def make_data():
    master_file = pd.read_csv("Data/Stormfront/annotations_metadata.csv")
    final_df = pd.DataFrame(columns=["Sentence", 'label'])

    for file_id, label in zip(master_file['file_id'], master_file['label']):
        sentence = open(f"Data/Stormfront/all_files/{file_id}.txt", encoding="utf8").readline()
        label = 1 if label == "hate" else 0
        final_df.loc[len(final_df.index), ['Sentence', 'label']] = [sentence, label] 
    
    final_df.to_csv("Data/Stormfront/stormfront_data.csv", index=False)


def get_ME_responses() -> None:
    """Get ME responses from ME's and makes new csv with the results"""
    dataset = pd.read_csv("Data/Stormfront/stormfront_data_identity.csv")
    # perspective AI call
    start = time.time()
    perspective_responses = run_Perspective_ME(dataset["Sentence"].tolist())
    dataset["perspective_ME_responses"] = perspective_responses[0]
    dataset['Perspective_data'] = perspective_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv("Data/Stormfront/stormfront_data_ME.csv", index=False)
    
    # openAI call
    print("Starting ME call...")
    start = time.time()
    OpenAI_ME_responses = run_me_caller(dataset, "Sentence")
    dataset["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
    dataset["OpenAI_ME_bool"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
    dataset['OpenAI_data'] = OpenAI_ME_responses[1]
    dataset.to_csv("Data/Stormfront/stormfront_data_ME.csv", index=False)
    print("Elapsed time:", time.time() - start)

    # Google ME call
    start = time.time()
    Google_responses = run_google_ME(dataset["Sentence"].tolist())
    dataset["Google_ME_responses"] = Google_responses[0]
    dataset['Google_data'] = Google_responses[1]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv("Data/Stormfront/stormfront_data_ME.csv", index=False)
    
    # Anthropic ME call
    start = time.time()
    Anthropic_responses = run_anthropic_ME(dataset["Sentence"].tolist())
    dataset["Anthropic_ME_responses"] = Anthropic_responses[0]
    dataset["Anthropic_ME_bool"] = Anthropic_responses[1]
    dataset['Anthropic_data'] = Anthropic_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv('Data/Stormfront/stormfront_data_ME.csv', index=False)
    
    # llama AI call
    start = time.time()
    OctoAI_responses = run_OctoAI_ME(dataset["Sentence"].tolist())
    dataset["OctoAI_ME_responses"] = OctoAI_responses[0]
    dataset["OctoAI_ME_bool"] = OctoAI_responses[1]
    dataset['OctoAI_data'] = OctoAI_responses[2]
    print("Elapsed time:", time.time() - start)
    dataset.to_csv("Data/Stormfront/stormfront_data_ME.csv", index=False)
    
    
def run_stormfront_audit():
    make_data()
    add_identity()
    get_ME_responses()
    
    ME_score_analysis(identity_type="small", data_type="Stormfront", file="stormfront_data_ME.csv", ME="PerspectiveAI")
    ME_score_analysis(identity_type="big", data_type="Stormfront", file="stormfront_data_ME.csv", ME="PerspectiveAI")
    openAI_analysis(identity_type="Big_identity", data_type="Stormfront", file="stormfront_data_ME.csv", toxic_col="label")
    openAI_analysis(identity_type="Sub_Identities", data_type="Stormfront", file="stormfront_data_ME.csv", toxic_col="label")
    ME_score_analysis(identity_type="big", data_type="Stormfront", file="stormfront_data_ME.csv", ME="Google")
    openAI_analysis(identity_type="Big_identity", data_type="Stormfront",
                    file="stormfront_data_ME.csv", toxic_col="label", ME="OctoAI")
    openAI_analysis(identity_type="Big_identity", data_type="Stormfront",
                    file="stormfront_data_ME.csv", toxic_col="label", ME="Anthropic")


if __name__ == "__main__":
    run_stormfront_audit()