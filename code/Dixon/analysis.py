import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
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
from main import score_calculator
from main import update_overall_table, ME_score_analysis


def make_stats_graphs(ME):
    data = pd.read_csv("Data/Dixon/Data_with_ME.csv")
    identity_dict = {}
    data["toxicity"] = data["toxicity"].apply(lambda x: 1 if x == 'toxic' else 0)
    dataset_score = score_calculator({"true": data["toxicity"], "pred": data[f"{ME}_ME_bool"]})
    for toxic, identity, identity_group, ME_score in zip(data["toxicity"], data["Sub_Identities"],
                                             data["Big_identity"], data[f"{ME}_ME_bool"]):
        if identity_group not in identity_dict.keys():
            identity_dict[identity_group] = {}
        if identity not in identity_dict[identity_group].keys():
            identity_dict[identity_group][identity] = {"true": [toxic], "pred": [ME_score]}
        else:
            identity_dict[identity_group][identity]["true"].append(toxic)
            identity_dict[identity_group][identity]["pred"].append(ME_score)
        
    
    for identity_group in identity_dict.keys():
        identities_fpr = {}
        max_fpr = {"identity": "", "FPR": -np.inf}
        for identity in identity_dict[identity_group].keys():
            identity_dict[identity_group][identity] = score_calculator(identity_dict[identity_group][identity])
            identities_fpr[identity] = identity_dict[identity_group][identity]["FPR"]
            if identity_dict[identity_group][identity]["FPR"] > max_fpr["FPR"]:
                max_fpr["FPR"] = identity_dict[identity_group][identity]["FPR"]
                max_fpr["identity"] = identity
                
        graph_rating_acc(ME, identity_group, "FPR", identity_dict[identity_group])
        graph_rating_acc(ME, identity_group, "Accuracy", identity_dict[identity_group])
        
        update_overall_table("Dixon", "OpenAI", f"FPR", \
            max_fpr['identity'], max_fpr['FPR'], dataset_score["FPR"], str(identities_fpr))
    
    if not os.path.exists(f"./Data/Dixon/{ME}"):
        os.makedirs(f"./Data/Dixon/{ME}/")
    
    df = pd.DataFrame(columns=["Identity", "Stats"])
    for key in identity_dict.keys():
        df.loc[len(df.index)] = [key, identity_dict[key]]
    df.to_csv(f"./Data/Dixon/{ME}/rating_scores.csv", index=False)
    

def graph_rating_acc(ME, identity_group:str, metric:str, identities_dict:dict) -> None:

    if not os.path.exists(f"./plots/Dixon/{ME}/{metric}"):
        os.makedirs(f"./plots/Dixon/{ME}/{metric}")
    
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation=90)
    metrics = []
    for identity in identities_dict.keys():
        metrics.append(identities_dict[identity][metric])
    plt.bar(list(identities_dict.keys()), metrics, width=0.7, label=identity_group) 
        
    
    # plt.axhline(y = category_scores[metric], color = 'black', linestyle = '-', label=f"Average {metric}")
    plt.xlabel('Identities')
    plt.ylabel(metric)
    plt.title(f'Flagged {metric} for {identity_group} by identity')
    save_path = f"./plots/Dixon/{ME}/{metric}/{identity_group}_{metric.lower()}_chart.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.8)
    plt.clf()
    plt.close()

    
if __name__ == "__main__":
    make_stats_graphs("OpenAI")
    make_stats_graphs("OctoAI")
    
    # ME_score_analysis(identity_type="small", data_type="Dixon", file="Data_with_ME.csv", ME="PerspectiveAI")
    # ME_score_analysis(identity_type="big", data_type="Dixon", file="Data_with_ME.csv", ME="PerspectiveAI")
    # ME_score_analysis(identity_type="small", data_type="Dixon", file="Data_with_ME.csv", ME="Google")
    # ME_score_analysis(identity_type="big", data_type="Dixon", file="Data_with_ME.csv", ME="Google")