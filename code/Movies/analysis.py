import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import os, sys, ast
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
from main import update_overall_table, parse_ME_response, ME_score_analysis


def score_helper(is_toxic:str, rating:str, categories_dict:dict, 
                 sub_ids:list, rating_identity_dict:dict, flagged_true:int) -> None:
    # bins = ["Age", "Disability", "LGBT-related", "Race-Ethnicity", "Religion", "Gender", "No identity"]
    categories_dict[rating]["pred"].append(flagged_true)
    if is_toxic:
        categories_dict[rating]["true"].append(1)
        for ids in sub_ids:
            if ids.lower() in rating_identity_dict[rating].keys():
                rating_identity_dict[rating][ids.lower()]["true"].append(1)
                rating_identity_dict[rating][ids.lower()]["pred"].append(flagged_true)
            elif ids:
                rating_identity_dict[rating][ids.lower()] = {"true": [1], "pred": [flagged_true]}
    else:
        categories_dict[rating]["true"].append(0)
        for ids in sub_ids:
            if ids.lower() in rating_identity_dict[rating].keys():
                rating_identity_dict[rating][ids.lower()]["true"].append(0)
                rating_identity_dict[rating][ids.lower()]["pred"].append(flagged_true)
            elif ids:
                rating_identity_dict[rating][ids.lower()] = {"true": [0], "pred": [flagged_true]}
    

def make_stats_graphs(ME, identity_type) -> None:
    dataset = pd.read_csv("./Data/Movies/TMDB_with_ME.csv")

    categories_dict = {"G" : {"true": [], "pred": []},
                       "PG" : {"true": [], "pred": []},
                       "PG-13" : {"true": [], "pred": []},
                       "R" : {"true": [], "pred": []},
                       "NC-17" : {"true": [], "pred": []},
                       }
    rating_identity_dict = {"G" : {}, "PG" : {}, "PG-13" : {}, "R" : {}, "NC-17" : {}}
    identity_type = "Big_identity" if identity_type == "big" else "Sub_Identities"
    for g, pg, pg13, r, nc17, flagged_true, identities in zip(dataset["G score"], dataset["PG score"], 
                                                             dataset["PG-13 score"], dataset["R score"], 
                                                             dataset["NC-17 score"], dataset[f"{ME}_ME_bool"],
                                                             dataset[identity_type]):
        
        sub_ids = identities.split(";") if type(identities) != float else ["No identity"]
        score_helper(g, "G", categories_dict, sub_ids, rating_identity_dict, flagged_true)
        score_helper(pg, "PG", categories_dict, sub_ids, rating_identity_dict, flagged_true)
        score_helper(pg13, "PG-13", categories_dict, sub_ids, rating_identity_dict, flagged_true)
        score_helper(r, "R", categories_dict, sub_ids, rating_identity_dict, flagged_true)
        score_helper(nc17, "NC-17", categories_dict, sub_ids, rating_identity_dict, flagged_true)

    for key in categories_dict.keys():
        categories_dict[key] = score_calculator(categories_dict[key])

    entire_data_metric = {"G" : {"true" : dataset["G score"], "pred": dataset[f"{ME}_ME_bool"]},
                          "PG" : {"true" : dataset["PG score"], "pred": dataset[f"{ME}_ME_bool"]},
                          "PG-13" : {"true" : dataset["PG-13 score"], "pred": dataset[f"{ME}_ME_bool"]},
                          "R" : {"true" : dataset["R score"], "pred": dataset[f"{ME}_ME_bool"]},
                          "NC-17" : {"true" : dataset["NC-17 score"], "pred": dataset[f"{ME}_ME_bool"]}}
    
    for key in rating_identity_dict.keys():
        entire_data_metric[key] = score_calculator(entire_data_metric[key])
        identities_fpr = {}
        max_fpr = {"identity": "", "FPR": -np.inf}
        for identity in rating_identity_dict[key].keys():
            rating_identity_dict[key][identity] = score_calculator(rating_identity_dict[key][identity])
            identities_fpr[identity] = rating_identity_dict[key][identity]["FPR"]
            if rating_identity_dict[key][identity]["FPR"] > max_fpr["FPR"]:
                max_fpr["FPR"] = rating_identity_dict[key][identity]["FPR"]
                max_fpr["identity"] = identity
        
        graph_rating_acc(ME, key, identity_type, "Accuracy", rating_identity_dict[key], categories_dict[key])
        graph_rating_acc(ME, key, identity_type, "Precision", rating_identity_dict[key], categories_dict[key])
        graph_rating_acc(ME, key, identity_type, "Recall", rating_identity_dict[key], categories_dict[key])
        graph_rating_acc(ME, key, identity_type, "TPR", rating_identity_dict[key], categories_dict[key])
        graph_rating_acc(ME, key, identity_type, "FPR", rating_identity_dict[key], categories_dict[key])

        update_overall_table("Movie plots", ME, f"FPR <= {key}", \
            max_fpr['identity'], max_fpr['FPR'], entire_data_metric[key]["FPR"], str(identities_fpr))
        
    identity_type = "Small_identities" if identity_type == "Sub_Identities" else "Big_identities"
    if not os.path.exists(f"./Data/Movies/{ME}"):
        os.makedirs(f"./Data/Movies/{ME}/")
        
    pd.DataFrame(rating_identity_dict).to_csv(f"./Data/Movies/{ME}/{identity_type}_rating_scores.csv")
    pd.DataFrame(categories_dict).to_csv(f"./Data/Movies/{ME}/rating_scores.csv")

    
def get_categories(identity_type) -> dict:
    tagged_files = os.listdir("./Data/Movies/tags")
    sub_categories_dict = {}
    big_categories_dict = {}
    for category in tagged_files:
        identitiy_bins = os.listdir(f"./Data/Movies/tags/{category}")
        sub_categories_dict[category] = []
        big_categories_dict[category] = []
        for identity_bin in identitiy_bins:
            bin_files = os.listdir(f"./Data/Movies/tags/{category}/{identity_bin}")
            big_categories_dict[category].append(identity_bin)
            for file in bin_files:
                sub_categories_dict[category].append(file.split("_tagged_movies.csv")[0])
                
    return sub_categories_dict if identity_type == "Sub_Identities" else big_categories_dict


def graph_rating_acc(ME, rating, identity_type, metric:str, identities_dict:dict, category_scores:dict) -> None:

    identity_path = "Small identities" if identity_type == "Sub_Identities" else "Big identities"
    if not os.path.exists(f"./plots/Movies/{ME}/{identity_path}/{metric}"):
        os.makedirs(f"./plots/Movies/{ME}/{identity_path}/{metric}")
    
    categories_dict = get_categories(identity_type)
    # print(categories_dict, identities_dict)
    plt.figure(figsize=(15, 10))
    plt.xticks(rotation=90)
    
    for category in categories_dict.keys():
        identites = categories_dict[category]
        metrics = []
        for identity in identites:
            identity = identity.lower()
            metrics.append(identities_dict[identity][metric]) if identity in identities_dict.keys() else metrics.append(0)
        plt.bar(identites, metrics, width=0.7, label=category)
        
    plt.axhline(y = category_scores[metric], color = 'black', linestyle = '-', label=f"Average {metric}")
    plt.xlabel('Identities')
    plt.ylabel(metric)
    plt.title(f'Flagged {metric} for {rating} by {identity_path} ({ME})')
    save_path = f"./plots/Movies/{ME}/{identity_path}/{metric}/{rating}_{metric.lower()}_chart.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.8)
    plt.clf()
    plt.close()


def score_calculator(metric_dict:dict) -> dict:
    cm = confusion_matrix(metric_dict["true"], metric_dict["pred"], labels=[0,1])
    scores_dict = {"Accuracy": 0, "Precision": 0, "Recall": 0}

    scores_dict["Accuracy"] = round(accuracy_score(metric_dict["true"], metric_dict["pred"]), 4)
    scores_dict["Precision"] = round(precision_score(metric_dict["true"], metric_dict["pred"], zero_division=0), 4)
    scores_dict["Recall"] = round(recall_score(metric_dict["true"], metric_dict["pred"], zero_division=0), 4)
    scores_dict["TPR"] = round(cm[1,1] / (cm[1,1] + cm[0,1]) if cm[1,1] + cm[0,1] > 0 else 1, 4)
    scores_dict["FPR"] = round(cm[0, 1] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0, 4)

        
    return scores_dict
    

def make_bounds():
    dataset = pd.read_csv("./Data/Movies/TMDB_with_ME.csv")
    responses = parse_ME_response(dataset["OpenAI_ME_responses"].tolist())
    category_bounds = {}
    for response in responses:
        for category in response["categories"].keys():
            if category not in category_bounds.keys():
                category_bounds[category] = {"upper_bound" : np.inf, "lower_bound" : -np.inf}
            if response["categories"][category]:
                if float(response["category_scores"][category]) < category_bounds[category]["upper_bound"]:
                    category_bounds[category]["upper_bound"] = float(response["category_scores"][category])
            else:
                if float(response["category_scores"][category]) > category_bounds[category]["lower_bound"]:
                    category_bounds[category]["lower_bound"] = float(response["category_scores"][category])
    upper = []
    lower = []
    categories = []
    
    for category in category_bounds.keys():
        upper.append(category_bounds[category]["upper_bound"])
        lower.append(category_bounds[category]["lower_bound"])
        categories.append(category)
    
    bounds_df = pd.DataFrame()
    bounds_df["Categories"] = categories
    bounds_df["upper_bound"] = upper
    bounds_df["lower_bound"] = lower
    bounds_df.to_csv("./Data/Movies/flagging_bounds.csv")
    
   
def graph_by_identity():
    dataset = pd.read_csv("./Data/Movies/TMDB_with_ME.csv")
    tagged_files = os.listdir("./Data/Movies/tags")
    identities = {}

    # getting all identities
    for category in tagged_files:
        identitiy_bins = os.listdir(f"./Data/Movies/tags/{category}")
        for identity_bin in identitiy_bins:
            bin_files = os.listdir(f"./Data/Movies/tags/{category}/{identity_bin}")
            for file in bin_files:
                curr_file = pd.read_csv(f"./Data/Movies/tags/{category}/{identity_bin}/{file}")
                identity = list(curr_file.columns)[-2]
                # values represent identity count and flag true count
                identities[identity.lower()] = [0, 0]
            identities[identity_bin.lower()] = [0, 0]
        
    movie_identities = dataset["Sub_Identities"].tolist()
    flagged_true = dataset["OpenAI_ME_bool"].tolist()
    age_rating = dataset["rating"].tolist()
    
    for identity, flagged in zip(movie_identities, flagged_true):
        if type(identity) is not float:
            sub_identity = identity.split(";")
            for value in sub_identity:
                identities[value.lower()][0] += 1
                if flagged:
                    identities[value.lower()][1] += 1
    df_list = []
    for identity in identities:
        if identities[identity][0] != 0:
            identities[identity] = identities[identity][1] / identities[identity][0]
            df_list.append([identity, identities[identity]])
    
    df = pd.DataFrame(df_list, columns=["Identity", "proportion_flagged"])
    df = df.sort_values(by='proportion_flagged', ascending=False)
    
    plt.figure(figsize=(15, 10))
    plt.bar(df["Identity"], df['proportion_flagged'], color='maroon', width=0.8)
    plt.xticks(rotation=90)
    plt.xlabel('Identities')
    plt.ylabel('Proportion (# flagged episodes / # total episodes)')
    plt.title('Flagged proportions by identity')
    plt.savefig("./plots/Movies/identities_flagged.png", bbox_inches='tight', pad_inches=0.8)
    
    age_identity_dict = {}
    for identity, flagged, rating in zip(movie_identities, flagged_true, age_rating):
        if rating not in age_identity_dict.keys():
            age_identity_dict[rating] = {}
            
        if type(identity) is not float:
            sub_identity = identity.split(";")
            for value in sub_identity:
                if value.lower() not in age_identity_dict[rating].keys():
                    age_identity_dict[rating][value.lower()] = [0, 0]
                    
                age_identity_dict[rating][value.lower()][0] += 1
                if flagged:
                    age_identity_dict[rating][value.lower()][1] += 1
                    
    df2_list = []
    col_name = ["Age rating"]
    count = 0
    for age in age_identity_dict:
        props = [age]
        for identity in identities:
            if identity in age_identity_dict[age].keys():
                if age_identity_dict[age][identity][0] != 0:
                    age_identity_dict[age][identity] = age_identity_dict[age][identity][1] / age_identity_dict[age][identity][0]
                else:
                    age_identity_dict[age][identity] = 0
                    
            else:
                age_identity_dict[age][identity] = 0
            if count == 0:
                col_name.append(identity)
            props.append(age_identity_dict[age][identity])
        count += 1
        df2_list.append(props)
    
    df = pd.DataFrame(df2_list, columns=col_name)
    df.to_csv("./Data/Movies/flagged_stats_age_identity.csv", index=True)

    df.plot(x='Age rating',
        kind='bar',
        stacked=False, 
        title='Flagged proportion by identity per age',
        figsize=(20, 15),
        xlabel='Age ratings',
        ylabel='Proportion (# flagged episodes / # total episodes)',
        width=0.8,
        rot=0,
        fontsize=12
        )

    plt.savefig("./plots/Movies/age_identities_flagged.png", pad_inches=1, bbox_inches='tight')

def graph_individual_age():
    if not os.path.exists(f"./plots/Movies/rating_identity_flags"):
        os.makedirs(f"./plots/Movies/rating_identity_flags")
        
    dataset = pd.read_csv("./Data/Movies/TMDB_with_ME.csv")
    average_flag = round(sum(dataset['OpenAI_ME_bool'].tolist())/dataset.shape[0], 4)
    df = pd.read_csv("./Data/Movies/flagged_stats_age_identity.csv")
    df.drop(columns=["Unnamed: 0"], inplace=True)
    age_rating_list = df.to_dict(orient='records')
    bins = ["disability", "men", "women", "non-white", "non-christian", "non-white", "lgbt-related"]

    plt.figure(figsize=(15, 10))
    for age_dict in age_rating_list:
        clean_age_dict = {}
        age_rating = age_dict['Age rating']
        del age_dict['Age rating']
        mini_dict = {}
        
        for key in age_dict.keys():
            if key not in bins:
                mini_dict[key] = age_dict[key]
            else:
                clean_age_dict[key] = mini_dict if len(mini_dict.keys()) > 0 else {key: age_dict[key]}
                mini_dict = {}
                plt.bar(clean_age_dict[key].keys(), clean_age_dict[key].values(), width=0.7)
        
        plt.axhline(y = average_flag, color = 'black', linestyle = '-', label="Average flag") 
        plt.xlabel('Identities')
        plt.ylabel('Proportion (# flagged episodes / # total episodes)')
        plt.title(f'Flagged proportions for {age_rating} by identity')
        plt.xticks(rotation=90)
        plt.savefig(f"./plots/Movies/rating_identity_flags/{age_rating}_identities_flagged.png", bbox_inches='tight', pad_inches=0.8)
        plt.clf()


def general_table(identity_type):
    APIs = ["OctoAI", "OpenAI"]

    data = pd.DataFrame(columns=["Rating", "Identity", "iFPR"])
    rating = []
    ide = []
    ifpr = []
    api = []
    
    for API in APIs:
        overall_stats_df = pd.read_csv(f"Data/Movies/{API}/rating_scores.csv")
        identity_stats_df = pd.read_csv(f"Data/Movies/{API}/{identity_type}_identities_rating_scores.csv")
        identity_stats_df.rename(columns={'Unnamed: 0': "Identity"}, inplace=True)
        overall_stats_df.rename(columns={'Unnamed: 0': "Metric"}, inplace=True)
        identity_stats_dict = identity_stats_df.to_dict(orient="records")
        fprs = overall_stats_df[overall_stats_df['Metric'] == 'FPR']
        for identity_dict in identity_stats_dict:
            identity = identity_dict["Identity"]
            del identity_dict["Identity"]
            for age_rating in identity_dict.keys():
                age_rating_dict = ast.literal_eval(identity_dict[age_rating])
                if fprs[age_rating].to_list()[0] != 0:
                    if identity != "no identity":
                        rating.append(age_rating)
                        ifpr.append(round(age_rating_dict['FPR']/fprs[age_rating].to_list()[0], 4))
                        ide.append(identity)
                        api.append(API)
            
    data["Rating"] = rating
    data["iFPR"] = ifpr
    data["Identity"] = ide
    data["API"] = api
    data.sort_values(by=["Rating", "iFPR"], inplace=True, ascending=False)
    data.to_csv(f"Data/Movies/{identity_type}_ifpr_table.csv", index=False)
        
    
if __name__ == "__main__":
    # make_bounds()
    graph_by_identity()
    make_stats_graphs(ME="OpenAI", identity_type="small")
    make_stats_graphs(ME="OpenAI", identity_type="big")
    make_stats_graphs(ME="OctoAI", identity_type="small")
    make_stats_graphs(ME="OctoAI", identity_type="big")
    ME_score_analysis(identity_type="big", data_type="Movies", file="TMDB_with_ME.csv", ME="PerspectiveAI")
    ME_score_analysis(identity_type="small", data_type="Movies", file="TMDB_with_ME.csv", ME="PerspectiveAI")
    ME_score_analysis(identity_type="big", data_type="Movies", file="TMDB_with_ME.csv", ME="Google")
    ME_score_analysis(identity_type="small", data_type="Movies", file="TMDB_with_ME.csv", ME="Google")
    graph_individual_age()
    general_table("Big")
    general_table("Small")

    pass
