import argparse, sys, warnings
import pandas as pd

def clean_and_split(s):
    if isinstance(s, list):
        s = ";".join(s)

    s = s.lower()

    return s.split(";")

def extract_unique_codes(col):
    return set([item for sublist in col for item in sublist])


def codes_to_cols(df, col_name, col_prefix):

    df[col_name] = df[col_name].astype(str).apply(clean_and_split)

    code_strs = extract_unique_codes(df[col_name])

    for code in code_strs:
        df[f"{col_prefix}_{col_name_safe_str(code)}"] = df[col_name].apply(lambda x: code in x)

    return df

def col_name_safe_str(s):
    s = s.lower()
    s = s.replace(" & ", "_")
    s = s.replace(", ", "_")
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = s.replace("/", "_") # will overwrite the duplicates that have / instead of _
    
    return s

def dataset_cleanup(df):
    df = codes_to_cols(df, "Big_identity", "BI")
    df = codes_to_cols(df, 'Sub_Identities', "SI")

    # perspective ME -----------------------------------------------------
    df['perspective_ME_responses'] = df['perspective_ME_responses'].apply(lambda x: "None" if "API error:" in x or "does not support" in x else x)
    df['perspective_ME_responses'] = df['perspective_ME_responses'].apply(eval)

    df["perspective_ME_score"] = df['perspective_ME_responses'].apply(
        lambda x: x['attributeScores']['TOXICITY']['spanScores'][0]['score']['value'] if isinstance(x, dict) else None)

    df["perspective_ME_summary"] = df['perspective_ME_responses'].apply(
        lambda x: x['attributeScores']['TOXICITY']['summaryScore']['value'] if isinstance(x, dict) else None)
    
    # Google ME scores ---------------------------------------------------
    df['Google_ME_responses'] = df['Google_ME_responses'].apply(eval)
    google_codes = extract_unique_codes(df['Google_ME_responses'].apply(lambda x: x.keys()))

    for code in google_codes:
        goog_col = f"Google_{col_name_safe_str(code)}"
        df[goog_col] = df['Google_ME_responses'].apply(lambda x: x[code])
    
    # OpenAI ME ---------------------------------------------------------
    df['OpenAI_ME_responses'] = df['OpenAI_ME_responses'].apply(eval)
    openai_codes = extract_unique_codes(df['OpenAI_ME_responses'].apply(lambda x: x['results'][0]['category_scores'].keys()))
    openai_col_flag = f"OpenAI_flagged"
    df[openai_col_flag] = df['OpenAI_ME_responses'].apply(lambda x: x['results'][0]['flagged'])

    for code in openai_codes:
        openai_col_cat = f"OpenAI_category_flag_{col_name_safe_str(code)}"
        openai_col_score = f"OpenAI_category_scores_{col_name_safe_str(code)}"

        df[openai_col_cat] = df['OpenAI_ME_responses'].apply(lambda x: x['results'][0]['categories'][code])
        df[openai_col_score] = df['OpenAI_ME_responses'].apply(lambda x: x['results'][0]['category_scores'][code])

    # OpenAI Normalized ---------------------------------------------------------
    df['OpenAI_normalized'] = df['OpenAI_normalized'].apply(eval)
    openai_norm_codes = extract_unique_codes(df['OpenAI_normalized'].apply(lambda x: x.keys()))

    for code in openai_norm_codes:
        openai_norm_score = f"OpenAI_normalized_{col_name_safe_str(code)}"
        df[openai_norm_score] = df['OpenAI_normalized'].apply(lambda x: x[code])

    df['OpenAI_normalized_max'] = df.filter(like='OpenAI_normalized_').max(axis='columns')

    filterlist = df.filter(['level_0', 'template','toxicity','Title','PG score','PG-13 score','age_rating','genres','Release year','text',
        'dataset_name','show_name','episode_title','subset_name','Toxic','Offensive','Unnamed: 0','Big_identity', 'Sub_Identities', 
        'perspective_ME_responses', 'OpenAI_ME_responses','severe_toxicity','identity_attack','obscene','sexual_explicit','insult','threat',
        'OpenAI_normalized', 'Anthropic_ME_responses','Google_ME_responses','OctoAI_ME_responses','OpenAI_ME_bool','S','H','V','HR',
        'SH','S3','H2','V2','OpenAI_data','Anthropic_data','OctoAI_data','Perspective_data','Google_data','women', 
        'lgbt-related', 'disability', 'non-white', 'christian', 'non-christian', 'white', 'men', 'straight', 'lgbt', 
        'physical-disability', 'middle-eastern-north-african', 'other-non-christian', 'asian', 'black', 'latinx', 
        'trans-nonbinary', 'jewish', 'mental-disability', 'lesbian', 'native-americans', 'bisexual', 'islam', 'gay', 
        'muslim', 'physical', 'trans', 'mental', 'african american', 'female', 'male', 'sikh', 'protestant', 'middle eastern', 
        'poc', 'indian', 'chinese', 'latino', 'african', 'homosexual', 'european', 'native american', 'buddhism', 'deaf','blind','taoist','paralyzed',
        'BI_nan','SI_nan'])
    df.drop(filterlist, inplace = True, axis='columns')


    return df

def main():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '-g':
        gen_filepath = args[1]
    else:
        print("Path to Gen AI data missing")

    if len(args) == 4 and args[2] == '-t':
        trad_filepath = args[3]
    else:
        print("Path to traditional data missing")

    gdf = pd.read_csv(gen_filepath)
    gdf['GenAI'] = 'True'

    gdf = dataset_cleanup(gdf)
    
    tdf = pd.read_csv(trad_filepath)
    tdf['GenAI'] = 'False'

    tdf = dataset_cleanup(tdf)

    df = pd.concat([tdf,gdf])
    df.to_csv("Data/all_combined_forstats.csv")    


if __name__ == "__main__":
    main()