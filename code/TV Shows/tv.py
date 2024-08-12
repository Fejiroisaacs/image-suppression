import sys, os
import pandas as pd
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
from Movies.movie_tagger import titles_recursively, get_movie_names_list
import requests
from run_anthropic_ME import run_anthropic_ME
from run_openai_ME import run_me_caller
from run_perspective_ME import run_Perspective_ME
from run_llama_ME import run_OctoAI_ME
from run_google_ME import run_google_ME
from main import conv_openAI_ME_data
import time
from Movies.get_movie_data import score_toxicity
from main import ME_score_analysis, openAI_analysis


def add_identity_using_tags(file_paths):
    # disability
    physical_disability=['disabled character','wheelchair','man uses a wheelchair','disabled veteran','blind man','blindness','blind cane','temporary blindness','blind driver','color blind','blind woman','blind gangster','blind worker','blind boy','pretending to be wheelchair bound','old woman uses a wheelchair','man uses a motorized wheelchair','man in a wheelchair','woman uses a wheelchair']
    mental_illness=['mental illness','mental instability','time in mental institution','pretending to be mentally challenged','mentally challenged','schizophrenia','manic behavior','retarded man', 'mentally ill']

    # sexuality
    lesbian_series_tags = ['lesbian','lesbian sex','lesbian couple','pregnant lesbian','lesbian interest','lesbian parent','lesbian affair','lesbian relationship','lesbian sex scene','implied lesbian sex','mistaken for a lesbian','first lesbian experience','lesbian parenthood','lesbian stalker','lesbian villain','lesbian ex lover','lesbian romance','lesbian lover','lesbian maid','lesbian cunnilingus','lesbian seduction','lesbian extramarital affair','lesbian sodomy','female female kiss']
    gay_series_tags = ['gay','gay character','gay kiss','gay son','gay sex','gay muslim','gay teenager','gay pride','gay parents','closeted gay','gay interest','pretending to be gay','gay bashing','gay man','gay pornography','mistaken for gay','gay bathhouse','implied gay sex','gay relationship','interrupted gay sex','gay slur','gay acceptance','gay angst','gay friend','gay dog','gay prisoner','gay joke','gay couple','gaydar','discovering someone is gay','gay man straight woman relationship','closeted gay man','gay bar','gay porn','suspected of being gay','gay co worker','male male kiss']
    bisexual_series_tags = ['bisexual','bisexuality','bisexual woman']
    trans_series_tags = ['transgender','transvestite','transgender prostitute','transphobia']
    straight_series_tags = ['husband wife relationship','ex husband ex wife relationship','wife murders her husband','wife leaves her husband','husband wife hug','husband murders his wife','husband and wife reunited','husband wife kiss','husband wife reconciliation','wife confesses infidelity to husband','husband wife estrangement','husband meets wife','ex husband ex wife reunion','husband and wife criminals','wife shoots her husband','husband accused of murdering his wife','husband cheats on wife','husband hits his wife with a belt','husband hits his wife','husband slaps wife','boyfriend girlfriend relationship','ex boyfriend ex girlfriend relationship','ex boyfriend ex girlfriend sex','ex boyfriend ex girlfriend reunion','male female kiss']
    
    # religon
    muslim_tags=['muslim','gay muslim','muslim girl','muslim family','muslim prayer','muslim woman','imam']
    christian=['christmas','christmas episode','reference to jesus christ','christmas tree','christmas party','jesus christ character','christmas lights','christmas gift','christmas eve','christmas pageant','christmas music','christmas present','christmas carol','office christmas party','family christmas','reference to a christmas carol','making a christmas card','christian subtext','antichrist','reference to the antichrist','christmas special','christmas bonus','christmas decorations','christianity','christianism','christian','christian cross','christmas song','christmas decoration','decorating a christmas tree','priest','murder of a priest','priest killed','impersonating a priest','sex with a priest','lapsed catholic','catholic church','catholic','catholic school','protestant church','protestant clergyman','mass']
    jewish = ['judaism','jewish wedding','jewish american','jewish people','rabbi','reference to moses','jewish','jew']
    
    tag_dict = {
        "Disability": {"Physical": physical_disability, "Mental": mental_illness},
        "LGBT-related": {"Lesbian": lesbian_series_tags, "Gay": gay_series_tags,
                         'Bisexual': bisexual_series_tags, "Trans": trans_series_tags},
        "Straight": {"Straight": straight_series_tags},
        "Non-Christian": {"Muslim": muslim_tags, "Jewish": jewish},
        "Christian": {"Christian": christian},
    }
    for file_path in file_paths:
        file = pd.read_csv(file_path)
        if 'Big_identity' not in file.columns:
            file['Big_identity'] = None
        if 'Sub_Identities' not in file.columns:
            file['Sub_Identities'] = None
        big_identities = []
        sub_identities = []
        
        for episode_tags, big, sub in file[['tags', 'Big_identity', 'Sub_Identities']].to_numpy():
            big = big.split(';') if type(big) == str else []
            sub = sub.split(';') if type(sub) == str else []
            if type(episode_tags) == str:
                for identity in tag_dict:
                    for sub_identity in tag_dict[identity]:
                        for tag in tag_dict[identity][sub_identity]:
                            if tag in episode_tags:
                                big.append(identity)
                                sub.append(sub_identity)
                                break
            big_identities.append(";".join(set(big)))
            sub_identities.append(";".join(set(sub)))
        
        file['Big_identity'] = big_identities
        file['Sub_Identities'] = sub_identities
        
        file.to_csv(file_path, index=False)
        print(sum([1 for identity in big_identities if len(identity)> 0 ]), len(big_identities))


def get_ME_responses() -> None:
    """Get ME responses from OpenAI and makes new csv with the results"""
    cols = ["wiki_descs", "episode-overview", "synopsis_with_character_names"]
    files = ["mid_wiki_data.csv", "short_TMDB_data.csv", "long_IMDB_data.csv"]
    for col, file in zip(cols, files):
    
        dataset = pd.read_csv(f"Data/TV Shows/{file}")
        save_path = "Data/TV Shows/" + file.split("data.csv")[0] + "with_ME.csv"

        # openAI call
        print("Starting ME call...")
        start = time.time()
        OpenAI_ME_responses = run_me_caller(dataset, col)
        dataset["OpenAI_ME_responses"] = OpenAI_ME_responses[0]
        dataset["OpenAI_ME_bool"] = conv_openAI_ME_data(OpenAI_ME_responses[0])
        dataset['OpenAI_data'] = OpenAI_ME_responses[1]
        dataset.to_csv(save_path, index=False)
        print("Elapsed time:", time.time() - start)
        
        # perspective AI call
        start = time.time()
        perspective_responses = run_Perspective_ME(dataset[col].tolist())
        dataset["perspective_ME_responses"] = perspective_responses[0]
        dataset['Perspective_data'] = perspective_responses[1]
        print("Elapsed time:", time.time() - start)
        dataset.to_csv(save_path, index=False)
        
        # Google ME call
        # dataset = pd.read_csv(save_path)
        start = time.time()
        Google_responses = run_google_ME(dataset[col].tolist())
        dataset["Google_ME_responses"] = Google_responses[0]
        dataset['Google_data'] = Google_responses[1]
        print("Elapsed time:", time.time() - start)
        dataset.to_csv(save_path, index=False)
    
        # Anthropic ME call
        start = time.time()
        Anthropic_responses = run_anthropic_ME(dataset[col].tolist())
        dataset["Anthropic_ME_responses"] = Anthropic_responses[0]
        dataset["Anthropic_ME_bool"] = Anthropic_responses[1]
        dataset['Anthropic_data'] = Anthropic_responses[2]
        print("Elapsed time:", time.time() - start)
        dataset.to_csv(save_path, index=False)
        
        # llama AI call
        # start = time.time()
        # OctoAI_responses = run_OctoAI_ME(dataset[col].tolist())
        # dataset["OctoAI_ME_responses"] = OctoAI_responses[0]
        # dataset["OctoAI_ME_bool"] = OctoAI_responses[1]
        # dataset['OctoAI_data'] = OctoAI_responses[2]
        # print("Elapsed time:", time.time() - start)
        # dataset.to_csv(save_path, index=False)
        

def make_data_subsets():
    wiki = pd.read_csv("Data/TV Shows/ShowsFull.csv")[["show_name", "show-id","episode_title", "gender-cts", 'Unnamed: 16']].dropna(subset='Unnamed: 16')
    tmdb = pd.read_csv("Data/TV Shows/ShowsFull.csv")[["show_name", "show-id","episode_title", "episode-overview", "gender-cts"]].dropna(subset='episode-overview')
    wiki.rename(columns={'Unnamed: 16': 'wiki_descs'}, inplace=True)
    wiki.to_csv('Data/TV Shows/mid_wiki_data.csv', index=False)
    tmdb.to_csv('Data/TV Shows/short_TMDB_data.csv', index=False)
    

def make_tags_general(our_tv_list:list[list], category_title:str,
               identity_title:str, 
               identities:list, 
               category_titles:list, 
               page_title_dict:dict=None,
               depths:list=None) -> None:
    
    """
    makes folders containing binned identities
    Args:
        our_tv_list (list[list]): list containing all tvs in our db
        identity_title (str): title for all sub identites
        identities (list): list of identities we want
        category_titles (list): list of wikipedia category titles
        page_title_dict (dict, optional): dict containing identities and wiki page title. Defaults to None.
        depths (list, optional): a list of how deep into each category page we want to go.
        deep refers to pages within a category page i.e. some pages have sub categories.
        go as deep as specified
    """
    
    assert len(identities) == len(category_titles)
    if depths != None:
        assert len(identities) == len(depths) 
    else:
        depths = [4 for _ in range(len(identities))]
    
    if not os.path.exists(f"./Data/TV Shows/tags/{category_title}"):
        os.makedirs(f"./Data/TV Shows/tags/{category_title}")
    if not os.path.exists(f"./Data/TV Shows/tags/{category_title}/{identity_title}"):
        os.makedirs(f"./Data/TV Shows/tags/{category_title}/{identity_title}")
        
    tv_list = [titles_recursively(category_title, depth=curr_depth) \
                        for category_title, curr_depth in zip(category_titles, depths)]
    
    if page_title_dict is not None:
        # key is the identity, value is the wiki page title
        for key, value in page_title_dict.items():
            location_of_identity = identities.index(key)
            more_tvs =  get_movie_names_list(value)
            tv_list[location_of_identity].extend(more_tvs)
        
    match_list = [get_match(our_tv_list, identity_tv_list) for identity_tv_list in tv_list]
    for identity_tv_match_list, identity in zip(match_list, identities):
        if len(identity_tv_match_list) > 0:
            tvs_with_tags = pd.DataFrame()
            tvs_with_tags[identity] = identity_tv_match_list
            tvs_with_tags.drop_duplicates(subset=[identity], inplace=True)
            
            try:
                old_file = pd.read_csv(f"./Data/TV Shows/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_tvs.csv")
                new_df = pd.concat([tvs_with_tags, old_file], ignore_index=False)
                new_df.drop_duplicates(subset=[identity], inplace=True)
                new_df.to_csv(f"./Data/TV Shows/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_tvs.csv", index=False)
                print(old_file.shape, new_df.shape)
            except Exception as e:
                print(e) # should be a directory doesn't exist error. Verify
                tvs_with_tags.to_csv(f"./Data/TV Shows/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_tvs.csv", index=False)


def get_match(our_tv_list:list, tv_list:list) -> list:
    """
    getting match between wiki film list and our list

    Args:
        our_tv_list (list): a list containing all the tvs in our db
        tv_list (list): a list of tvs to compare from wikipedia

    Returns:
        list: a list contaning tvs in both lists
    """
    match_list = []
    for tv in our_tv_list:
        if tv in tv_list \
            or f"{tv} (TV Series)" in tv_list \
            or f"{tv} (TV)" in tv_list \
            or f"{tv} episodes" in tv_list \
            or f"{tv} (American TV series)" in tv_list:
            match_list.append(tv)
    
    return match_list


def add_identity(file_paths) -> None:
    """
    Adds identity tags to the df with movies
    """
    for file_path in file_paths:
        tagged_files = os.listdir("./Data/TV Shows/tags")
        data = pd.read_csv(file_path)
        all_movies = data['show_name']
        movie_tag_dict = {}
        identity_list = []
        categories_list = []
        sub_identity_list = []
        
        for category in tagged_files:
            identitiy_bins = os.listdir(f"./Data/TV Shows/tags/{category}")
            for identity_bin in identitiy_bins:
                bin_files = os.listdir(f"./Data/TV Shows/tags/{category}/{identity_bin}")
                for file in bin_files:
                    curr_file = pd.read_csv(f"./Data/TV Shows/tags/{category}/{identity_bin}/{file}")
                    identity_name = list(curr_file.columns)[-1]
                    movies = list(curr_file[identity_name])
                    for movie in movies:
                        if movie not in movie_tag_dict.keys():
                            movie_tag_dict[movie] = [[category, identity_bin, identity_name]]
                        else:
                            movie_tag_dict[movie].append([category, identity_bin, identity_name])

        for tv in all_movies:
            if tv in movie_tag_dict.keys():
                curr_categories = []
                curr_identities = []
                curr_subids = []
                for value in movie_tag_dict[tv]:
                    curr_categories.append(value[0]) if value[0] not in curr_categories else None
                    curr_identities.append(value[1]) if value[1] not in curr_identities else None
                    curr_subids.append(value[2]) if value[2] not in curr_subids else None
                identities = ";".join(curr_identities)
                categories = ";".join(curr_categories)
                sub_ids = ";".join(curr_subids) if len(curr_subids) > 0 else ""
                identity_list.append(identities)
                categories_list.append(categories)
                sub_identity_list.append(sub_ids)
            else:
                identity_list.append("")
                categories_list.append("")
                sub_identity_list.append("")
            
        data["Identity_category"] = categories_list
        data["Big_identity"] = identity_list
        data["Sub_Identities"] = sub_identity_list
        
        data.to_csv(file_path, index=False)


def men_tags(our_shows):
    try:
        dfs = pd.read_html('https://www.reddit.com/r/MensLib/comments/eb0ir1/a_megalist_of_films_and_tv_series_showing/')
        movies = []
        for df in dfs:
            if "TV SERIES" in df.columns:
                movies.extend([movie.split("(")[0] for movie in df['TV SERIES'].tolist() if type(movie) == str])
                
        if not os.path.exists('Data/TV Shows/tags/Gender/Men/'):
            os.makedirs('Data/TV Shows/tags/Gender/Men/')
        pd.DataFrame(get_match(our_shows, movies), columns=['Men']).drop_duplicates().to_csv('Data/TV Shows/tags/Gender/Men/men_tagged_tvs.csv', index=False)
        return set(movies)
    except Exception as e:
        print(e)
        return []


def white_tagged(our_list):
    european_countries = ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
                          "Bulgaria", "Croatia", "Czech Republic", "Denmark", "Estonia", "Finland", "France",
                          "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
                          "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro",
                          "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "San Marino",
                          "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City"]

    all_shows = []
    for country in european_countries:
        all_shows.extend([show.split('(')[0] for show in titles_recursively(f'Category:Television shows set in {country}')]) 
    
    path = 'Data/TV Shows/tags/Race-Ethnicity/Non-White'
    if not os.path.exists('Data/TV Shows/tags/Race-Ethnicity/White'):
        os.makedirs('Data/TV Shows/tags/Race-Ethnicity/White')
    
    non_white = []
    for file in os.listdir(path):
        shows = pd.read_csv(f'{path}/{file}')
        non_white.extend(shows[shows.columns[0]].tolist())
    
    white_shows = [show for show in get_match(our_list, all_shows) if show not in non_white]
    pd.DataFrame(white_shows, columns=['White']).drop_duplicates().to_csv('Data/TV Shows/tags/Race-Ethnicity/White/white_tagged_tvs.csv', index=False)


def main():
    shows = pd.read_csv("Data/TV Shows/show_list.csv")['show_name'].tolist()
    
    # make_tags_general(our_tv_list=shows, category_title="Sexuality", identity_title="LGBT-related",
    #                     identities=["LGBT", 'LGBT'], 
    #                     category_titles=['Category:LGBT-related television shows', 'Category:LGBT-related television'],
    # )
    
    # make_tags_general(our_tv_list=shows,
    #                   category_title="Religion", identity_title="Christian", identities=["Christian", 'Christian', 'Christian', 'Christian'],
    #                   category_titles=["Category:Television series about Christianity", 'Category:Christian television',
    #                                    'Category:Catholic television', 'Category:Television series about nuns']
    # )
    
    # make_tags_general(our_tv_list=shows,
    #                 category_title="Religion",
    #                 identity_title="Non-Christian",
    #                 identities=["Islam", "Jewish", "Other-Non-Christian", "Jewish", 'Islam'],
    #                 category_titles=["Category:Television series about Islam","Category:Television series about Jews and Judaism",
    #                                 "Category:Television series about Buddhism", "Category:Jewish television",
    #                                 'Category:Television shows about Islam'],
    # )

    # make_tags_general(our_tv_list=shows,
    #                   category_title="Race-Ethnicity",
    #                   identity_title="Non-White",
    #                   identities=["Black", "Black", "Black", "Asian", "Native-Americans", "Native-Americans", "Latinx",
    #                               "Latinx", "Latinx", "Asian", "Asian", 'Asian',"Middle-Eastern-North-African",
    #                               'Middle-Eastern-North-African', 'Native-Americans', 'Asian', 'Asian'],
    #                   category_titles=["Category:African-American television",
    #                                    "Category:American black television series",
    #                                    "Category:2000s American black sitcoms",
    #                                    "Category:Asian-American television",
    #                                    "Category:Television shows about Native Americans",
    #                                    "Category:Native American television",
    #                                    "Category:Hispanic and Latino American sitcoms",
    #                                    "Category:Hispanic and Latino American television",
    #                                    'Category:Spanish television series',
    #                                    'Category:Chinese television series by genre',
    #                                    'Category:Chinese television shows',
    #                                    'Category:Chinese American television',
    #                                    'Category:Television series set in the Middle East',
    #                                    'Category:Arabic television series',
    #                                    'Category:Indigenous television in Canada',
    #                                    'Category:21st-century South Korean television series debuts',
    #                                    'Category:Indian English-language television shows'
    #                                    ],
    # )

    # make_tags_general(our_tv_list=shows,
    #                     category_title="Disability",
    #                     identity_title="Disability",
    #                     identities=["Physical-Disability", "Physical-Disability", "Mental-Disability", "Mental-Disability", "Mental-Disability"],
    #                     category_titles=['Category:Television shows about disability', 'Category:Obesity in television', 
    #                                      'Category:Mental disorders in television', 'Category:Down syndrome in television',
    #                                      'Category:Autism in television']
    # )

    # white_tagged(shows)
    
    # men_tags(shows)
    
    # tag women using IMDB list
    
    paths = ['Data/TV Shows/short_TMDB_with_ME.csv', 'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv',
            'Data/TV Shows/short_TMDB_data.csv', 'Data/TV Shows/mid_wiki_data.csv', 'Data/TV Shows/long_IMDB_data.csv']

    add_identity(paths)
    add_identity_using_tags(paths)
    
    files = ['long_IMDB_with_ME.csv', 'long_IMDB_with_ME.csv', 'mid_wiki_with_ME']
    identity_types = ['small', 'big']
    AIs = ['PerspectiveAI']
    # ME_score_analysis(identity_type="small", data_type="TV Shows", file="long_IMDB_with_ME.csv", ME="PerspectiveAI", ex="")
    # ME_score_analysis(identity_type="big", data_type="TV Shows", file="long_IMDB_with_ME.csv", ME="PerspectiveAI", ex="")
    
    # ME_score_analysis(identity_type="small", data_type="TV Shows", file="short_TMDB_with_ME.csv", ME="PerspectiveAI", ex='')
    # ME_score_analysis(identity_type="big", data_type="TV Shows", file="short_TMDB_with_ME.csv", ME="PerspectiveAI", ex='')
    
    # ME_score_analysis(identity_type="small", data_type="TV Shows", file="mid_wiki_with_ME.csv", ME="PerspectiveAI", ex='')
    # ME_score_analysis(identity_type="big", data_type="TV Shows", file="mid_wiki_with_ME.csv", ME="PerspectiveAI", ex='')

    
    
def imdb_age_ratings():    
    input_data = 'Data/TV Shows/mid_wiki_data.csv'
    other_data = pd.read_csv(input_data)
    other_data.drop(columns=['tags'], inplace=True)
    final = pd.read_csv('Data/TV Shows/age_ratings.csv')
    
    final['episode_title'] = final['episode_title'].apply(lambda x: x.capitalize()).tolist()
    other_data['episode_title'] =  other_data['episode_title'].apply(lambda x: x.capitalize()).tolist()
    final['show_name'] = final['show_name'].apply(lambda x: x.capitalize()).tolist()
    other_data['show_name'] =  other_data['show_name'].apply(lambda x: x.capitalize()).tolist()
    
    
    combined = pd.merge(final, other_data, how='inner', on=['show_name', 'episode_title']).drop_duplicates()
    combined.to_csv(input_data, index=False)



def toxicity_scores():
    
    paths = ['Data/TV Shows/short_TMDB_with_ME.csv', 'Data/TV Shows/mid_wiki_with_ME.csv', 'Data/TV Shows/long_IMDB_with_ME.csv',
            'Data/TV Shows/short_TMDB_data.csv', 'Data/TV Shows/mid_wiki_data.csv', 'Data/TV Shows/long_IMDB_data.csv']
    
    for path in paths:
        dataset = pd.read_csv(path)
        vals_to_change = {'G': 'TV-G', 'TV-PG': 'PG', '18+': 'TV-MA', 'R': 'TV-MA', 'TV-14': 'PG-13', '13+' : 'PG-13'}
        
        for rating in vals_to_change:
            row_index = dataset.loc[dataset['age_rating'] == rating].index

            for index in row_index:
                dataset.loc[index, 'age_rating'] = vals_to_change[rating]

        print(set(dataset['age_rating'].tolist()))
    
        ratings_set = ['TV-Y', 'TV-Y7', 'TV-Y7-FV', 'TV-G', 'PG', 'PG-13', '16+', 'M', 'TV-MA', 'R',]
        toxicity_scores = score_toxicity(dataset, ratings_set, col='age_rating')
        print(toxicity_scores)
        print(toxicity_scores.shape)
        for i in [4,5]:
            dataset[f"{ratings_set[i]} score"] = toxicity_scores[:, i]
        print("Finished appending scores")
        print(dataset.columns)
        dataset.to_csv(path, index=False)

if __name__ == "__main__":
    # make_data_subsets()
    # get_ME_responses()
    # imdb_age_ratings()
    # toxicity_scores()
    main()
    
    pass