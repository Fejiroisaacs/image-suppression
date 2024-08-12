"""
source: https://wikipedia-api.readthedocs.io/_/downloads/en/latest/pdf/
Author: Oghenefejiro Anigboro
Getting movie identity tags from wikipedia using two apis
"""

import wikipedia as wiki
import pandas as pd
import wikipediaapi as wikiapi
import os, requests
import matplotlib.pyplot as plt
    
def main() -> None: 
    data = pd.read_csv("./Data/Movies/TMDB_with_genre.csv")
    our_movie_list = [data['Title'].tolist(), data["Release year"].tolist()]
    
    
    # tag_straight(our_movie_list[0], our_movie_list[1])
    
    # make_tags_general(our_movie_list=our_movie_list,
    #                     category_title="Sexuality",
    #                     identity_title="LGBT-related",
    #                     identities=["Gay", "Lesbian", "LGBT", "Bisexual", "Trans-Nonbinary", "Bisexual", "Trans-Nonbinary"], 
    #                     category_titles=["Category:Gay-related films",
    #                                     "Category:Lesbian-related films",
    #                                     "Category:LGBT-related films",
    #                                     "Category:Bisexuality-related films",
    #                                     "Category:Transgender-related films",
    #                                     "Category:Male bisexuality in film",
    #                                     "Category:Films about trans men"
    #                                     ],
    #                     page_title_dict={"Lesbian": "List of feature films with lesbian characters",
    #                                     "Trans-Nonbinary": "List of feature films with transgender characters"},
    #                     depths=[4, 4, 4, 4, 3, 1, 1]
    # )
    
    # make_tags_general(our_movie_list,
    #                 category_title="Gender",
    #                 identity_title="Men",
    #                 identities=["Men", "Men", "Men", "Men"],
    #                 category_titles=["Category:Films about brothers", "Category:Films about kings",
    #                                  "Category:Films about father–child relationships", "Category:Films about princes"]
    # )
    
    # make_tags_general(our_movie_list,
    #                   category_title="Religion",
    #                   identity_title="Christian",
    #                   identities=["Christian"],
    #                   category_titles=["Category:Films about Christianity"],
    #                   page_title_dict={"Christian": "List of Christian films"},
    #                   depths=[2]
    # )
    # make_tags_general(our_movie_list,
    #                 category_title="Religion",
    #                 identity_title="Non-Christian",
    #                 identities=["Islam", "Jewish",
    #                             "Other-Non-Christian", "Other-Non-Christian", "Other-Non-Christian",
    #                             "Other-Non-Christian", "Other-Non-Christian", "Other-Non-Christian",
    #                             "Other-Non-Christian", "Other-Non-Christian", "Other-Non-Christian",
    #                             "Other-Non-Christian"],
    #                 category_titles=["Category:Films about Islam","Category:Films about Jews and Judaism",
    #                                 "Category:Films about Buddhism", "Category:Films about new religious movements",
    #                                 "Category:Films about Buddhism", "Category:Films about Islam", "Category:Films about Sikhism",
    #                                 "Category:Films about Jews and Judaism", "Category:Films about Satanism",
    #                                 "Category:Films about Spiritism", "Category:Films about Voodoo", "Category:Films about Zoroastrianism"],
    #                 depths=[3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4 ,4]
    # )

    # make_tags_general(our_movie_list,
    #                   category_title="Race-Ethnicity",
    #                   identity_title="Non-White",
    #                   identities=["Black", "Asian", "Native-Americans", "Latinx",
    #                               "Latinx", "Latinx", "Asian", "Asian",
    #                               "Middle-Eastern-North-African", "Middle-Eastern-North-African",
    #                               "Native-Americans", "Native-Americans", "Native-Americans", 
    #                               "Native-Americans", "Native-Americans"],
    #                   category_titles=["Category:African-American_films",
    #                                    "Category:Films about Asian Americans",
    #                                    "Category:Films about Native Americans",
    #                                    "Category:Films about Mexican Americans",
    #                                    "Category:Hispanic and Latino American films",
    #                                    "Category:Mexican films",
    #                                    "Category:Chinese films","Category:Asian films",
    #                                    "Category:Middle East in fiction",
    #                                    "Category:Films set in the Middle East",
    #                                    "Category:Native American cinema","Category:Inuit films", 
    #                                    "Category:Animated films about Native Americans", 
    #                                    "Category:Films set in the Inca Empire", 
    #                                    "Category:Films set in the Aztec Triple Alliance"],
    #                   page_title_dict={"Latinx": "List of Chicano films", "Native-Americans": "List of Indigenous Canadian films"}
    # )

    # make_tags_general(our_movie_list,
    #                     category_title="Disability",
    #                     identity_title="Disability",
    #                     identities=["Physical-Disability", "Physical-Disability", "Physical-Disability",
    #                                 "Physical-Disability", "Physical-Disability", "Physical-Disability","Physical-Disability", 
    #                                 "Mental-Disability", "Mental-Disability", "Mental-Disability", "Mental-Disability"],
    #                     category_titles=["Category:Films about parasports", "Category:Films about amputees", 
    #                                     "Category:Films about blind people", "Category:Films about people with cerebral palsy",
    #                                     "Category:Films about deaf people", "Category:Films about people with paraplegia or tetraplegia",
    #                                     "Category:Films about people with dwarfism", "Category:Films about autism", "Category:Films about intellectual disability",
    #                                     "Category:Films about mental disorders", "Category:Films about mental health"]
    # )

    white_tagged(our_list=our_movie_list[0], year_list=our_movie_list[1])


def white_tagged(our_list, year_list):
    european_countries = ["Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina",
                          "Bulgaria", "Croatia", "Czech Republic", "Denmark", "Estonia", "Finland", "France",
                          "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
                          "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro",
                          "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "San Marino",
                          "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City"]

    all_shows = []
    for country in european_countries:
        all_shows.extend([show.split('(')[0] for show in titles_recursively(f'Category:Films set in {country} by city')]) 
        all_shows.extend([show.split('(')[0] for show in titles_recursively(f'Category:Animated films set in {country}')]) 
        all_shows.extend([show.split('(')[0] for show in titles_recursively(f'Category:Documentary films about {country}')]) 
        all_shows.extend([show.split('(')[0] for show in titles_recursively(f'Category:Films set in {country}')]) 
    print(len(all_shows))
    
    path = 'Data/Movies/tags/Race-Ethnicity/Non-White'
    if not os.path.exists('Data/Movies/tags/Race-Ethnicity/White'):
        os.makedirs('Data/Movies/tags/Race-Ethnicity/White')
    
    non_white = []
    for file in os.listdir(path):
        shows = pd.read_csv(f'{path}/{file}')
        non_white.extend(shows[shows.columns[0]].tolist())
    
    all_shows = [show for show in all_shows if show not in non_white]
    shows, years = get_match(our_list, all_shows, year_list)
    white_tagged_df = pd.DataFrame(columns=['White', 'Release year'])
    white_tagged_df['White'] = shows
    white_tagged_df['Release year'] = years
    white_tagged_df.drop_duplicates().to_csv('Data/Movies/tags/Race-Ethnicity/White/white_tagged_movies.csv', index=False)


def more_mens_movies():
    try:
        dfs = pd.read_html('https://www.reddit.com/r/MensLib/comments/eb0ir1/a_megalist_of_films_and_tv_series_showing/')
        movies = []
        for df in dfs:
            if "FILM NAME (WITH IMDB LINK)" in df.columns:
                movies.extend([movie.split("(")[0] for movie in df['FILM NAME (WITH IMDB LINK)'].tolist() if type(movie) == str])
        return movies
    except Exception as e:
        print(e)
        return []


def bechdeltest():
    bechdel_movies = requests.get('https://bechdeltest.com/api/v1/getMoviesByTitle').json()
    movie_list = []
    match_list = []
    for movie_dict in bechdel_movies:
        movie_list.append([movie_dict['title'], movie_dict['year']])
    
    our_file = pd.read_csv("Data/Movies/TMDB_with_genre.csv")[["Title", 'Release year']]
    for movie, year in our_file.to_numpy():
        if [movie, year] in movie_list:
            match_list.append([movie, year])
            
    pd.DataFrame(match_list, columns=['Women', 'Release year']).to_csv('Data/Movies/tags/Gender/Women/women_tagged_movies.csv', index=False)
    

def tag_straight(our_movie_list, movie_years):
    movies = titles_recursively('Category:American romantic comedy films', depth=4)
    path = 'Data/Movies/tags/Sexuality/LGBT-related'
    all_lgbt_tagged = []
    files = os.listdir(path)
    for file in files:
        data = pd.read_csv(f'{path}/{file}')
        all_lgbt_tagged.extend(data[data.columns[0]])
    
    straight_movies = [movie for movie in movies if movie not in all_lgbt_tagged]
    matches = get_match(our_movie_list, straight_movies, movie_years)
    
    if not os.path.exists(f"./Data/Movies/tags/Sexuality/Straight"):
        os.makedirs(f"./Data/Movies/tags/Sexuality/Straight")
    pd.DataFrame([[movie, year] for movie, year in zip(matches[0], matches[1])], columns=["Straight", "Release year"]).drop_duplicates().\
        to_csv("Data/Movies/tags/Sexuality/Straight/straight_tagged_movies.csv", index=False)


def make_tags_general(our_movie_list:list[list], category_title:str,
               identity_title:str, 
               identities:list, 
               category_titles:list, 
               page_title_dict:dict=None,
               depths:list=None) -> None:
    
    """
    makes folders containing binned identities
    Args:
        our_movie_list (list[list]): list containing all movies in our db
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
    
    if not os.path.exists(f"./Data/Movies/tags/{category_title}"):
        os.makedirs(f"./Data/Movies/tags/{category_title}")
    if not os.path.exists(f"./Data/Movies/tags/{category_title}/{identity_title}"):
        os.makedirs(f"./Data/Movies/tags/{category_title}/{identity_title}")
        
    movie_list = [titles_recursively(category_title, depth=curr_depth) \
                        for category_title, curr_depth in zip(category_titles, depths)]
    
    if identity_title == "Men":
        movie_list.extend(more_mens_movies())
        
    if page_title_dict is not None:
        # key is the identity, value is the wiki page title
        for key, value in page_title_dict.items():
            location_of_identity = identities.index(key)
            more_movies =  get_movie_names_list(value)
            movie_list[location_of_identity].extend(more_movies)
        
    match_list = [get_match(our_movie_list[0], identity_movie_list, our_movie_list[1]) \
                                                            for identity_movie_list in movie_list]
    
    for identity_movie_match_list, identity in zip(match_list, identities):
        if len(identity_movie_match_list) > 0:
            movies_with_tags = pd.DataFrame()
            movies_with_tags[identity] = identity_movie_match_list[0]
            movies_with_tags["Release year"] = identity_movie_match_list[1]
            movies_with_tags.drop_duplicates(subset=[identity, "Release year"], inplace=True)
            
            try:
                old_file = pd.read_csv(f"./Data/Movies/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_movies.csv")
                new_df = pd.concat([movies_with_tags, old_file], ignore_index=False)
                new_df.drop_duplicates(subset=[identity, "Release year"], inplace=True)
                new_df.to_csv(f"./Data/Movies/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_movies.csv", index=False)
                print(old_file.shape, new_df.shape)
            except Exception as e:
                print(e) # should be a directory doesn't exist error. Verify
                movies_with_tags.to_csv(f"./Data/Movies/tags/{category_title}/{identity_title}/{identity.lower()}_tagged_movies.csv", index=False)


def get_movie_names_list(page_title:str) -> list:
    """
    gets movies list from pages with tables

    Args:
        page_title (str): name of page

    Returns:
        list: all movies found
    """
    try:
        movie_list = []
        page = wiki.page(page_title, auto_suggest=False).html().encode("UTF-8")
        df = pd.read_html(page)
        for frame in df:
            try:
                titles = frame["Title"]
                for title in list(set(titles)):
                    movie_list.append(title)
            except KeyError as ke:
                pass
            try:
                films = frame["Film"]
                for film in list(set(films)):
                    movie_list.append(film)
            except KeyError as ke2:
                pass
    except ValueError as ve:
        print("Error:", ve)
        
    return movie_list if len(movie_list) > 0 else movies_from_wiki_list(page_title)


def movies_from_wiki_list(page_title:str) -> list:
    """
    get movie list from wikipedia when they are in lists
    Args:
        page_title (str): title of wiki page

    Returns:
        list: list of movies found
    """
    try:
        wiki_wiki = wikiapi.Wikipedia('Benchmark_tagger', 'en')
        page = wiki_wiki.page(page_title)
        films = page.section_by_title("Films").text
        movies = films.split("\n")
        return movies
    except Exception as e:
        print("Error", e)
        return []


def get_match(our_movie_list:list, movie_list:list, release_year:list=None) -> list:
    """
    getting match between wiki film list and our list

    Args:
        our_movie_list (list): a list containing all the movies in our db
        movie_list (list): a list of movies to compare from wikipedia
        release_year (list): a list of movie release years

    Returns:
        list: a list contaning movies in both lists
    """
    match_list = []
    year_list = []
    release_year = ["" for _ in range(len(our_movie_list))] if release_year == None else release_year
    for i in range(len(our_movie_list)):
        movie = our_movie_list[i]
        if movie in movie_list \
            or f"{movie} ({release_year[i]} film)" in movie_list \
            or f"{movie} (film)" in movie_list \
            or f"{movie} (American film)" in movie_list\
            or f"{movie} ({release_year[i]} American film)" in movie_list\
            or f"{movie} ({release_year[i]})" in movie_list:
            match_list.append(movie)
            year_list.append(release_year[i])
    
    return match_list, year_list


def titles_recursively(category, depth:int=4) -> list:
    """
    helper function to get movie titles recursively

    Args:
        category (wiki object)
        depth (int, optional): depth of category search. Defaults to 4. seems to be the most "resonable" for all search

    Returns:
        list: all movie title found from recursive search
    """
    wiki_wiki = wikiapi.Wikipedia(user_agent='Benchmark_tagger', language='en', \
        extract_format=wikiapi.ExtractFormat.WIKI)
    page = wiki_wiki.page(category)
    movies = get_titles_recursively(page.categorymembers, max_level=depth)
    
    return movies
    
    
def get_titles_recursively(categorymembers, level=0, max_level=4) -> list:
    """
    gets the titles of movies from wiki pages with categories and subcategories
    returns a list of all movie titles found
    """
    titles = []
    for c in categorymembers.values():
        if "Category:" not in c.title:
            title = c.title
            titles.append(title)
        else:
            print(c.title)
        if c.ns == wikiapi.Namespace.CATEGORY and level < max_level:
            titles.extend(get_titles_recursively(c.categorymembers, \
                level=level + 1, max_level=max_level))
            
    return titles


def add_identity_tags() -> None:
    """
    Adds identity tags to the df with movies
    """
    
    tagged_files = os.listdir("./Data/Movies/tags")
    data = pd.read_csv("./Data/Movies/full_TMDB_wiki_with_scores.csv")
    all_movies = data['Title']
    movie_release_year = data["Release year"]
    movie_tag_dict = {}
    identity_list = []
    categories_list = []
    sub_identity_list = []
    
    for category in tagged_files:
        identitiy_bins = os.listdir(f"./Data/Movies/tags/{category}")
        for identity_bin in identitiy_bins:
            bin_files = os.listdir(f"./Data/Movies/tags/{category}/{identity_bin}")
            for file in bin_files:
                curr_file = pd.read_csv(f"./Data/Movies/tags/{category}/{identity_bin}/{file}")
                identity_name = list(curr_file.columns)[-2]
                identity_release_year = curr_file["Release year"]
                movies = list(curr_file[identity_name])
                for i in range(len(movies)):
                    if movies[i] not in movie_tag_dict.keys():
                        movie_tag_dict[movies[i]] = {}
                        movie_tag_dict[movies[i]][identity_release_year[i]] = [[category, identity_bin, identity_name]]
                    else:
                        if identity_release_year[i] not in movie_tag_dict[movies[i]].keys():
                            movie_tag_dict[movies[i]][identity_release_year[i]] = [[category, identity_bin, identity_name]]
                        else:
                            movie_tag_dict[movies[i]][identity_release_year[i]].append([category, identity_bin, identity_name])


    for i in range(len(all_movies)):
        if all_movies[i] in movie_tag_dict.keys():
            if movie_release_year[i] in movie_tag_dict[all_movies[i]].keys():
                curr_categories = []
                curr_identities = []
                curr_subids = []
                for value in movie_tag_dict[all_movies[i]][movie_release_year[i]]:
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
        else:
            identity_list.append("")
            categories_list.append("")
            sub_identity_list.append("")
        
    data["Identity_category"] = categories_list
    data["Big_identity"] = identity_list
    data["Sub_Identities"] = sub_identity_list
    
    data.to_csv("./Data/Movies/TMDB_plots_scores_identity.csv", index=False)
    

def read_me_table() -> None:
    """
    Generates bar chart with proportions of identites for movies found
    """
    tagged_files = os.listdir("./Data/Movies/tags")
    identities = []
    count = []
    categories = {}
    for category in tagged_files:
        identitiy_bins = os.listdir(f"./Data/Movies/tags/{category}")
        category_count = 0
        for identity_bin in identitiy_bins:
            bin_files = os.listdir(f"./Data/Movies/tags/{category}/{identity_bin}")
            for file in bin_files:
                curr_file = pd.read_csv(f"./Data/Movies/tags/{category}/{identity_bin}/{file}")
                category_count += curr_file.shape[0]
                if file != 'lgbt_tagged_movies.csv' and file != 'non-christian_tagged_movies.csv':
                    identities.append(list(curr_file.columns)[-2])
                    count.append(curr_file.shape[0])
        categories[category] = category_count
                

    plt.figure(figsize = (15, 10))
    plt.bar(identities, count, color ='navy')
    addlabels(identities, count)
    plt.xticks(rotation=90)
    plt.ylabel("Number of Movies")
    plt.xlabel("Identity")
    plt.title("Movies per identity")
    plt.savefig("./plots/Movies/identities_split.png", format="png", bbox_inches='tight', pad_inches=0.5)

    plt.clf()
    plt.bar(list(categories.keys()), list(categories.values()), color ='navy')
    addlabels(list(categories.keys()), list(categories.values()))
    plt.ylabel("Number of Movies")
    plt.xlabel("Binned Category")
    plt.title("Movies per binned identity")
    plt.savefig("./plots/Movies/identities_binned_split.png", format="png", bbox_inches='tight', pad_inches=0.5)


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center', fontsize=14)

if __name__ == "__main__":
    # main()
    # bechdeltest()
    # add_identity_tags()
    # read_me_table()
    # more_mens_movies()
    pass