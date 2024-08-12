""""
Description: A program to use the wikipedia API to gather TV and movie data
Author:
Date:
Created for use in the GPT benchmark suite, more information here: 
"""

# Imports:
import wikipedia as wiki
import pandas as pd
import tmdbsimple as tmdb
import time, datetime


tmdb.API_KEY = 'ENTER API KEY'

def main():
    get_TMDB_shows()
    make_episode_db()
    pass

def conv_ids(id_list):
    id_dict = {
            10759: "Action & Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            10762: "Kids",
            9648: "Mystery",
            10763: "News",
            10764: "Reality",
            10765: "Sci-Fi & Fantasy",
            10766: "Soap",
            10767: "Talk",
            10768: "War & Politics",
            37: "Western",
            10402: "Music",
            36: "History",
            10749: "Romance"
    }
    
    return [id_dict[id] for id in id_list if id in id_dict.keys()]


def get_TMDB_shows():
    top_tv = []
    i = 1
    discover = tmdb.Discover()
    while i < 501:
        try:
            response = discover.tv(language='en-US', sort_by='popularity.desc', page=i, region='US')
            for result in response['results']:
                if result['original_language'] == 'en':
                    top_tv.append([result['original_name'], result['id'], result['first_air_date'][:4], conv_ids(result['genre_ids'])])
            i += 1
        except Exception as e:
            print(e) # should never print
            i += 1 
            pass
        
    df = pd.DataFrame(top_tv, columns=["show_name", 'show-id', 'release', 'genres'])
    df.to_csv("Data/TV Shows/show_list.csv", index=False)
    

def ct_gender(crew_list):
    gender_dict = {
        0 : "Not specified",
        1 : "Female",
        2 : "Male",
        3 : "Non-Binary"
    }
    ct_dict = {}
    
    for crew in crew_list:
        if gender_dict[crew['gender']] in ct_dict.keys():
            ct_dict[gender_dict[crew['gender']]] += 1
        else:
            ct_dict[gender_dict[crew['gender']]] = 1
            
    return ct_dict

    
def make_episode_db():
    episodes_df = pd.DataFrame(columns=["show_name","show-id", "episode_title", "episode-overview", "gender-cts"])
    shows_df = pd.read_csv("Data/TV Shows/show_list.csv")
    show_obj = tmdb.tv.TV_Seasons
    
    for name, id in zip(shows_df["show_name"], shows_df["show-id"]):
        try:
            show = show_obj(id, season_number=1).info(append_to_response="season 1")['episodes']
            try:
                for episode in show:
                    episodes_df.loc[len(episodes_df.index)] = [name, id, episode['name'], episode['overview'], ct_gender(episode['crew'])]
            except Exception as e:
                print(e)
                pass
        except Exception as e:
            print(e) 
            pass
                
    episodes_df.to_csv("Data/TV Shows/episodes_db.csv")
    

def get_age_ratings():
    file = pd.read_csv("Data/TV Shows/show_list.csv")
    age_ratings = []
    ct = 0
    for show_id in file['show-id'].tolist():
        try:
            ct+=1
            tv = tmdb.TV(show_id)
            sub_ratings = []
            for cr in tv.content_ratings()['results']:
                if cr['iso_3166_1'] == 'US':
                    sub_ratings.append(cr['rating'])
            age_ratings.append(sub_ratings)
        except:
            age_ratings.append([])
    file['age_ratings'] = age_ratings
    file.to_csv("Data/TV Shows/show_list_ratings.csv", index=False)

get_age_ratings() 

# main()