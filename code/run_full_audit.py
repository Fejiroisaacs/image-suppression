from Cjadams.adams import run_kaggle_challenge_audit
from Dixon.dixon import run_kaggle_audit
from Markov.markov import run_openAI_audit
from Movies.get_movie_data import run_movies_audit
from Stormfront.stormfront import run_stormfront_audit
from Tweets.tweets_dataset import run_tweets_audit
from TV_Shows.tv import run_tv_audit


def run_full_audit():
    run_kaggle_challenge_audit()
    run_kaggle_audit()
    run_openAI_audit()
    run_movies_audit()
    run_stormfront_audit()
    run_tweets_audit()
    run_tv_audit()
    
    
if __name__ == "__main__":
    run_full_audit()