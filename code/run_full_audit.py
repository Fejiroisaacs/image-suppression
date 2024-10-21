from Cjadams.adams import run_kaggle_challenge_audit
from Dixon.dixon import run_kaggle_audit
from Markov.markov import run_openAI_audit
from Movies.get_movie_data import run_movies_audit
from Stormfront.stormfront import run_stormfront_audit
from Tweets.tweets_dataset import run_tweets_audit
from TV_Shows.tv import run_tv_audit
import matplotlib.pyplot as plt
import pandas as pd
from main import make_bounds, add_normalization, identity_imedians, identity_ifprs, \
    comparison_chart_openAI, comparison_chart_other, dataset_table, add_metrics, make_ME_fpr_charts\
        , make_ME_med_charts, get_overal_FPR, combine_data, make_lw_data, add_has_slurs


def run_full_audit():
    run_kaggle_challenge_audit()
    run_kaggle_audit()
    run_openAI_audit()
    run_movies_audit()
    run_stormfront_audit()
    run_tweets_audit()
    run_tv_audit()


def analysis():
    add_normalization()
    add_has_slurs()
    combine_data()
    make_bounds()
    comparison_chart_openAI()
    comparison_chart_other("PerspectiveAI")
    comparison_chart_other("Google")
    dataset_table(append=False) 

    print()
    print(pd.read_csv('./Data/OpenAI_flagging_bounds.csv').to_latex(index=False), '\n')
    print(pd.read_csv('Data/iavg_table.csv').fillna('').astype('str').to_latex(index=False), '\n')
    print(pd.read_csv('Data/ifpr_table.csv').fillna('').astype('str').to_latex(index=False))
    print(pd.read_csv('Data/imed_table.csv').fillna('').astype('str').to_latex(index=False))
    print(pd.read_csv('Data/ifpr_table_sub_identities.csv').fillna('').astype('str').to_latex(index=False))
    print(pd.read_csv('Data/imed_table_sub_identities.csv').fillna('').astype('str').to_latex(index=False))
    gendata = pd.read_csv('Data/GenAI_sub_ids.csv').astype('str')
    print(gendata[[col for col in gendata.columns if 'CI' not in col]].to_latex(index=False))
    trad_data = pd.read_csv('Data/Traditional_sub_ids.csv').astype('str')
    print(trad_data[[col for col in trad_data.columns if 'CI' not in col]].to_latex(index=False))
    
    identity_imedians('Data/Combined/genAI_combined.csv', label_cols=['true_label'], dataset_name='GenAI', score_types=['True label'], other_table=True)
    identity_ifprs('Data/Combined/genAI_combined.csv', label_cols=['true_label'], dataset_name='GenAI', score_types=['True label'], other_table=True)
    identity_imedians('Data/Combined/traditional_combined.csv', label_cols=['true_label'], dataset_name='Traditional', score_types=['True label'], other_table=True)
    identity_ifprs('Data/Combined/traditional_combined.csv', label_cols=['true_label'], dataset_name='Traditional', score_types=['True label'], other_table=True)


    add_metrics(identity_ifprs)
    add_metrics(identity_imedians)
    
    plt.rcParams.update({'font.size': 48})
    make_ME_med_charts()
    make_ME_fpr_charts()
    get_overal_FPR()
    make_lw_data()
    
if __name__ == "__main__":
    run_full_audit()
    analysis()