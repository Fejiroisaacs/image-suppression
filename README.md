# Data and code for the paper

## Identity-related Speech Suppression in Generative AI Content Moderation

## SCHEMA

| Column Name | Source   | Explanation |
|---|---|---|
| index       | -   | \[0..543858\] Number of items in dataset     |
| dataset_name   | -    | Name of the dataset where row is from       |
| subset_name     | -    | Subset of original data  |
| data_type    | -    | GenAI or Traditional       |
| text    | -    | Dataset text/value   |
| word_length       | -    | Number of words in text  |
| true_label       | -    |  Overall toxicity label |
| Big_identity       | -    | Semi-column separated list of texts big (grouped) identity  |
| Sub_Identities       | -    | Semi-column separated list of texts sub (in group) identity     |
| has_slur       | -   | If text contains slurs from our slurs list   |
| PG score       | Movies/TV Shows    | PG appropriate boolean |
| PG-13 score       | Movies/TV Shows   | PG-13 appropriate boolean  |
| S     | OpenAI   | Subset of OpenAI data with sexual label  |
| H       | OpenAI    | Subset of OpenAI data with hate label |
| V       | OpenAI    | Subset of OpenAI data with violence label  |
| HR     | OpenAI    | Subset of OpenAI data with harassment label  |
| SH       | OpenAI    | Subset of OpenAI data with self-harm label  |
| S3   | OpenAI   | Subset of OpenAI data with sexual/minors label |
| H2    | OpenAI    | Subset of OpenAI data with hate/threatening label  |
| V2      | OpenAI   | Subset of OpenAI data with violence/graphic label  |
| Hate     | TweetEval    | Subset of TweetEval data with hate label  |
| Offensive     | TweetEval    | Subset of TweetEval data with offensive label  |
| severe_toxicity | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with severe toxicity label |
| obscene | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with sexual explicit label |
| sexual_explicit | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with - label  |
| identity_attack | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with identity attack label  |
| insult | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with insult label  |
| threat | Jigsaw Kaggle | Subset of Jigsaw Kaggle data with threat label  |
| jigsaw_score | ME API - `v1alpha1` | The toxicity score of returned by Jigsaw |
| openAI_flag | ME API - `text-moderation-007` | Boolean value of flag by OpenAI |
| openAI_score | ME API - `text-moderation-007` | The max normalized score from OpenAI |
| anthropic_flag       | ME API - `claude-3-haiku-20240307`   | Boolean of anthropic response  |
| google_score      | ME API - Unknown Model   | The max score from google response |
| llama_guard_flag     | ME API - `llamaguard-2-8b`  | Boolean of llama response  |
| OpenAI_data     | - | Tuple containing date of run and OpenAI model name  |
| Anthropic_data | - | Tuple containing date of run and Anthropic model name |
| OctoAI_data | - | Tuple containing date of run and llama model name |
| Jigsaw_data | - | Tuple containing date of run and jigsaw model name |
| Google_data | - | Tuple containing date of run and google model name |

## Data subsets

### Jigsaw Kaggle

Jigsaw Unintended Bias in Toxicity Classification data with a total of 445,293 unique identity tagged elements. We excluded rows with no identity tags. Rows without values in any of these columns -- male, female, transgender, other_gender, heterosexual, homosexual_gay_or_lesbian, bisexual, other_sexual_orientation, christian, jewish, muslim, hindu, buddhist, atheist, other_religion, black, white, asian, latino, other_race_or_ethnicity, physical_disability, intellectual_or_learning_disability, psychiatric_or_mental_illness, other_disability. Further information about data collection, anonymization, and preparation is discussed in [the kaggle page](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview)

### Jigsaw Bias

Data from Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification paper. This subset of the dataset we used contains 60,560 rows. We extrated rows with phrases containing the identities we focussed on in our research. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://arxiv.org/pdf/1903.04561)

### Stormfront

Data (10,944 entries) from the Hate speech dataset from a white supremacist forum. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://aclanthology.org/W18-51.pdf)

### TweetEval

We used the hate (12,962) and offensive (14,100) subset of the data from the Unified Benchmark and Comparative Evaluation for Tweet Classification paper. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://arxiv.org/pdf/2010.12421)

### OpenAI

Complete data (1,680 entries) from the 'A Holistic Approach to Undesired Content Detection in the Real World' paper. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://arxiv.org/abs/2208.03274)

### Movies

Shows selected from TMDB's [Top 10000 Movies](https://www.themoviedb.org/movie?language=en-US) list. This dataset contains 6,001 movies released in US with original language in English. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://)

### TV Synopsis

Shows selected from TMDB's [Top 10000 Popular TV Shows](https://www.themoviedb.org/tv?language=en-US) list. This dataset contains each episode in S1 of each of these shows, with a total of 14,760 unique episodes. Further information about data collection, anonymization, and preparation is discussed in [the associated paper](https://)

## FILESTRUCTURE

Dataset (compressed): `speech_suppression_database.7z`
