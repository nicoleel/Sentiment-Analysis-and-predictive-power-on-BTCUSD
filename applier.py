# Save
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
import torch
from joblib import dump, load
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv


# Load
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
model = AutoModel.from_pretrained("vinai/bertweet-base").to('mps')  # You need to define the class of the model again
model.load_state_dict(torch.load('model_parameters.pth'))
model.eval()
classifier = LogisticRegression()
classifier = load('classifier.joblib')

def predict_sentiment_BERTweet(tweet):
    input = tokenizer(tweet, padding=True, truncation=True, max_length=128, return_tensors='pt').to('mps')
    with torch.no_grad():
        embedding = model(**input).last_hidden_state[:, 0, :].cpu().numpy()
    return classifier.predict(embedding)[0]

df_tweet_corpus = pd.read_csv('Data/Bitcoin_tweets_light_cleaned.csv', index_col='date', parse_dates=True)
df_tweet_corpus = df_tweet_corpus[df_tweet_corpus['user_followers'] > 500000]

tweet_corpus_text = df_tweet_corpus['text'].dropna().to_list()

results = []
chunk = 1000

def data_generator(data, chunk_size):
    for i in range(0, len(data), chunk_size):
        yield data[i:i+chunk_size]

for batch in tqdm(data_generator(tweet_corpus_text, chunk)):
    result_chunk = list(map(predict_sentiment_BERTweet,batch)) 
    results.extend(result_chunk)
    results_df = pd.DataFrame(results, columns=['sentiment'])
    results_df.to_csv('Data/Bitcoin_tweets_sentiment_500000.csv', index=False)


