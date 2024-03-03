import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

#read the data
df = pd.read_csv("data/Spotify_final_dataset.csv", low_memory=False)
# remove duplicates
df = df.drop_duplicates(subset="Song Name")
