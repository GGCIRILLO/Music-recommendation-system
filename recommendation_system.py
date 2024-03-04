import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import webbrowser

# read the data
df = pd.read_csv("data/Spotify_final_dataset.csv", low_memory=False)
# keep only unique songs
df = df.drop_duplicates(subset="Song Name")
# remove rows that contain missing (NaN) values
df = df.dropna(axis=0)
# We need Position,Artist Name,Song Name
df = df.drop(df.columns[3:], axis=1)
#Removing spaces from the “Artist Name” column  
# Because if there is a space between the names, CountVectorizer will count that single name as one word.
df["Artist Name"] = df["Artist Name"].str.replace(" ", "")
# concatenates the values in each row into a single string in a new column named "data." 
# (.apply is used to convert each row into a single string)
df["data"] = df.apply(lambda value: " ".join(value.astype("str")), axis=1)

# models
# convert the "data" column into a matrix of token counts.
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
# calculates the cosine similarity between the vectors. 
# the result is a similarity matrix where each entry (i, j) represents the similarity between the ith and jth songs.
similarities = cosine_similarity(vectorized)

# Assign the new dataframe with `similarities` values.
# build a matrix where the rows and columns are the song names and the values are the cosine similarities.
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()

system = True
while system:
    while True:
        song = input("Enter the song name: ")
        
        if song in df_tmp.columns:
            #create a new dataframe with the 15 most similar songs
            recommendation = df_tmp.nlargest(16, song)["Song Name"]
            break
        else:
            print("Song not found, please try again")
            
    print("You should check out these songs: \n")
    for i, song in enumerate(recommendation.values[1:]):
        print(f"{i+1}. {song}")

    while True:
        next_command = input("\nDo you want to listen to any song? [yes, no]: ")

        if next_command == "yes":
            try:
                song_number = int(input("Enter the number of the song you want to listen to: "))
                if 1 <= song_number <= len(recommendation) - 1:
                    song_to_listen = recommendation.values[song_number]
                    webbrowser.open(f'https://open.spotify.com/search/{song_to_listen.replace(" ", "%20")}')
                else:
                    print("Invalid song number. Please enter a number within the range.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        elif next_command == "no":
            break
        else:
            print("Please type 'yes' or 'no'")

while True:
    next_command = input("Do you want to generate again for the next song? [yes, no] ")

    if next_command == "yes":
        break
    elif next_command == "no":
        system = False
        break
    else:
        print("Please type 'yes' or 'no'")