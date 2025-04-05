import pandas as pd
import numpy as np 
import pickle


df = pd.read_csv('data/spotify.csv')


df['explicit'] = df['explicit'].astype(int)

df['energy_danceability_ratio'] = df['energy'] / df['danceability']
df['loudness_energy_ratio'] = df['loudness'] / (df['energy'])  


df.to_csv('data/save_data/spotify_processed.csv', index=False)
