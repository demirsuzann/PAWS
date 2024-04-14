import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

breeds_df = pd.read_csv("dog.csv")


features = breeds_df[['Adaptability', 'ApartmentSuitability', 'InexperiencedOwner', 'ToleranceAlone', 'Friendliness', 'Shedding', 'Health', 'Size', 'TrainAbility', 'Noisy', 'ExerciseNeeds', 'AgeAverage']]

#MODEL
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', NearestNeighbors(n_neighbors=10))
])
pipeline.fit(features)

joblib.dump(pipeline, 'knn_breed_recommender_pipeline.pkl')
