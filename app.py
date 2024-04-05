import streamlit as st
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np 
import pandas as pd 


st.set_page_config(layout = "wide", page_title="Choose your future friend", page_icon=":service_dog:")

@st.cache_data
def get_data():
    dataframe = pd.read_csv('breeds.csv')
    return dataframe

 
@st.cache_data
def get_pipeline():
    pipeline = joblib.load('knn_breed_recommender_pipeline.pkl')
    return pipeline


st.title(":dog: :rainbow[My Dog Friend] :dog:")
main_tab,  recommendation_tab = st.tabs(["Home Page",  "Recommandation System"])
df = get_data()

# Ana Sayfa

left_col, right_col = main_tab.columns(2)

left_col.write("""Deciding to welcome a dog into your life is an immensely rewarding decision, but it also comes with significant responsibilities. With hundreds of dog breeds, each possessing its own distinct appearance and characteristics, selecting the ideal one can prove to be a daunting task. Even for those who have a particular breed in mind, factors such as availability or discovering a better-suited alternative can complicate the decision-making process. To address this challenge, I've developed a comprehensive dog recommendation system.
\nThis system streamlines the process of finding the perfect canine companion by employing either collaborative filtering or content-based filtering methodologies. Collaborative filtering enables users to discover the most popular dog breeds, drawing insights from collective preferences. On the other hand, content-based filtering facilitates the identification of breeds similar to a selected dog breed, leveraging specific traits and attributes.
\nWhether users seek broad recommendations based on popular choices or targeted suggestions aligning with their preferences, this recommendation system caters to diverse needs. It serves as a valuable resource for individuals exploring dog ownership, particularly those lacking in-depth knowledge about different breeds or harboring specific preferences regarding canine traits.""")

right_col.image("dogs.jpeg", width=600)

pipeline = get_pipeline()


col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)

# Kullanıcı girdilerini al
Adaptability = col_features1.slider("Adaptability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
ToleranceAlone = col_features1.slider("Tolerance Alone", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
Friendliness = col_features1.slider("Friendliness", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
Health = col_features1.slider("Health",  min_value=0.0, max_value=1.0, step=0.1, value=0.5)
Intelligence = col_features1.slider("Intelligence", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
ExerciseNeeds = col_features1.slider("Exercise Needs", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
TemperatureTolerance = col_features1.slider("Temperature Tolerance", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
AgeAverage = col_features1.slider("Age Average", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
MaximumLifespan = col_features1.slider("Maximum Lifespan", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
ApartmentSuitability = col_features1.slider("Apartment Suitability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
Size = col_features1.slider("Size", min_value=0.0, max_value=1.0, step=0.1, value=0.5)

features = np.array([Adaptability , ToleranceAlone, Friendliness, Health, Intelligence , ExerciseNeeds, TemperatureTolerance, AgeAverage,ApartmentSuitability,Size]).reshape(1, -1)

if col_features2.button("Öneri Getir!"):

    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features), n_neighbors=10)

    recommended_index = indices[0][1]
    recommended_dog = df.iloc[recommended_index]

    col_recommendation.image(recommended_dog['url'])
    col_recommendation.write(f"**{recommended_dog['breed']}**")
