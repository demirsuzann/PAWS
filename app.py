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


st.title(":dog: :rainbow[Köpek İyi Bir Arkadaştır!] :dog:")
main_tab,  recommendation_tab = st.tabs(["Home Page",  "Get Recommendation"])
df = get_data()

# Ana Sayfa

left_col, right_col = main_tab.columns(2)

left_col.write("""Köpek sahiplenmek, yaşamınıza neşe ve sevgi getirecek harika bir karardır. Ancak, hangi köpek ırkının sizin yaşam tarzınıza en uygun olduğunu seçmek önemlidir. İşte sizin için geliştirdiğimiz öneri sistemi, doğru köpek ırkını bulmanıza yardımcı olacak!
\nAdaptasyon Yeteneği: Köpeğinizin yeni ortamlara ne kadar kolay uyum sağladığı önemlidir. Siz değişiklikleri seviyor musunuz yoksa rutininizden mi hoşlanıyorsunuz?
\nYalnızlık Toleransı: Köpeğiniz yalnız kaldığında nasıl davranır? Çoğu zaman mı evde yalnız mı kalıyorsunuz?
\nArkadaş Canlısı Davranış: Evinizde çocuklar veya diğer hayvanlar var mı? Köpeğinizin başkalarıyla iyi geçinmesini mi istiyorsunuz?
\nSağlık Durumu: Bakım gerektiren köpeklerden mi hoşlanıyorsunuz yoksa daha az bakım isteyenler mi?
\nZeka Seviyesi: Eğitilebilir bir köpek mi istiyorsunuz yoksa daha bağımsız bir karaktere sahip olan mı?
\nEgzersiz İhtiyacı: Aktif bir yaşam tarzınız mı var yoksa daha sakin bir ortam mı tercih ediyorsunuz?
\nSıcaklık Toleransı: İklim şartlarına ne kadar dayanıklı olması önemli mi?
\nYaş Ortalaması: Köpeğinizin uzun ömürlü olmasını mı istiyorsunuz?
\nApartman Uygunluğu: Küçük bir dairede mi yaşıyorsunuz yoksa geniş bir bahçeniz mi var?
\nKüçük, orta veya büyük boyutta bir köpek mi istiyorsunuz?
\nBu özellikler, sizin için en uygun köpek ırkını belirlemekte yardımcı olacaktır. Öneri sistemimiz, tercihlerinize en iyi şekilde uyacak köpekleri bulmanıza yardımcı olacak ve yeni bir dost kazanmanın ne kadar harika olduğunu gösterecektir!""")

right_col.image("dogs.png", width=600)

pipeline = get_pipeline()


col_features1, col_features2, col_recommendation = recommendation_tab.columns(3)

# Kullanıcı girdilerini al
Adaptability = col_features1.slider("Adaptability", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
ToleranceAlone = col_features1.slider("Tolerance Alone", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
Friendliness = col_features1.slider("Friendliness", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
Health = col_features1.slider("Health",  min_value=1.0, max_value=5.0, step=0.5, value=3.0)
Intelligence = col_features1.slider("Intelligence", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
ExerciseNeeds = col_features1.slider("Exercise Needs", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
TemperatureTolerance = col_features1.slider("Temperature Tolerance", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
AgeAverage = col_features1.slider("Age Average", min_value=7.0, max_value=21.0, step=0.5, value=14.0)
ApartmentSuitability = col_features1.slider("Apartment Suitability", min_value=1.0, max_value=5.0, step=0.5, value=3.0)
Size = col_features1.slider("Size", min_value=1.0, max_value=5.0, step=0.5, value=3.0)

features = np.array([Adaptability , ToleranceAlone, Friendliness, Health, Intelligence , ExerciseNeeds, TemperatureTolerance, AgeAverage,ApartmentSuitability,Size]).reshape(1, -1)

if col_features2.button("Öneri Getir!"):

    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features), n_neighbors=10)

    for i in range(0, 10):
      recommended_index = indices[0][i]
      recommended_dog = df.iloc[recommended_index]
      recommended_distances = distances[0][i]

      col_recommendation.image(recommended_dog['Images'])
      col_recommendation.write(f"**{recommended_dog['breed']}**")
      col_recommendation.write(f"**{recommended_dog['url']}**")
      col_recommendation.write(f"**{recommended_distances}**")
