import math
import streamlit as st
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

page_by_image = """
<br />
<style>
[data-testid="stAppViewContainer"] {
background-image:  url("https://i.pinimg.com/originals/bf/ad/eb/bfadeb0e8849a04a7deae2c1a509ee3f.jpg");
background-size: cover;
background-color: rgba(255, 255, 255, 0.2);
}
[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
[data-testid="stToolbar"] {
right:2rem;
}
</style>
"""


st.set_page_config(layout="wide", page_title="Choose your future friend", page_icon=":service_dog:")
st.markdown(page_by_image, unsafe_allow_html=True)



find_dogs_count = 20
result_column_count = 5

@st.cache_data
def get_data():
    dataframe = pd.read_csv('dog.csv')
    return dataframe


@st.cache_data
def get_pipeline():
    pipeline = joblib.load('knn_breed_recommender_pipeline.pkl')
    return pipeline

st.image("dogs.png", use_column_width=True)
main_tab, recommendation_tab, group_tab = st.tabs(["**Özellikler**", "**Özellik Öneri Uygulaması**","**Köpek Grubuna Göre Öneri Uygulaması**"])
df = get_data()

# Ana Sayfa


with main_tab:
    st.title("**🐶Köpek İyi Bir Arkadaştır!🐶**")
    st.subheader("Köpek sahiplenmek, yaşamınıza neşe ve sevgi getirecek harika bir karardır. Ancak, hangi köpek ırkının sizin yaşam tarzınıza en uygun olduğunu seçmek önemlidir. İşte sizin için geliştirdiğimiz iki öneri sistemi, doğru köpek ırkını bulmanıza yardımcı olacak!", divider='rainbow')
    col_main_features = main_tab.columns(2)
    col_main_features[0].write("""**Özellik Öneri Uygulaması ile sahip olmak istediğiniz köpek özelliklerini uygulamada puanlayarak önerileri görebilirsiniz.**
\n**Adaptasyon Yeteneği**: Köpeğinizin yeni ortamlara ne kadar kolay uyum sağladığı önemlidir. Siz değişiklikleri seviyor musunuz yoksa rutininizden mi hoşlanıyorsunuz?
\n**Yalnızlık Toleransı**: Köpeğiniz yalnız kaldığında nasıl davranır? Çoğu zaman mı evde yalnız mı kalıyorsunuz?
\n**Arkadaş Canlısı Davranış**: Evinizde çocuklar veya diğer hayvanlar var mı? Köpeğinizin başkalarıyla iyi geçinmesini mi istiyorsunuz?
\n**Sağlık**: Sağlık problemi yaşayan köpekleri bilmek veya sağlık konusunda problem yaşamayacak bir köpek istersiniz değil mi?
\n**Eğitilme Seviyesi**: Eğitilebilir bir köpek mi istiyorsunuz yoksa daha bağımsız bir karaktere sahip olan mı?
\n**Egzersiz İhtiyacı**: Aktif bir yaşam tarzınız mı var yoksa daha sakin bir ortam mı tercih ediyorsunuz?
\n**Tüy Dökme**: Tüy dökme oranı köpek seçiminiz için önemli mi?
\n**Yaşam Ortalaması**: Köpeğinizin uzun ömürlü olmasını mı istiyorsunuz?
\n**Apartman Uygunluğu**: Küçük bir dairede mi yaşıyorsunuz yoksa geniş bir bahçeniz mi var?
\n**Deneyimsiz Sahip Uygunluğu**: İlk kez mi bir köpek sahibi olacaksınız? Deneyimsizliğinize uygun köpek tercih etmek ister misiniz?
\n**Havlamaya Yatkınlık**: Köpeğiniz havlamaya yatkınlığı tercihinizi belirler mi?
\n**Boyut**: Küçük, orta veya büyük boyutta bir köpek mi istiyorsunuz?
\n"""
, border=True)
    col_main_features[1].write("""**Listelenmiş olan kategoriler veya köpek türleri için 8 farklı grup bulunmaktadır.Köpek grubuna göre öneri uygulaması ile seçiminize göre belirli gruplara özgü köpek cinslerini öğrenebilirsiniz.**

\n**Melez Irk Köpekler(Mixed Breed Dogs)**: Bu köpekler genellikle iki veya daha fazla farklı ırkın kombinasyonundan oluşur. Genellikle ebeveyn ırklarından özelliklerin bir karışımını sergilerler.

\n**Eşlikçi Köpekler(Companion Dogs)**: Aynı zamanda oyuncak veya kucağa alınan köpekler olarak da bilinen bu ırklar genellikle küçük boyutta olup başlıca arkadaş olarak yetiştirilirler. Sevecen ve sosyal doğalarıyla bilinirler.

\n**Av Köpekleri(Hound Dog)**: Av köpek ırkları güçlü koku alma duyularıyla tanınırlar ve genellikle av veya iz sürme amaçları için kullanılırlar. Farklı tiplerine göre (örneğin, koku avcıları, görüş avcıları) belirgin özelliklere sahiptirler.

\n**Teriyer Köpekler(Terrier Dogs)**: Teriyerler enerjik, kavgacı köpekler olarak bilinirler ve kararlılık ve avlanma içgüdüleriyle tanınırlar. İlk olarak fare gibi zararlıları avlamak için yetiştirilmişlerdir.

\n**Çalışma Köpekleri(Working Dogs)**: Bu ırklar tarih boyunca çeşitli görevler için yetiştirilmişlerdir, örneğin koruma, sürü yönetimi, kızak çekme veya kurtarma görevleri. Zeki, eğitilebilir ve genellikle güçlü bir çalışma ahlakına sahiptirler.

\n**Spor Köpekleri(Sporting Dogs)**: Spor köpekleri avlanma ve geri getirme gibi faaliyetler için yetiştirilirler. Genellikle aktif, enerjik ve saha denemeleri, çeviklik ve itaat gibi dış mekan aktivitelerinde başarılıdırlar.

\n**Çoban Köpekleri(Herding Dogs)**: Çoban ırkları genellikle sürüleri yönlendirme amacıyla geliştirilmişlerdir. Son derece zeki, eğitilebilir ve sürü yönlendirme içgüdülerine sahiptirler.

\n**Melez Köpekler(Hybrid Dogs)**: Melez köpekler, iki farklı safkan köpeğin bilinçli olarak çiftleştirilmesiyle ortaya çıkan çapraz ırklardır. Genellikle her iki ebeveyn ırktan istenen özellikleri bir araya getirirler. **"""
)
    st.subheader("Bu uygulamalar sizin için en uygun köpek ırkını belirlemekte yardımcı olacaktır. İki öneri sistemimiz de tercihlerinize en iyi şekilde uyacak köpekleri bulmanıza yardımcı olacak ve yeni bir dost kazanmanın ne kadar harika olduğunu gösterecektir!")




pipeline = get_pipeline()
recommendation_tab.write("Lütfen aşağıdaki özellikler için seçimlerinizi girin")

col_features = recommendation_tab.columns(6)

# Özellik listesi örnek olarak")
#Slider Değerlerini Dönüştür
numberOptions = ['Düşük', 'Az', 'Orta', 'İyi', 'Çok İyi']
exOptions = ['Düşük', 'Az', 'Orta', 'Yüksek', 'Çok Yüksek']
sizeOption = ['Küçük', 'Orta Küçük', 'Orta', 'Büyük', 'İri']
def option_to_number(option):
    if option == numberOptions[0]:
        return 1
    if option == numberOptions[1]:
        return 2
    if option == numberOptions[2]:
        return 3
    if option == numberOptions[3]:
        return 4
    if option == numberOptions[4]:
        return 5
    return 0


def option_to_size(opt):
    if opt == sizeOption[0]:
        return 1
    if opt == sizeOption[1] :
        return 2
    if opt == sizeOption[2]:
        return 3
    if opt == sizeOption[3]:
        return 4
    if opt == sizeOption[4]:
        return 5
    return 0


def option_to_ex(option):
    if option == exOptions[0]:
        return 1
    if option == exOptions[1]:
        return 2
    if option == exOptions[2]:
        return 3
    if option == exOptions[3]:
        return 4
    if option == exOptions[4]:
        return 5
    return 0



Adaptability = option_to_number(col_features[0].select_slider("Adaptasyon Yeteneği", options=numberOptions, value="Orta"))
ToleranceAlone = option_to_number(col_features[0].select_slider("Yalnızlık Toleransı", options=numberOptions, value="Orta"))
Friendliness = option_to_number(col_features[1].select_slider("Arkadaş Canlısı Davranma", options=numberOptions, value="Orta"))
Health = option_to_number(col_features[1].select_slider("Sağlık", options=numberOptions, value="Orta"))
TrainAbility = option_to_number(col_features[2].select_slider("Eğitilme Seviyesi", options=numberOptions, value="Orta"))
ExerciseNeeds = option_to_ex(col_features[2].select_slider("Egzersiz İhtiyacı", options=exOptions, value="Orta"))
Shedding = option_to_ex(col_features[3].select_slider("Düy Dökme", options=exOptions, value="Orta"))
AgeAverage = option_to_ex(col_features[3].select_slider("Yaşam Ortalaması(Yıl)", options=exOptions, value="Orta"))
ApartmentSuitability = option_to_number(col_features[4].select_slider("Apartman Uygunluğu", options=numberOptions, value="Orta"))
Size = option_to_size(col_features[4].select_slider("Boyut", options=sizeOption, value="Orta"))
InexperiencedOwner = option_to_number(col_features[5].select_slider("Deneyimsiz Sahip Uygunluğu", options=numberOptions, value="Orta"))
Noisy = option_to_ex(col_features[5].select_slider("Havlamaya Yatkınlık", options=exOptions, value="Orta"))

features = np.array([Adaptability, ApartmentSuitability, InexperiencedOwner, ToleranceAlone, Friendliness, Shedding, Health, Size, TrainAbility,  Noisy, ExerciseNeeds,  AgeAverage
       ]).reshape(1, -1)

if col_features[2].button("Öneri İstiyorum!"):
    distances, indices = pipeline.named_steps['knn'].kneighbors(pipeline.named_steps['scaler'].transform(features),
                                                                n_neighbors=find_dogs_count)

    for i in range(0, math.ceil(find_dogs_count / result_column_count)):
        col_rec = recommendation_tab.columns(result_column_count)

        for col_index in range(0, result_column_count):
            index = i * result_column_count + col_index
            recommended_dog_index = indices[0][index]
            recommended_dog = df.iloc[recommended_dog_index]
            recommended_dog_distance = distances[0][index]

            col = col_rec[col_index]

            col.write(f"**{recommended_dog['breed']}**")
            col.image(recommended_dog['Images'],use_column_width=True)
            col.write(f"**{recommended_dog['url']}**")
            col.write(f"**Grubu:** **{recommended_dog['breed_group']}**")
            col.write(f"KNN Uzaklık: {recommended_dog_distance}")


#Köpek Grubuna Göre Öneri Getir

selected_group = group_tab.selectbox('Lütfen bir grup seçin:', df['breed_group'].unique())

def draw_dog(parent, dog):
    parent.write(f"**{dog['breed']}**")
    parent.image(dog['Images'], use_column_width=True)
    parent.write(f"**{dog['url']}**")
    parent.write(f"**Grubu:** **{dog['breed_group']}**")


if group_tab.button("Öneri ver!"):
    gr_col = None
    selection = df['breed_group'] == selected_group
    filtered_dogs = df.loc[selection]
    for dog_index in range(filtered_dogs.shape[0]):
        col = dog_index % 5
        if col == 0:
            gr_col = group_tab.columns(5)
        dog = filtered_dogs.iloc[dog_index]
        draw_dog(gr_col[col], dog)




