import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Berfungsi untuk mendapatkan input pengguna untuk setiap fitur
features = ['World Rank', 'National Rank', 'Employability Rank', 'Research Rank', 'Score']

def get_user_input():
    user_input = {}
    for feature in features:
        user_input[feature] = st.number_input(f'Masukkan nilai untuk {feature}', min_value=0.0)
    return user_input

# Path absolut dari file CSV
DATA_URL = 'D:\TUGAS\Semester 4\Penggalian Data & Analitika Proses Bisnis\Praktikum\IDP - Deployment\WORLD UNIVERSITY RANKINGS.csv'

# Membaca file CSV
df = pd.read_csv('WORLD UNIVERSITY RANKINGS.csv', encoding='utf-8')

# Sidebar navigation
page = st.sidebar.selectbox("Options", ["Home", "EDA", "Modelling"])

if page == "Home":
    # judul dashboard
    st.title("Prediksi Tingkat Kualitas Universitas")

    # Gambar header
    st.image("https://storage.googleapis.com/kaggle-datasets-images/2515270/4268609/5a882dc4680834177010909c0596d976/dataset-cover.jpg", width=700)

    # Menampilkan Bussiness Understanding
    st.subheader('Bussiness Understanding')
    st.write("**Business Objective**")
    st.markdown("""
            Tujuan utama dari analisis ini adalah menganalisis tingkat kualitas universitas berdasarkan indikator yang 
            mempengaruhi untuk membangun model prediktif. Dengan pemahaman yang lebih baik tentang faktor-faktor yang 
            berkontribusi terhadap kualitas universitas, kita dapat memberikan wawasan kepada para pemangku kepentingan 
            dalam pengambilan keputusan, seperti calon mahasiswa, lembaga pendidikan, dan pemangku kepentingan lainnya.
            """)
    
    
    st.write("**Assess Situation**")
    st.markdown("""
            Saat ini, terdapat beragam peringkat universitas yang tersedia, namun demikian, mungkin ada kebutuhan untuk
            memiliki prediksi yang lebih akurat dan terkini tentang kualitas universitas. Dengan memahami peringkat dan 
            indikator yang mempengaruhinya, kita dapat mengidentifikasi faktor-faktor kunci yang membentuk reputasi dan
            kualitas universitas.
            """)
    
    st.write("**Data Mining Goals**")
    st.markdown("""
            - Mengidentifikasi hubungan antara indikator-indikator seperti World Rank, National Rank, Education Rank, Employability Rank, Faculty Rank, Research Rank dengan tingkat kualitas universitas.
            - Membangun model kalsifikasi yang dapat memprediksi tingkat kualitas universitas berdasarkan indikator-indikator tersebut.
            - Mengevaluasi performa dan akurasi model kalsifikasi yang dibangun, serta mengidentifikasi faktor-faktor yang paling berpengaruh dalam memprediksi kualitas universitas.
            """)
    
    st.write("**Project Plan**")
    st.markdown("""
            - **Pengumpulan Data:** Mengumpulkan dataset yang mencakup indikator-indikator seperti World Rank, Institution, Location, National Rank, Education Rank, Employability Rank, Faculty Rank, Research Rank, dan Score dari sumber yang dapat dipercaya.
            - **Pembersihan dan Persiapan Data:** Menyaring data yang tidak relevan atau hilang, menangani nilai-nilai yang hilang atau tidak valid, serta menyesuaikan format data agar sesuai dengan model klasfikasi.
            - **Eksplorasi Data:** Menganalisis hubungan antara variabel-variabel independen (indikator-indikator) dan variabel dependen (tingkat kualitas universitas).
            - **Pengembangan Model:** Membangun model klasifikasi menggunakan teknik data mining yang sesuai.
            - **Evaluasi Model:** Menguji performa dan akurasi model klasifikasi menggunakan metrik evaluasi yang relevan.
            """)
    
    # Menampilkan DataFrame
    st.subheader('Dataframe')
    st.write(df)
    st.write("Dataframe yang ditampilkan adalah dataframe yang berisi informasi World Rank, Institution, Location, National Rank, Education Rank, Employability Rank, Faculty Rank, Research Rank dan Score.")   

elif page == "EDA":
    st.header("Exploratory Data Analysis (EDA)")

    # Bagi halaman menjadi dua kolom
    left_column, right_column = st.columns(2)

    # Histogram untuk kolom 'Score'
    with left_column:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['Score'], color='skyblue', edgecolor='black')
        ax.set_title('Distribution of Score')
        ax.set_xlabel('Score')
        ax.set_ylabel('Frekuensi')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Bar plot untuk 10 negara teratas
    with right_column:
        sns.set(style="whitegrid")
        # Hitung jumlah universitas di setiap negara dan ambil 10 negara teratas
        top_10_countries = df['Location'].value_counts()[:10]
        # membuat bar plot untuk 10 negara teratas menggunakan Seaborn
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=top_10_countries.index, y=top_10_countries.values, hue=top_10_countries.index, legend=False, palette="viridis")
        plt.title('Top 10 Countries with the Most Universities', fontsize=16)
        # Sesuaikan label sumbu
        plt.xlabel('Country', fontsize=14)
        plt.ylabel('Number of Universities', fontsize=14)
        # Sesuaikan label sumbu x agar lebih mudah dibaca
        plt.xticks(rotation=45, ha='right')
        # Tambahkan grid untuk keterbacaan yang lebih baik
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # tampilkan plot
        st.pyplot(fig)

    # Scatter plot antara 'Score' and 'World Rank'
    with left_column:
        fig, ax = plt.subplots(figsize=(8, 6))  # Create Matplotlib  figure and axes
        sns.scatterplot(x='Score', y='World Rank', data=df, color='blue', ax=ax)  # Plot on the created axes
        ax.set_title('Relationship between Score and World Rank')
        ax.set_xlabel('Score')
        ax.set_ylabel('World Rank')
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # Pie chart untuk 10 universitas terbaik
    with right_column:
        # Kelompokkan data berdasarkan kolom "Institution" dan ambil nilai tertinggi dari kolom "Score" 
        top_universities = df.groupby('Institution')['Score'].max().nlargest(10)
        # membuat visualisasi pie chart 
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(top_universities, labels=top_universities.index, autopct='%1.1f%%', startangle=140)
        ax.set_title('Top 10 Universities with the Highest Score')
        # Show plot
        st.pyplot(fig)
    
    # Penjelasan untuk setiap visualisasi
    st.subheader("Penjelasan Visualisasi")
    st.write("**Histogram**")
    st.write("Distribusi skor cenderung miring ke kanan, menunjukkan sebagian besar skor berada pada rentang yang lebih rendah. Rentang 65-69 memiliki frekuensi tertinggi, diikuti oleh rentang 70-79, 80-89, dan 90-100 secara berurutan, dengan frekuensi yang semakin rendah.")

    st.write("**Bar Plot**")
    st.write("Gambar visualisasi di atas menunjukkan 10 negara dengan jumlah universitas terbanyak dalam dataset. Dari visualisasi tersebut, dapat dilihat bahwa negara dengan jumlah universitas terbanyak adalah Amerika Serikat (USA).")

    st.write("**Scatter Plot**")
    st.write("Pada gambar visualisasi diatas Garis diagonal menurun menunjukkan hubungan kuat antara kolom Score dan World Rank. Ketika nilai Score meningkat, peringkat dunianya cenderung menurun, dan sebaliknya, Ketika nilai Score menurun, peringkat dunianya cenderung naik")

    st.write("**Pie Chart**")
    st.write("Gambar visualisasi diatas adalah gambar pie chart yang menampilkan 10 Universitas Teratas dengan Nilai Tertinggi dalam dataset. Sehingga, dari gambar visualisasi diatas dapat diketahui universitas dengan nilai tertinggi adalah Universitas Harvard dengan presentase 10.7%")
    

elif page == "Modelling":
    # Muat model dari file
    file_path = 'knn.pkl'
    with open(file_path, 'rb') as f:
        clf = pickle.load(f)

    st.header("Modelling")

    # Dapatkan masukan pengguna untuk setiap fitur
    user_input = get_user_input()

    # Ubah input pengguna menjadi DataFrame
    input_df = pd.DataFrame(user_input, index=[0])

    # Lakukan prediksi
    prediction = clf.predict(input_df)

    if st.button('Predict'):
        # Lakukan prediksi
        prediction = clf.predict(input_df)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        prediction_label = "0 (Bad)" if prediction[0] == 0 else "1 (Good)"
        st.write(f"Tingkat Kualitas Universitas: {prediction_label}")



 

    
