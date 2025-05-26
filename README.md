# Laporan Proyek Machine Learning - Muhammad Nurul Fatwa Al Fajar
## Domain Proyek
Proyek machine learning ini berfokus pada bidang **Holtikultura/Pertanian Apel/Pertanian**, dengan judul **Proyek Pertama : Prediksi Apel**

### Latar Belakang


![foto apel](https://png.pngtree.com/thumb_back/fw800/background/20230730/pngtree-thousands-of-red-apples-stacked-up-in-piles-on-the-counter-image_10175220.jpg)

Sebagai salah satu produsen utama apel di kawasan Asia Tenggara, Indonesia menghasilkan lebih dari 523.738 Ton pada Tahun 2022. Komoditas ini memiliki peranan penting, terutama dalam mendukung penghidupan petani lokal serta pertumbuhan sektor agrikultur nasional [[1](https://dataindonesia.id/agribisnis-kehutanan/detail/produksi-apel-di-indonesia-sebanyak-523738-ton-pada-2022)].Namun demikian, menjaga mutu apel tetap menjadi tantangan besar di industri ini. Kualitas yang menurun, misalnya karena ukuran yang terlalu kecil, tingkat kematangan yang tidak seragam, atau tekstur yang kurang renyah, dapat menurunkan nilai jual dan menyebabkan kerugian di berbagai lini distribusi. Dengan memanfaatkan pendekatan predictive analytics, pelaku industri apel dapat memperoleh gambaran prediktif terkait mutu buah[[2](https://www.researchgate.net/publication/378686300_Optimasi_Algoritma_Naive_Bayes_Untuk_Klasifikasi_Buah_Apel_Berdasarkan_Fitur_Warna_RGB/fulltext/65e46bc8adc608480af78397/Optimasi-Algoritma-Naive-Bayes-Untuk-Klasifikasi-Buah-Apel-Berdasarkan-Fitur-Warna-RGB.pdf?origin=scientificContributions)]. Teknologi ini membuka peluang untuk optimalisasi panen bagi petani, pengurangan limbah distribusi bagi penyalur, dan jaminan mutu bagi konsumen. Selain itu, prediksi berbasis data juga dapat membantu menciptakan kestabilan harga dan ketersediaan apel berkualitas di pasar.

## Business Understanding
Pengembangan model prediksi untuk menilai kualitas apel memiliki potensi besar dalam memberikan dampak positif bagi berbagai pihak, khususnya petani dan distributor. Dengan bantuan model ini, proses panen dapat menjadi lebih efisien, nilai jual apel dapat meningkat, dan kepercayaan konsumen terhadap produk juga bisa lebih terjaga. Sebagai contoh, prediksi yang akurat terkait kualitas apel akan membantu petani dalam proses klasifikasi hasil panen dan dalam menetapkan harga jual yang sesuai di pasar.
### Problem Statements
Berdasarkan uraian di atas, beberapa pertanyaan utama yang ingin dijawab dalam proyek ini antara lain:
-  Bagaimana merancang model machine learning yang mampu memprediksi kualitas apel berdasarkan data kuantitatif
-  Algoritma pembelajaran mesin apa yang memberikan performa prediksi terbaik?
### Goals
Tujuan dari proyek ini meliputi:
- Mengembangkan model machine learning yang mampu memprediksi kualitas apel berdasarkan data kuantitatif.
- Membandingkan beberapa algoritma model untuk menemukan akurasi terbaik dalam memprediksi kualitas apel.
- Mengembangkan sebuah sistem atau algoritma sederhana yang dapat dimanfaatkan oleh petani dan distributor sebagai alat bantu dalam menilai dan memasarkan produk apel secara lebih efisien.

### Solution Statements
Untuk menjawab permasalahan yang telah dirumuskan, solusi yang diusulkan dalam proyek ini meliputi:
- Mengembangkan model machine learning yang mampu mengklasifikasikan kualitas apel berdasarkan data kuantitatif
- Melakukan evaluasi terhadap beberapa algoritma machine learning untuk menentukan model dengan akurasi terbaik dalam memprediksi kualitas apel.
- Menerapkan beberapa metode pembelajaran mesin, antara lain:
    * K-Nearest Neighbor (KNN)
      Algoritma klasifikasi berbasis jarak yang menentukan kualitas apel berdasarkan kedekatan karakteristik data dengan data lain yang sudah diketahui kelasnya.[[3](https://www.geeksforgeeks.org/k-nearest-neighbours/)]
    * Random Forest
      adalah algoritma machine learning yang kuat yang dapat digunakan untuk berbagai tugas termasuk regresi dan klasifikasi. Ini adalah metode ensemble, yang berarti bahwa model random forest terdiri dari banyak decision tree kecil, yang
      disebut estimator, yang masing-masing menghasilkan prediksi mereka sendiri. Random forest menggabungkan prediksi estimator untuk menghasilkan prediksi yang lebih akurat .[[4](https://www.ibm.com/think/topics/random-forest)]
    * Naive Bayes adalah Model klasifikasi berbasis probabilistik yang mengasumsikan independensi antar fitur, dan memanfaatkan prinsip Teorema Bayes untuk prediksi.[[5](https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/bayesian/naive_bayes.html#:~:text=Naive%20Bayes%20is%20a%20high,sentiment%20analysis%2C%20and%20recommender%20systems.)]

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Dari Datasets**

| Tipe | Keterangan |
| ------ | ------ |
| Title | Apple Quality |
| Source | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data) |
| Maintainer | [Nidula Elgiriyewithana](https://www.kaggle.com/nelgiriyewithana) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | Computer Science, Education, Food, Data Visualization, Classification,Exploratory Data Analysis |
| Usability | 10.00 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer, data ini didapat dari sebuah perusahaan pertanian Amerika, yang disediakan secara publik di kaggle dengan nama datasets yaitu: Apple Quality

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| 0.0 | -3.970049 |-2.512336 | 5.346330 |-1.012009 | 1.844900 |0.329840	| -0.491590483  |good |
| 1.0 | -1.195217 |-2.839257 | 3.664059 |1.588232 | 0.853286 | 0.867530 | -0.722809367  |good |
| 2.0 | -0.292024 |	-1.351282 | -1.738429 | -0.342616 | 2.838636 |-0.038033	| 2.621636473  |bad |
| 3.0 | -0.657196 |-2.271627 | 1.324874 |-0.097875 | 3.637970 |-3.413761	| 0.790723217  |good |
| 4.0 | 1.364217 |-1.296612 | -0.384658 | -0.553006 | 3.030874 | -1.303849	| 0.501984036  |good |

4001 rows × 9 columns

Tabel 1. EDA Deskripsi Variabel

Berdasarkan hasil eksplorasi data awal (Exploratory Data Analysis), dataset Apple Quality telah melalui proses cleaning dan normalization oleh pembuatnya, sehingga sangat cocok digunakan, terutama bagi pemula dalam bidang data science.
Berikut Informasi Pada Dataset :
- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- ada 1 missing value pada dataset dan akan dihapus
- ada 1 fitur yang tidak digunakan dan akan dihapus
- 7 kolom memiliki tipe data numerik float64, yaitu: A_id, Size, Weight, Sweetness, Crunchiness, Juiciness, dan Ripeness.
- 2 kolom lainnya bertipe data object, yaitu: Acidity dan Quality.
- ada 210 outlier yang akan dihapus berdasarkan analisis IQR
- Data Duplikat Tidak Ditemukan
- **Kolom tidak relevan**: `A_id` dihapus karena hanya ID
  
### Variabel-variabel dalam Dataset
- `A_id` : Merupakan identitas unik untuk masing-masing apel
- `Size` : Menunjukkan ukuran dari apel
- `Weight` : Mewakili massa atau bobot apel
- `Sweetness` : Menggambarkan seberapa manis rasa apel
- `Crunchiness` : Menunjukkan tingkat kerenyahan dari tekstur apel
- `Juiciness` : Mengindikasikan seberapa segar atau berair apel tersebut
- `Ripeness` : Menggambarkan level kematangan buah
- `Acidity` : Menyatakan kadar keasaman yang dimiliki apel
- `Quality` : Menjadi indikator penilaian kualitas keseluruhan apel
  
Dari sembilan fitur yang tersedia dalam dataset, diketahui bahwa fitur A_id hanya berfungsi sebagai identifikasi unik setiap entri data dan tidak memiliki pengaruh langsung terhadap penilaian kualitas apel. Oleh karena itu, fitur ini dianggap tidak relevan untuk proses pelatihan model machine learning dan akan dihapus dari dataset sebelum proses pemodelan dilakukan.

### EDA - Univariate Analysis ( Analisis Univariat )

![13f79c57-ec39-4f96-b371-e4bab4154ca5](https://github.com/user-attachments/assets/518521db-bf5b-475d-b7d9-153fd9218d49)


Gambar 1a. Analisis Univariat (Data Bertipe Kategori) 

![4f4fc852-43c2-47de-a33f-ca5cb4541a8c](https://github.com/user-attachments/assets/97684b43-aad0-479a-994f-399af7db3c73)


Gambar 1b. Analisis Univariat (Data Bertipe Numerik) 

Distribusi Data dan Karakteristik Numerik :

Berdasarkan Gambar 1a, distribusi kategori kualitas apel terdiri dari dua kelas: 
- bad sebanyak 1928 sampel, dan
- good sebanyak 1862 sampel.
  
Selisih jumlah keduanya relatif kecil, sehingga data dapat dianggap cukup seimbang.

Pada Gambar 1b, data numerik menunjukkan karakteristik sebagai berikut:

- Size berkisar antara -2 hingga 2, dengan nilai rata-rata sekitar -0.51.
- Weight memiliki rata-rata -0.99 dan nilai maksimum mencapai 3.08.
- Sweetness memiliki nilai rata-rata -0.48.
- Crunchiness berada pada kisaran 0 hingga 2, mengindikasikan rata-rata apel tergolong renyah.
- Juiciness dan Ripeness masing-masing berada di sekitar nilai 0.50 dan 0.53.
- Acidity memiliki nilai rata-rata sekitar 0.06.

Nilai-nilai tersebut mengindikasikan bahwa data numerik telah melalui proses normalisasi z-score, di mana setiap nilai dikurangi dengan rata-rata dan dibagi dengan standar deviasi. Hal ini membuat skala seluruh fitur menjadi seragam dan memudahkan algoritma pembelajaran mesin dalam proses pelatihan model.

Dalam kasus ini, nilai rata-rata (mean) untuk fitur Size tercatat sebesar -0.51, sementara standar deviasinya tidak dijelaskan secara eksplisit. Namun, karena nilai-nilainya berada dalam rentang -2 hingga 2, besar kemungkinan fitur ini telah dinormalisasi menggunakan metode z-score normalization, yang bertujuan untuk menyetarakan skala data dengan rata-rata mendekati 0 dan deviasi standar mendekati 1.

Fitur numerik lainnya seperti Weight, Sweetness, Crunchiness, Juiciness, Ripeness, dan Acidity juga menunjukkan pola distribusi yang serupa, sehingga dapat diasumsikan bahwa seluruh fitur numerik dalam dataset ini telah melalui proses normalisasi yang sama. Proses ini penting untuk memastikan bahwa setiap fitur memiliki kontribusi yang setara dalam pelatihan model machine learning.


 

### EDA - Multivariate Analysis

![Multivariate_Analysis](https://github.com/user-attachments/assets/cfbbc97e-fd13-450e-bba4-473cc4db6186)

Gambar 2a. Analisis Multivariat

![Matrix_Korelasi](https://github.com/user-attachments/assets/785266c1-db35-470f-9806-d9a863fe5adb)

Gambar 2b. Analisis Matriks Korelasi

Pada Gambar 2a, digunakan visualisasi pairplot dari pustaka Seaborn untuk melihat hubungan antar fitur dalam dataset. Hasil plot memperlihatkan bahwa sebagian besar pasangan fitur menunjukkan pola sebaran yang acak. Namun, terdapat indikasi adanya korelasi negatif antara Size dan Sweetness, di mana semakin kecil ukuran apel, maka kecenderungan rasanya semakin manis.

Sedangkan pada Gambar 2b, ditampilkan correlation matrix yang mengukur hubungan antar fitur numerik. Terlihat bahwa fitur Juiciness memiliki korelasi positif sebesar `0.24` terhadap target Acidity, yang berarti semakin juicy buahnya, semakin tinggi tingkat keasamannya, meskipun hubungan ini tidak terlalu kuat.

## Data Preparation
Tahapan Data Preparation ini mencakup  :

1. Data Cleaning :
Tindakan yang dilakukan dalam proses pembersihan data mencakup:
- penyesuaian data : ada beberapa data yang tidak sesuai yang akan mempengaruhi hasil akhir model dan akan dibersihkan.
- Konversi tipe kolom: Menyesuaikan tipe data agar sesuai kebutuhan analisis.
- Train-test split: Membagi dataset menjadi data pelatihan dan pengujian.
- Normalisasi: Menyesuaikan skala fitur agar memiliki rentang nilai yang sebanding, untuk meningkatkan performa model machine learning.

------------------------------------------------------------------------------------------------------------------------------------

### 1. Melakukan Penghapusan Kolom "A_id" Karena tidak diperlukan
penghapusan kolom ini dilakukan karena A_id tidak memberikan dampak pada model yang dilatih

| kode |
| --- |
| data.drop("A_id", axis=1, inplace=True) |

### 2. Penyesuaian Terhadap Missing Value

Pada proyek ini tidak ditemukan adanya data duplikat, namun terdapat missing value pada salah satu fitur. Jumlah missing value yang ditemukan hanyalah 1 data, sehingga untuk penanganannya digunakan metode dropping, yaitu menghapus baris yang memiliki nilai kosong. Metode ini dipilih karena dampaknya terhadap keseluruhan data sangat kecil dan tidak signifikan.

membuang nilai kosong
|kode|
| --- |
|data_miss = data[data.isnull().any(axis=1)]
data_miss|

### 3. Menghapus semua baris di dataframe data yang mengandung setidaknya satu nilai kosong (NaN).

|kode|
| ----------------------- |
|data.dropna(inplace=True)
data.isnull().sum().sum()|

|Hasil|
| --- |
| np.int64(0) |

Data Tidak Relevan Sudah Kita Buang

### 4. Mengubah Tipe Data Kolom Acidity Yang Semula "Object" menjadi float64

|kode|
| --- |
|data["Acidity"] = data["Acidity"].astype("float64")|

### 5. Mengeliminasi Outlier dari Dataset

Berdasarkan hasil deteksi sebelumnya, terdapat sekitar 210 outlier yang tersebar pada beberapa fitur numerik. Untuk menangani hal ini, digunakan metode IQR (Interquartile Range) yang efektif dalam mengidentifikasi nilai-nilai ekstrem di luar distribusi utama data. Rumus perhitungannya adalah:

$$ IQR = Q_3 - Q_1$$

- Q1 (25% dari data) adalah kuartil pertama
- Q3 (75% dari data) adalah kuartil ketiga

Nilai yang berada di bawah ( Q1 - 1.5 x IQR ) atau di atas ( Q3 + 1.5 x IQR ) dianggap sebagai **outlier** dan akan dihapus dari dataset untuk menjaga kualitas dan konsistensi data selama proses pelatihan model.

Sebagai hasil dari proses ini, jumlah data berkurang dari 4000 menjadi 3790 baris. Langkah ini bertujuan untuk menjaga kualitas data dan memastikan performa model machine learning tetap optimal tanpa gangguan dari nilai-nilai ekstrem.

### 6. Mengubah nilai di kolom Quality dari bentuk teks menjadi angka.

|kode|
| --- |
| # mengubah nilai di kolom Quality dari bentuk teks menjadi angka.
data['Quality'] = data['Quality'].apply(lambda x: 1 if x == 'good' else 0)  # good:1 , bad:0 |


### 7. memisahkan data fitur dan label sebelum melakukan pelatihan
|kode|
| --- |
|# Untuk memisahkan data fitur dan label sebelum melakukan pelatihan model machine learning.
x = data.loc[:, data.columns != 'Quality']
y = data['Quality']
x.shape, y.shape|

| hasil |
| --- |
|((3790, 7), (3790,))|

Kode tersebut digunakan untuk memisahkan fitur (X) dan label/target (y) dari sebuah dataset sebelum model machine learning dilatih.

### 8. Train-Test-Split

| kode |
| --- |
|# Split data menjadi 80% untuk pelatihan dan 20% untuk pengujian|
|x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=60)|

Selanjutnya, dilakukan pembagian data menjadi data latih dan data uji menggunakan fungsi train_test_split dari library sklearn.model_selection. Pembagian dilakukan dengan proporsi 80% untuk data latih dan 20% untuk data uji, serta menggunakan nilai random_state=60 agar hasil pembagian data dapat direproduksi secara konsisten.

### 9. Normalisasi
|kode|
| --- |
|# Normalisasi fitur menggunakan Min-Max Scaling|
|normalizer = MinMaxScaler()|
|# Replace 'X_train' with 'x_train'.|
|normalizer.fit(x_train)|
| --- |
|X_train_scaled = normalizer.transform(x_train)|
|X_test_scaled = normalizer.transform(x_test)|

Mengapa penting? Untuk memastikan semua fitur (kolom) memiliki pengaruh yang seimbang saat digunakan oleh algoritma machine learning, karena banyak algoritma sensitif terhadap perbedaan skala nilai.

Untuk proses normalisasi, digunakan library sklearn.preprocessing dengan metode MinMaxScaler. Metode ini mentransformasi setiap fitur numerik ke dalam skala rentang [0, 1], yang bertujuan untuk menghindari dominasi fitur tertentu akibat perbedaan skala nilai. Seluruh proses ini penting untuk memastikan model yang dibangun dapat bekerja secara optimal dan memberikan hasil prediksi yang akurat.

# Modeling
Pada proyek ini, dilakukan pemodelan menggunakan tiga algoritma machine learning yang umum digunakan dalam kasus klasifikasi. Tujuannya adalah untuk membandingkan performa masing-masing algoritma dan memilih model dengan akurasi terbaik dalam mengklasifikasikan kualitas apel. Adapun ketiga algoritma tersebut adalah sebagai berikut:

## 1. K-Nearest Neighbors (KNN)

K-Nearest Neighbors (KNN) adalah algoritma machine learning yang bersifat non-parametrik dan berbasis instance-based learning. Algoritma ini digunakan baik untuk klasifikasi maupun regresi, dengan prinsip utama bahwa data baru akan diklasifikasikan berdasarkan kesamaan dengan sejumlah tetangga terdekat dari data pelatihan.

Pada proyek ini, KNN digunakan dengan parameter sebagai berikut:

- n_neighbors=5: 5 tetangga terdekat yang akan digunakan untuk membuat prediksi.
- weights='distance': memberikan bobot lebih besar kepada tetangga yang lebih dekat dengan data yang diprediksi.

Keunggulan KNN:

- Serbaguna: Dapat digunakan untuk klasifikasi maupun regresi.
- Sederhana: Mudah diimplementasikan dan dipahami secara intuitif.
- Tidak perlu pelatihan model: Karena KNN langsung menggunakan data latih untuk prediksi (lazy learner).

Kelemahan KNN:

- Sensitif terhadap outlier: Data yang menyimpang dapat mempengaruhi hasil prediksi secara signifikan
- Waktu komputasi tinggi: Terutama pada dataset besar, karena seluruh dataset digunakan saat prediksi.
- Pemilihan nilai K yang optimal: Nilai K yang terlalu kecil bisa menyebabkan overfitting, sedangkan nilai yang terlalu besar dapat menyebabkan underfitting.

## 2. Random Forest

Random Forest adalah algoritma ensemble learning yang membangun sejumlah Decision Tree secara acak, lalu menggabungkan hasilnya melalui voting (untuk klasifikasi) atau rata-rata (untuk regresi). Tujuan utama dari pendekatan ini adalah untuk meningkatkan akurasi dan stabilitas model, sekaligus mengurangi risiko overfitting yang sering terjadi pada decision tree tunggal.

Pada proyek ini, digunakan parameter:
- max_depth=20: untuk mengatur kedalaman maksimum dari masing-masing pohon keputusan (decision tree) dalam hutan, disini memakai 20.

Keunggulan Random Forest:

- Akurasi prediksi tinggi, karena menggabungkan banyak pohon yang mengurangi risiko kesalahan satu pohon.
- Mampu menangani data berdimensi tinggi serta data dengan variabel kategori dan numerik.
- Robust terhadap outlier dan noise pada data.

Kelemahan Random Forest:

- Rentan overfitting pada dataset kecil, karena terlalu banyak pohon bisa mempelajari noise.
- Waktu pelatihan dan prediksi lebih lama, apalagi jika jumlah pohon besar.
- Kurang interpretatif: Sulit untuk memahami kontribusi tiap fitur dalam prediksi akhir secara langsung.

## 3. Naïve Bayes Classifier

Naïve Bayes Classifier adalah algoritma klasifikasi berbasis teorema Bayes, yang mengasumsikan bahwa setiap fitur (variabel) bersifat independen terhadap fitur lainnya. Meskipun asumsi ini seringkali tidak realistis dalam praktik, algoritma ini tetap menunjukkan performa yang baik dalam berbagai kasus, terutama untuk dataset yang besar dan sederhana.

Algoritma ini menghitung probabilitas posterior dari setiap kelas berdasarkan nilai fitur dari data yang diamati, lalu memilih kelas dengan probabilitas tertinggi sebagai prediksi.

Keunggulan Naïve Bayes Classifier:

- Mudah dipahami dan diimplementasikan, cocok untuk pemula dalam machine learning.
- Cepat dalam pelatihan dan prediksi, bahkan pada dataset berukuran besar.
- Efisien dalam menangani data teks, seperti pada aplikasi spam filtering atau sentiment analysis.

Kelemahan Naïve Bayes Classifier:

- Asumsi independensi antar fitur sering kali tidak valid pada data dunia nyata.
- Sensitif terhadap fitur dengan nilai nol, yang bisa menyebabkan probabilitas akhir menjadi nol (namun bisa diatasi dengan Laplace Smoothing).
- Performa kurang optimal untuk dataset yang sangat kompleks atau memiliki hubungan antar fitur yang kuat.


## Evaluation

Dalam tahap evaluasi, metrik yang digunakan adalah Accuracy. Metrik ini mengukur sejauh mana model mampu memprediksi kelas dengan benar.

Rumus Accuracy:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TP + TN + FP + FN}} \times 100%\%$$

Keterangan:

- TP (True Positive): Data positif yang diprediksi benar sebagai positif.
- TN (True Negative): Data negatif yang diprediksi benar sebagai negatif.
- FP (False Positive): Data negatif yang salah diprediksi sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Data positif yang salah diprediksi sebagai negatif (Kesalahan Tipe II).
- Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 3 buah model yang latih:

| Model | Accuracy |
| ------ | ------ |
| KNN | 0.89 |
| RandomForest  | 0.88 |
| Naive Bayes | 0.64 |


Tabel 3. Hasil Accuracy

![Akurasi_Model](https://github.com/user-attachments/assets/3bd88f0c-bbb1-48e3-839f-71c76a318d3e)

Gambar 3. Visualisasi Accuracy Model

### Kesimpulan Evaluasi

Model KNN yang diperoleh hasil menunjukkan performa terbaik dalam klasifikasi kualitas apel, dengan akurasi mencapai **89.8%**. Hasil ini menjawab pertanyaan utama dalam proyek, yaitu:

- **Bagaimana cara membangun model machine learning untuk memprediksi kualitas apel?**
- **Algoritma apa yang paling akurat?**

Model ini dapat memberikan manfaat nyata bagi petani maupun distributor apel, di antaranya:

- Membantu dalam **klasifikasi otomatis hasil panen**.
- Mempermudah penentuan **standar mutu dan harga jual**.
- Mengurangi ketergantungan pada penilaian manual yang subjektif.

Dengan demikian, pendekatan klasifikasi menggunakan KNN tidak hanya memberikan hasil akurasi yang tinggi, tetapi juga **relevan dan aplikatif untuk digunakan dalam sistem pendukung keputusan di sektor hortikultura**, khususnya untuk komoditas apel.



## Referensi
1. Sarnita Sadya.(2022). Produksi Apel Indonesia Sebanyak 509.544 Ton pada 2022. https://dataindonesia.id/agribisnis-kehutanan/detail/produksi-apel-di-indonesia-sebanyak-523738-ton-pada-2022

2. M. Afriansyah, Joni Saputra, Yuan Sa’adati, Valian Yoga Pudya Ardhana,  https://www.researchgate.net/publication/378686300_Optimasi_Algoritma_Naive_Bayes_Untuk_Klasifikasi_Buah_Apel_Berdasarkan_Fitur_Warna_RGB/fulltext/65e46bc8adc608480af78397/Optimasi-Algoritma-Naive-Bayes-Untuk-Klasifikasi-Buah-Apel-Berdasarkan-Fitur-Warna-RGB.pdf?origin=scientificContributions

3. K-Nearest Neightbour https://www.geeksforgeeks.org/k-nearest-neighbours/

4. What Is Random Forest?  https://www.ibm.com/think/topics/random-forest

5. Naive Bayes https://docs.rapidminer.com/latest/studio/operators/modeling/predictive/bayesian/naive_bayes.html#:~:text=Naive%20Bayes%20is%20a%20high,sentiment%20analysis%2C%20and%20recommender%20systems.
