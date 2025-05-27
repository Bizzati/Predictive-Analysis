# Laporan Proyek Machine Learning - Bizzati Hanif Raushan Fikri

## **Domain Proyek**
Performa akademik siswa merupakan tolak ukur keberhasilan proses belajar mengajar di lembaga pendidikan. Namun, penilaian secara konvensional 
seringkali terlambat menemukan siswa yang membutuhkan intervensi khusus. Penerapan machine learning dapat membantu memprediksi 
hasil belajar berdasarkan kebiasaan dan karakteristik siswa, sehingga pihak sekolah atau bimbingan belajar dapat:

1. Mengidentifikasi lebih cepat siswa yang berisiko rendah performanya.
2. Merancang program remedial yang tepat sasaran.
3. Mengukur dampak perubahan metode pengajaran secara kuantitatif.

Proyek ini memanfaatkan data kebiasaan belajar (jam belajar, latihan soal, jadwal tidur) dan atribut demografis (skor sebelumnya, partisipasi ekstrakurikuler) 
untuk membangun model regresi prediktif. Dengan model yang akurat, institusi pendidikan dapat meningkatkan efisiensi 
intervensi akademik dan hasil belajar secara signifikan.

## Business Understanding

### Problem Statements

1. Bagaimana memprediksi *Performance Index* siswa baru berdasarkan kombinasi fitur numerik dan kategorikal?
2. Algoritma regresi mana (Linear Regression, Ridge, Lasso) yang memberikan prediksi paling akurat?
3. Bagaimana sistem prediksi ini dapat membantu keputusan pihak sekolah?

### Goals

- Membangun pipeline regresi (preprocessing + modeling) dengan GridSearchCV (5-fold CV).
- Mencapai **MAE ≤ 5.0** pada test set.
- Menginterpretasi koefisien dan signifikansi seluruh fitur.

### Solution Statements

1. Implementasi `ColumnTransformer`:
  - `StandardScaler` pada fitur numerik
  - `OneHotEncoder(drop='first')` pada `Extracurricular Activities`
2. Uji tiga algoritma regresi (Linear, Ridge, Lasso) dengan GridSearchCV untuk Ridge & Lasso.
3. Evaluasi akhir pada test set menggunakan MAE dan R², serta analisis koefisien model terbaik.

## Data Understanding

Dataset yang digunakan pada proyek berasal dari kaggle dengan sumber: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

Dataset ini disusun untuk kebutuhan eksplorasi regresi linier dan berisi data simulasi tentang kebiasaan belajar siswa, partisipasi 
dalam kegiatan ekstrakurikuler, serta hasil akademik mereka. Dengan total 10.000 observasi, dataset ini memberikan ruang cukup luas 
untuk membangun model prediktif yang kuat dan dapat diuji secara statistik menggunakan pendekatan machine learning.

### Variabel-variabel pada Student Performance dataset adalah sebagai berikut:
- `study_time` : jam belajar per minggu
- `previous_scores` : rata-rata skor tes sebelumnya
- `sleep_hours` : jam tidur rata-rata per malam
- `sample_papers_practiced` : jumlah soal latihan yang dikerjakan
- `extracurricular_activities` : partisipasi kegiatan ekstrakurikuler; Yes/No

### Hasil EDA (*Exploratory Data Analysis*)

Demi memahami dan memvalidasi kondisi & kualitas dataset yang akan digunakan sebelum masuk ke modeling, saya melakukan beberapa tahap eksplorasi
data:
- **Pengecekan distribusi data:** Masing-masing fitur terdistribusi dengan rata ini termasuk fitur kategorikal dengan perbandingan 49/51, terkecuali
untuk data `performance_index` yang merupakan target utama. Untuk memastikan hal ini digunakan visualisasi histogram serta piechart khusus fitur kategorikal.
- **Pengecekan korelasi antar-fitur:** Kebanyakan korelasi fitur tampak lemah dengan fitur target terkecuali untuk dua variabel yaitu `previous_scores` dan
`hours_studied`. Untuk memastikan hal ini digunakan visualisasi heatmap dengan menginklusikan fitur kategorikal dengan penerapan one-hot-encoding.

## Data Preparation

Tahapan ini mempersiapkan data agar siap dipakai oleh algoritma regresi. Semua langkah dilakukan berdasarkan karakteristik dataset siswa.

1. **Menghapus Duplikasi**

   - Ditemukan 127 baris duplikat setelah pemeriksaan `df.duplicated().sum()`, seluruh baris tersebut didrop demi mencegah bias pada data yang duplikat

2. **Menangani Outlier**

   - Metode IQR (interquartile range) diterapkan pada fitur numerik. Tidak ada outlier ekstrem yang terdeteksi (`drop` indeks = 0), sehingga **data dibiarkan utuh**
     untuk menjaga variabilitas natural.

3. **Encoding Fitur Kategorikal**

   - Fitur `Extracurricular Activities` (Yes/No) diubah menjadi biner menggunakan `OneHotEncoder(drop='first')`. Menghasilkan kolom `Extracurricular Activities_Yes`
     dengan nilai 0/1.

4. **Scaling Fitur Numerik**

   - Fitur numerik (`Hours Studied`, `Previous Scores`, `Sleep Hours`, `Sample Question Papers Practiced`) distandarisasi menggunakan `StandardScaler` (mean=0, std=1).
   - Scaler *fit* hanya pada data train, lalu *transform* pada train dan test agar tidak terjadi kebocoran data.

5. **Train-Test Split**

   - Data dibagi menjadi 80% untuk pelatihan dan 20% untuk pengujian menggunakan `train_test_split(random_state=42)`.
   - Pembagian ini memastikan evaluasi model pada data yang belum pernah dilihat.

6. **Hyperparameter Tuning**

   - Untuk model regresi *Ridge* dan *Lasso*, dilakukan pencarian `alpha` optimal menggunakan `GridSearchCV` 5-fold CV.
   - `param_grid_ridge = {'reg__alpha':[0.1,1,10,100]}` dan `param_grid_lasso = {'reg__alpha':[0.001,0.01,0.1,1,10]}`.

## Modeling

### Model 1 : **Linear Regression**

#### Cara Kerja

Linear Regression adalah model regresi paling dasar yang berusaha mencari garis lurus terbaik untuk memetakan hubungan antara fitur input dan target. 
Model ini menghitung koefisien dari setiap fitur untuk meminimalkan selisih kuadrat antara prediksi dan nilai aktual.

#### Parameter

Model digunakan dengan pengaturan default dari `sklearn.linear_model.LinearRegression`:

* `fit_intercept=True`: menghitung nilai intercept.
* `normalize='deprecated'`: tidak digunakan karena scaling dilakukan terpisah.
* `n_jobs=None`: pelatihan dilakukan secara default (single thread).

#### Kelebihan

* Sangat cepat dan efisien dalam pelatihan.
* Mudah diinterpretasi karena setiap koefisien menunjukkan pengaruh linear dari fitur terhadap target.
* Cocok sebagai baseline untuk regresi.

#### Kekurangan

* Tidak memiliki mekanisme regularisasi.
* Rentan terhadap multikolinearitas dan outlier.

### Model 2 : **Ridge Regression**

#### Cara Kerja

Ridge Regression adalah versi regularisasi dari Linear Regression yang menambahkan penalti L2 terhadap koefisien. 
Hal ini membantu mengurangi varians model dan mencegah overfitting, terutama saat terdapat multikolinearitas atau fitur dengan kontribusi serupa.

#### Parameter dan Tuning

Model dituning menggunakan `GridSearchCV` untuk mencari nilai terbaik dari hyperparameter `alpha`, yaitu:

```python
param_grid_ridge = {'reg__alpha': [0.1, 1, 10, 100]}
```

Pemilihan nilai `alpha` dilakukan menggunakan validasi silang 5-fold dengan metrik `neg_mean_absolute_error`.

#### Kelebihan

* Memberikan trade-off bias-variance yang lebih baik.
* Mengurangi risiko overfitting dan stabil terhadap multikolinearitas.
* Semua fitur tetap dipertahankan.

#### Kekurangan

* Memerlukan tuning parameter `alpha`.
* Koefisien dapat tereduksi secara signifikan, mengurangi kejelasan interpretasi.

### Model 3 : **Lasso Regression**

#### Cara Kerja

Lasso Regression menambahkan penalti L1 terhadap fungsi kehilangan, yang secara alami mendorong beberapa koefisien menjadi nol. 
Hal ini membuat Lasso dapat digunakan sebagai metode seleksi fitur.

#### Parameter dan Tuning

Model dituning menggunakan `GridSearchCV` dengan grid:

```python
param_grid_lasso = {'reg__alpha': [0.001, 0.01, 0.1, 1, 10]}
```

#### Kelebihan

* Melakukan regularisasi sekaligus seleksi fitur.
* Cocok untuk dataset dengan banyak fitur tidak relevan.

#### Kekurangan

* Dapat menghilangkan fitur penting secara arbitrer jika fitur berkorelasi.
* Memerlukan tuning `alpha` yang lebih sensitif dibanding Ridge.

---

### Proses Improvement (Hyperparameter Tuning)

Untuk meningkatkan performa Ridge dan Lasso, dilakukan:

* **GridSearchCV (5-fold CV)** pada data training:

  * **Ridge:** `alpha` diuji pada `[0.1, 1, 10, 100]`.
  * **Lasso:** `alpha` diuji pada `[0.001, 0.01, 0.1, 1, 10]`.
* **Scoring:** `neg_mean_absolute_error` digunakan untuk memilih parameter yang meminimalkan MAE.
* Setelah tuning, model terbaik (`best_estimator_`) dievaluasi pada test set.

> Hasil tuning menunjukkan bahwa **Ridge** dengan `alpha = {ridge_alpha}` dan **Lasso** dengan `alpha = {lasso_alpha}`
memberikan MAE CV masing-masing {ridge\_mae:.3f} dan {lasso\_mae:.3f}.

## Evaluation

### Evaluation Metrics

Untuk mengukur performa model dalam proyek ini, digunakan dua metrik utama:

#### 1. Mean Absolute Error (MAE)
- MAE menghitung rata-rata kesalahan absolut antara nilai aktual ($y_i$) dan prediksi ($\hat y_i$).
- Metrik ini cocok untuk regresi karena memberikan estimasi seberapa jauh prediksi rata-rata model dari nilai sebenarnya, dalam satuan asli target.
- MAE tidak terlalu sensitif terhadap outlier, sehingga memberikan penilaian yang lebih stabil.

**Formula MAE**:  

$$
{MAE} = \frac{1}{n} \sum_{i=1}^{n} \bigl|y_i - \hat y_i\bigr|
$$

#### 2. Koefisien Determinasi (R²)
- R² menunjukkan proporsi varians target yang berhasil dijelaskan oleh model.
- Nilai R² berkisar antara 0 hingga 1, di mana 1 berarti model menjelaskan semua variasi data, dan 0 berarti tidak ada penjelasan sama sekali.
- Metrik ini membantu menilai seberapa baik model menangkap pola data.

**Formula R²**:  

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat y_i)^2}{\sum_{i=1}^{n} (y_i - \bar y)^2}
$$

### Pemilihan Model Terbaik

Berdasarkan hasil evaluasi test set:

| Model             | Test MAE | Test R² |
| ----------------- | -------: | ------: |
| Linear Regression |    1.646 |   0.988 |
| Ridge Regression  |    1.646 |   0.988 |
| Lasso Regression  |    1.646 |   0.988 |

- **MAE** dan **R²** ketiga model sangat mirip dengan perbandingan yang dapat dibilang insignifikan.
- Namun, **Ridge Regression** dipilih sebagai model terbaik karena:

  1. **Regularisasi L2** memberikan kestabilan koefisien di berbagai subset data.
  2. Tetap mempertahankan semua fitur sehingga interpretasi lengkap.
  3. Tawarkan trade-off bias-variance yang lebih baik dibanding Linear biasa.
  4. Menghasilkan skor MAE terkecil dibanding model lainnya dengan perbedaan 0,000001 dengan regresi linear

Model Ridge ini kemudian digunakan untuk analisis final dan rekomendasi kuantitatif.

### Feature Importance (Ridge Regression)

Selain evaluasi hasil test model sebelumnya, untuk membuktikan faktor fitur yang paling signifikan dalam keputusan model, dilakukan
pengujian *feature importance* dengan hasil:

| Feature                          | Contribution (%) | Coefficient |
| -------------------------------- | ---------------: | ----------: |
| Previous Scores                  |            65.5% |      17.620 |
| Hours Studied                    |            27.4% |       7.373 |
| Sleep Hours                      |             3.0% |       0.802 |
| Extracurricular Activities (Yes) |             2.1% |       0.574 |
| Sample Question Papers Practiced |             2.0% |       0.540 |

Sampel di atas membuktikan tingginya bias model terhadap 2 fitur yaitu `Previous Scores` dan `Hours Studied` lebih besar >95% dibanding fitur lainnya.


## Conclusion

Dengan hasil analisis sebelumnya dapat disimpulkan beberapa hal terutama untuk menjawab *business undestanding* di awal:

1. **Model untuk prediksi performance index**: Model **Ridge Regression** memenuhi target MAE ≤ 5.0 dengan skor terendah MAE ≈ 1.65 dan R² ≈ 0.988.
2. **Algoritma Terbaik**: Ketiga algoritma (Linear, Ridge, Lasso) memiliki performa setara, namun ridge dipilih karena
   memberikan kestabilan koefisien di berbagai subset data.
3. **Rekomendasi**: Berdasarkan analisa *feature importance*, peningkatan **Previous Scores** dan **Hours Studied** adalah strategi paling efektif untuk
   menaikkan nilai siswa.
