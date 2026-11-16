# Klasifikasi Grade Nutri-Score Menggunakan Supervised Machine Learning

**Portfolio Project — Dzaki Rafif Fakhriyanto**  
**Kategori:** Supervised Learning · Multiclass Classification · Model Serving

## Ringkasan Proyek

Proyek ini membangun pipeline machine learning untuk memprediksi grade Nutri-Score produk makanan kemasan dari lima kelas berurutan: **A, B, C, D, dan E**. Fokus proyek bukan hanya memperoleh skor prediksi, tetapi juga menjaga evaluasi agar tidak mengalami kebocoran data, membandingkan model secara adil, menganalisis kesalahan, dan menyiapkan model untuk digunakan melalui API.

Notebook utama: `notebooks/01_NutriScore_Supervised_Learning.ipynb`.

## Tujuan Penelitian

1. Mengukur seberapa baik grade Nutri-Score dapat diprediksi dari informasi gizi, kategori produk, dan penanda alergen.
2. Mengidentifikasi fitur yang paling berpengaruh terhadap hasil prediksi.
3. Menguji apakah fitur rekayasa berbasis pengetahuan domain membantu model.
4. Membandingkan klasifikasi biasa, pendekatan ordinal, dan ensemble untuk target yang memiliki urutan A–E.
5. Menyimpan pipeline dan menyediakan contoh endpoint prediksi.

## Dataset

Notebook memuat tabel master `nutriscore_master` dari database SQLite `datawrangling_fix_supervised.db`, dengan cadangan berupa `nutriscore_master.csv`.

- Data awal: **5.052 baris dan 22 kolom**.
- Sumber integrasi yang dicatat pada notebook: `foods_health_scores_allergens` dan `foods_dietary_restrictions` (Kaggle).
- Setelah pembersihan: **4.405 produk**.
- Grup produk–merek unik: **4.237**.
- Kolom mencakup grade Nutri-Score, kategori produk, kandungan gizi per 100 g, beberapa penanda alergen, serta informasi NOVA/Ecoscore untuk eksperimen ablasi.

> **Catatan atribusi:** notebook mencatat nama dua sumber dataset Kaggle, tetapi tidak memuat URL halaman dataset atau rincian lisensinya. Sebelum mempublikasikan dataset ke repository, lengkapi tautan sumber, atribusi, dan pastikan ketentuan lisensi mengizinkan redistribusi. Dataset tidak disertakan secara otomatis dalam repository ini.

## Metodologi

Alur eksperimen terdiri dari:

1. **Setup dan pemuatan data** — membaca tabel SQLite atau CSV cadangan.
2. **Audit kualitas data** — memeriksa missing values, duplikasi, distribusi kelas, dan nilai gizi yang tidak masuk akal.
3. **Data cleaning** — menstandardisasi tipe dan teks, menghapus label di luar A–E, menghapus duplikat, mengubah nilai gizi yang mustahil menjadi missing, dan menghapus baris tanpa informasi gizi.
4. **Feature engineering** — membentuk rasio nutrisi, pendekatan poin Nutri-Score, flag kategori produk, dan fitur alergen.
5. **EDA dan uji statistik** — membandingkan karakteristik fitur terhadap grade menggunakan Kruskal–Wallis dan korelasi Spearman.
6. **Group-aware split** — memisahkan data berdasarkan gabungan nama produk dan merek agar grup yang sama tidak muncul di train dan test sekaligus.
7. **Pipeline preprocessing** — imputasi, scaling bila dibutuhkan, encoding kategori, dan model ditempatkan dalam pipeline agar proses preprocessing dipelajari hanya dari data latih pada setiap fold.
8. **Model comparison dan tuning** — membandingkan baseline, model klasifikasi, model ordinal, dan soft-voting ensemble.
9. **Evaluasi akhir** — menggunakan test set yang ditahan, bootstrap confidence interval, metrik ordinal, dan classification report.
10. **Ablation, interpretasi, error analysis, dan model serving** — mengevaluasi kontribusi kelompok fitur, permutation importance, pola kesalahan, serialisasi model, dan contoh FastAPI.

## Model yang Dibandingkan

- Dummy Classifier (baseline mayoritas)
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting
- Ordinal-HGB (pendekatan Frank–Hall)
- Soft-voting Random Forest + HistGradientBoosting
- Rule-based sederhana dari pendekatan poin (baseline non-ML)

Model final dipilih berdasarkan **macro-F1 cross-validation pada data latih**, bukan berdasarkan skor test set.

## Hasil Utama

### Evaluasi pada test set

| Metrik | Hasil |
|---|---:|
| Accuracy | **84,1%** |
| Macro-F1 | **0,837** |
| Balanced accuracy | **0,836** |
| Quadratic Weighted Kappa (QWK) | **0,936** |
| MAE dalam langkah grade | **0,186** |
| Prediksi dengan selisih maksimal satu grade | **97,6%** |
| Ukuran test set | **881 produk** |

95% bootstrap CI:
- Accuracy: **81,5%–86,3%**
- Macro-F1: **0,809–0,860**

### Model final

**Random Forest (tuned)** terpilih berdasarkan macro-F1 cross-validation pada data latih:

- CV macro-F1: **0,830 ± 0,013**
- CV accuracy: **0,836**
- CV QWK: **0,934**

### Baseline dan per kelas

- Dummy baseline: macro-F1 CV **0,078**.
- Rule-based baseline: macro-F1 CV **0,394**.
- Test macro-F1 model final: **0,837**.

F1 per grade pada test set:

| Grade | F1-score | Support |
|---|---:|---:|
| A | 0,897 | 164 |
| B | 0,751 | 128 |
| C | 0,814 | 213 |
| D | 0,830 | 179 |
| E | 0,892 | 197 |

## Interpretasi Fitur

Permutation importance pada test set menunjukkan penurunan macro-F1 terbesar saat fitur berikut diacak:

1. `sugars_100g`: **0,2964**
2. `saturated_fat_100g`: **0,2747**
3. `sodium_100g`: **0,2416**
4. `energy_kcal`: **0,2014**
5. `fat_100g`: **0,1838**

Hasil ini menunjukkan bahwa kandungan gula, lemak jenuh, dan natrium merupakan fitur yang paling berpengaruh pada model yang diuji.

## Studi Ablasi Fitur

Macro-F1 cross-validation berdasarkan kelompok fitur:

| Skenario | Macro-F1 |
|---|---:|
| Nutrisi mentah | 0,7865 |
| + Rasio | 0,7883 |
| + Poin Nutri-Score | 0,8114 |
| + Kategori produk | 0,8302 |
| + Alergen (konfigurasi model final pada ablasi) | 0,8211 |
| + NOVA & Ecoscore | 0,8290 |

Penambahan fitur kategori memberi peningkatan yang terlihat pada rangkaian ablasi ini. Penambahan alergen tidak meningkatkan skor dibanding skenario kategori saja; selisihnya masih berada dalam variasi antar-fold yang dilaporkan pada notebook. Fitur alergen dipertahankan sebagai konteks produk, bukan diklaim sebagai pendorong performa.

## Analisis Error

- Akurasi kategori minuman: **0,800** (446 produk uji).
- Akurasi non-minuman: **0,883** (435 produk uji).
- Kategori snacks: **0,899** (168 produk uji).
- Kategori dairies: **0,891** (119 produk uji).
- Kategori plant-based foods and beverages: **0,812** (340 produk uji).
- Dari 881 produk uji, terdapat 140 kesalahan prediksi; **97,6% seluruh prediksi** tetap berselisih maksimal satu grade.

Kategori dengan aturan penilaian khusus dan keterbatasan informasi komposisi menjadi area yang perlu diperhatikan.

## Pencegahan Data Leakage

Desain evaluasi notebook menerapkan:

- Group-aware train/test split berdasarkan nama produk dan merek; tidak ada grup yang beririsan antara train dan test.
- Imputasi, scaling, dan encoding berada di dalam `Pipeline`.
- Cross-validation berkelompok dilakukan pada data latih.
- Test set hanya digunakan pada evaluasi akhir.

## Model Serving

Notebook menyimpan pipeline dengan `joblib`, metadata eksperimen ke `metadata.json`, dan membuat contoh API dengan FastAPI.

Contoh menjalankan API setelah file model, `nutriscore_utils.py`, dan folder output tersedia:

```bash
uvicorn app:app --reload
```

Dokumentasi API lokal tersedia di `http://127.0.0.1:8000/docs`.

Endpoint yang dibuat:
- `GET /health`
- `POST /predict`

## Menjalankan Proyek

### 1. Siapkan environment

Disarankan menggunakan Python 3.12.

```bash
python -m venv .venv
```

Aktifkan environment, lalu jalankan:

```bash
pip install -r requirements.txt
```

### 2. Siapkan data

Letakkan salah satu file berikut di direktori kerja notebook:

- `datawrangling_fix_supervised.db` (prioritas utama; tabel `nutriscore_master` harus tersedia), atau
- `nutriscore_master.csv` (fallback).

Notebook saat ini mencari file pada direktori kerja, `/content`, `/content/drive/MyDrive`, dan `/mnt/user-data/uploads`. Jika menggunakan struktur folder `data/`, sesuaikan fungsi `find_file()` agar juga memeriksa folder tersebut.

### 3. Jalankan notebook

Buka `notebooks/01_NutriScore_Supervised_Learning.ipynb` melalui Jupyter atau Google Colab, pastikan file data tersedia, lalu jalankan sel secara berurutan.

## Batasan Penelitian

- Dataset tidak menyediakan persentase buah, sayur, dan kacang yang relevan pada perhitungan Nutri-Score.
- Label berasal dari data crowdsourcing sehingga berpotensi mengandung noise pencatatan.
- Beberapa kategori seperti minuman, lemak/minyak, dan keju dapat mengikuti aturan khusus dan jumlah sampelnya terbatas.
- Hasil hanya berlaku pada data dan rancangan evaluasi yang digunakan; performa tidak menjamin generalisasi ke semua produk atau versi aturan Nutri-Score.
- Fitur poin yang direkayasa merupakan pendekatan berbasis fitur yang tersedia, bukan implementasi resmi penuh dari algoritma Nutri-Score.

## Pengembangan Selanjutnya

- Menambahkan informasi komposisi/ingredients untuk menangkap persentase buah, sayur, dan kacang.
- Menguji model gradient boosting tambahan dan kalibrasi probabilitas.
- Menambahkan SHAP untuk interpretasi prediksi individual.
- Memperluas validasi ke sumber data dan kelompok produk lain.

## Tech Stack

Python · Pandas · NumPy · SciPy · Scikit-learn · Matplotlib · Seaborn · SQLite · Joblib · FastAPI · Uvicorn · Jupyter Notebook

---

**Author: Dzaki Rafif Fakhriyanto**
