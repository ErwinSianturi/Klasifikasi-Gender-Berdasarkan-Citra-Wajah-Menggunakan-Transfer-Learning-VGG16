# Klasifikasi Gender Berdasarkan Citra Wajah Menggunakan Transfer Learning VGG16

Proyek ini merupakan bagian dari mata kuliah **Kecerdasan Buatan (CERTAN)** oleh **Kelompok 10 (Semester 6)**. Proyek ini bertujuan untuk membangun model Deep Learning yang mampu mengklasifikasikan gender (Laki-laki/Man dan Perempuan/Woman) berdasarkan citra wajah menggunakan pendekatan **Transfer Learning** dengan arsitektur **VGG16** sebagai ekstraktor fitur utama, yang dipadukan dengan arsitektur *custom* Convolutional Neural Network (CNN).

---

## 📌 Deskripsi Proyek

Klasifikasi gender melalui wajah merupakan salah satu topik penting dalam bidang *Computer Vision* dan pemrosesan citra digital. Proyek ini mengimplementasikan model klasifikasi biner untuk mendeteksi apakah citra wajah yang diinput merupakan laki-laki (`man`) atau perempuan (`woman`). 

Dengan memanfaatkan bobot yang telah dilatih sebelumnya pada dataset skala besar (ImageNet) menggunakan model **VGG16**, model ini dapat mengenali fitur-fitur wajah secara efisien meskipun dilatih dengan dataset yang relatif terbatas.

---

## 📁 Struktur Direktori Proyek

Penyusunan file dalam repositori proyek ini adalah sebagai berikut:

```text
Project-CERTAN-Kelompok-10/
├── DataSet/
│   └── Male and Female face dataset/
│       └── Link.txt             # Link Google Drive untuk mengunduh dataset asli
├── uji coba/                    # Folder berisi gambar untuk pengujian model secara mandiri
│   ├── WhatsApp Image...jpg     # Foto-foto sampel uji coba
│   ├── african-woman-634220.jpg
│   └── istockphoto-1248519461.jpg
├── Main.ipynb                   # Jupyter Notebook utama berisi pipeline pembuatan model
└── README.md                    # Dokumentasi proyek (file ini)
```

---

## 📊 Dataset

Dataset yang digunakan adalah **Male and Female Face Dataset** yang diperoleh melalui Kaggle (diunduh secara otomatis via `kagglehub` atau melalui tautan Drive yang disediakan di `DataSet/Male and Female face dataset/Link.txt`).

### Karakteristik & Pemrosesan Dataset:
1. **Saringan Ukuran Gambar**: Citra wajah difilter untuk memastikan kualitas resolusi. Hanya gambar dengan dimensi minimal **65x65 piksel** yang digunakan dalam pelatihan.
2. **Pembagian Dataset (Data Split)**:
   - Dataset dibagi menggunakan rasio **80% data latih (train)** dan **20% data validasi/uji (test)**.
   - **Data Train**: 4.352 gambar (terdiri dari kelas `man` dan `woman`).
   - **Data Test/Validation**: 1.088 gambar (terdiri dari kelas `man` dan `woman`).
   - **Total Citra yang Digunakan**: 5.440 gambar.
3. **Augmentasi Data**: 
   Untuk mencegah terjadinya *overfitting* dan meningkatkan generalisasi model, augmentasi data diterapkan pada data latih menggunakan Keras `ImageDataGenerator`:
   - Normalisasi skala piksel (`rescale=1.0/255`)
   - Rotasi acak hingga 20 derajat (`rotation_range=20`)
   - Pergeseran horizontal dan vertikal (`width_shift_range=0.2`, `height_shift_range=0.2`)
   - Distorsi shearing (`shear_range=0.2`)
   - Zoom in/out acak (`zoom_range=0.2`)
   - Pembalikan arah horizontal (`horizontal_flip=True`)
   - Penanganan area kosong dengan mode `nearest`

---

## 🏗️ Arsitektur Model

Model dibangun secara sekuensial dengan menggabungkan kekuatan ekstraksi fitur dari VGG16 dan beberapa layer klasifikasi *custom* tambahan:

```
[ Input: 64 x 64 x 3 (RGB) ]
           │
           ▼
[ Base Model: VGG16 (Pre-trained ImageNet) ]  <-- Bobot dibekukan (frozen)
           │
           ▼
[ Custom CNN Layers (Batch Normalization, Conv2D, MaxPool, Dropout) ]
           │
           ▼
[ Flatten Layer ]
           │
           ▼
[ Dense Layers: 3x (2048 unit, ReLU) dengan Dropout (0.5) ]
           │
           ▼
[ Output Layer: Dense (1 unit, Sigmoid) ]     <-- Klasifikasi Biner (Laki-laki / Perempuan)
```

### Detail Layer Custom Classifier:
Setelah layer dasar VGG16, model menambahkan serangkaian layer Convolutional tambahan yang bertahap menurun jumlah filternya (dari 128 ke 32 filter) yang bertujuan untuk mengekstrak fitur lebih spesifik pada resolusi rendah, diikuti oleh 3 layer Dense berukuran besar (2048 unit) dengan Dropout berkala (rate 0.5) untuk mencegah overfitting sebelum berakhir pada neuron output bersigmoid.

---

## ⚙️ Hyperparameter & Proses Training

Proses kompilasi dan pelatihan model menggunakan konfigurasi sebagai berikut:

* **Optimizer**: `Adam` dengan learning rate awal `0.001`
* **Loss Function**: `Binary Crossentropy` (karena klasifikasi bersifat biner)
* **Metrics**: `Accuracy`
* **Batch Size**: 64
* **Dimensi Input (Target Size)**: 64x64 piksel
* **Jumlah Epoch**: 10 epoch
* **Callbacks**:
  - `ReduceLROnPlateau`: Mengurangi learning rate secara otomatis dengan faktor pengali `0.5` apabila nilai loss tidak menunjukkan perbaikan setelah `8` epoch (patience), dengan batas minimal learning rate `0.01`.
  - `ModelCheckpoint`: Menyimpan model terbaik dengan format `model.keras` berdasarkan nilai `val_loss` terkecil sepanjang epoch berlangsung.
  - `EarlyStopping`: Menghentikan proses latihan lebih cepat jika performa validasi mengalami stagnasi.

---

## 📈 Hasil Evaluasi Training

Selama 10 epoch pelatihan, model menunjukkan peningkatan performa yang sangat baik dan konvergen:

* **Epoch 1**: `accuracy: 0.6558` | `loss: 0.6385` | `val_accuracy: 0.8125` | `val_loss: 0.4439`
* **Epoch 5**: `accuracy: 0.8329` | `loss: 0.3831` | `val_accuracy: 0.8548` | `val_loss: 0.3298`
* **Epoch 10**: `accuracy: 0.8797` | `loss: 0.2975` | `val_accuracy: 0.9272` | `val_loss: 0.2063`

### Ringkasan Hasil Akhir:
* **Akurasi Data Latih (Training Accuracy)**: **87.97%**
* **Akurasi Data Validasi (Validation Accuracy)**: **92.72%** (Performa sangat baik dengan tingkat kesalahan klasifikasi yang minim)
* **Loss Validasi Terendah**: **0.2063**

---

## 🧪 Cara Menjalankan Uji Coba (Inference)

Untuk melakukan prediksi gender pada citra wajah baru secara mandiri, Anda dapat menggunakan potongan kode Python berikut (juga diimplementasikan pada bagian akhir `Main.ipynb`):

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# 1. Muat model yang telah dilatih
model = load_model('model.keras')

# 2. Tentukan lokasi gambar uji coba
image_path = 'uji coba/nama_file_gambar.jpg'
target_size = (64, 64)

# 3. Lakukan preprocessing pada gambar
img = image.load_img(image_path, target_size=target_size)
x = image.img_to_array(img) / 255.0  # Normalisasi pixel
x = np.expand_dims(x, axis=0)        # Menambahkan dimensi batch

# 4. Lakukan prediksi
prediction = model.predict(x)

# 5. Tentukan hasil berdasarkan threshold 0.5
if prediction[0] < 0.5:
    print(f"Hasil Prediksi: Laki-laki (Skor: {prediction[0][0]:.4f})")
else:
    print(f"Hasil Prediksi: Perempuan (Skor: {prediction[0][0]:.4f})")

# 6. Tampilkan gambar
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## 👥 Anggota Kelompok 10

* **Nama Anggota Kelompok 10** - *Semester 6 - Universitas/Instansi*

---
*Proyek ini disusun dan diselesaikan sebagai tugas praktikum/akhir mata kuliah Kecerdasan Buatan (CERTAN).*
