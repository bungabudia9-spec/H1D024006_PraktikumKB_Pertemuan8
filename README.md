# Rock-Paper-Scissors Image Classification (CNN)

## Deskripsi Project
Project ini mengimplementasikan **Convolutional Neural Network (CNN)** menggunakan TensorFlow/Keras untuk mengenali dan mengklasifikasikan gambar tangan manusia menjadi tiga kategori: **Batu (Rock)**, **Kertas (Paper)**, dan **Gunting (Scissors)**. Proyek ini menggunakan teknik *image augmentation* dan pembagian dataset otomatis dengan rasio 80% training dan 20% validation.

---

## Arsitektur Model
Model dibangun menggunakan struktur **Sequential** dengan detail sebagai berikut:

* **Convolutional Layer:** 3 lapis Conv2D (32, 64, dan 128 filter) dengan aktivasi **ReLU** untuk ekstraksi fitur citra.
* **Pooling Layer:** MaxPooling2D (2x2) di setiap blok untuk mereduksi dimensi data.
* **Flatten Layer:** Mengubah matriks fitur menjadi vektor 1D.
* **Dense Layer (Hidden):** 512 neuron dengan aktivasi **ReLU**.
* **Dense Layer (Output):** 3 neuron dengan aktivasi **Softmax** untuk klasifikasi multi-kelas.

**Total Parameter:** 19,035,203 (Sekitar 72.61 MB)

---

## Hasil Evaluasi
Berdasarkan hasil training selama 10 epoch, model mencapai performa yang sangat tinggi:

* **Validation Accuracy:** 97.25%
* **Validation Loss:** 0.0846



Grafik menunjukkan model memiliki stabilitas akurasi yang sangat baik pada data validasi, dengan penurunan nilai loss yang konsisten hingga akhir epoch.

---

## Cara Menjalankan
1.  **Instalasi Dependency:**
    ```bash
    pip install tensorflow matplotlib numpy
    ```
2.  **Persiapan Dataset:** Pastikan folder dataset `rps-cv-images` berada di direktori yang sama dengan file script.
3.  **Eksekusi Program:**
    ```bash
    python praktikum8_cnn.py
    ```

---

## Identitas
* **Nama:** Bunga Budi Ambarwati
* **NIM:** H1D024006
* **Matakuliah:** Praktikum Kecerdasan Buatan (Pertemuan 8)
* **Fakultas:** Teknik (Informatika), Universitas Jenderal Soedirman
