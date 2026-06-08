# Analisis Data Kategorikal: Pengaruh Pendapatan Keluarga dan Kondisi Sanitasi terhadap Status Stunting pada Balita

Repositori ini memuat berkas proyek, kode pemrograman, dan laporan ilmiah untuk memenuhi tugas mata kuliah **Analisis Data Kategorikal**. Penelitian ini berfokus pada pemodelan statistik variabel dependen nominal dikotomus (stunting) menggunakan variabel independen skala kontinu (pendapatan keluarga dan skor sanitasi lingkungan).

## Identitas Mahasiswa
- **Nama**: Anwar Rohmadi
- **NIM**: 247411027
- **Mata Kuliah**: Analisis Data Kategorikal
- **Tautan Repositori**: [https://github.com/anwarrohmadi2006/analisis-data-kategorikal-stunting](https://github.com/anwarrohmadi2006/analisis-data-kategorikal-stunting)

---

## Struktur Repositori
Repositori ini disusun secara sistematis dengan struktur berkas sebagai berikut:
```text
├── analisis_stunting.ipynb       # Jupyter Notebook interaktif (kode + visualisasi)
├── README.md                     # Panduan repositori akademik
├── .gitignore                    # Konfigurasi pengabaian berkas temporary
└── extracted/
    ├── analisis_stunting.py      # Kode program Python utama (.py)
    ├── laporan_analisis_stunting.md  # Naskah laporan format Markdown
    ├── laporan_analisis_stunting.pdf # Laporan resmi format PDF (Times New Roman, Justified, KaTeX)
    └── [Visualisasi & Gambar Output...]
```

---

## Metodologi Analisis
Penelitian ini menerapkan beberapa metode statistika untuk data kategorikal:

1. **Analisis Deskriptif & Uji Asosiasi**:
   - Statistika deskriptif pada masing-masing kelompok status stunting (mean, standar deviasi, nilai minimum, dan nilai maksimum).
   - Uji ukuran asosiasi menggunakan **Point Bi-Serial Correlation Coefficient** ($r_{pb}$) untuk mengukur kekuatan hubungan nominal-kontinu.

2. **Model Regresi Logistik Biner**:
   - **Model A (Income)**:
     $$\text{logit}(P(\text{Stunting}=1)) = \ln\left(\frac{P(\text{Stunting}=1)}{1 - P(\text{Stunting}=1)}\right) = \beta_0 + \beta_1 \cdot \text{Income}$$
   - **Model B (Sanitation)**:
     $$\text{logit}(P(\text{Stunting}=1)) = \ln\left(\frac{P(\text{Stunting}=1)}{1 - P(\text{Stunting}=1)}\right) = \beta_0 + \beta_1 \cdot \text{Sanitation}$$

3. **Model Probit (Pembanding)**:
   - Estimasi model dengan link function probit $\Phi^{-1}(p)$ untuk membandingkan kecocokan model berdasarkan metrik informasi:
     $$P(\text{Stunting}=1) = \Phi(\beta_0 + \beta_1 \cdot X)$$

4. **Metrik Evaluasi**:
   - Uji Likelihood Ratio ($G^2$) untuk signifikansi model secara serentak.
   - *Pseudo R-Square* (McFadden).
   - Matriks Klasifikasi (*Confusion Matrix*) untuk mengukur akurasi, presisi, sensitivitas, dan spesifisitas.
   - Kurva karakteristik operasi penerima (*ROC Curve*) dan luasan di bawah kurva (*AUC*).

---

## Persyaratan Sistem & Dependensi
Pemodelan ini dibangun menggunakan bahasa pemrograman Python 3 dengan dependensi utama:
- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## Petunjuk Menjalankan Proyek
### Menjalankan Jupyter Notebook (.ipynb)
1. Buka Jupyter Notebook/Lab lokal Anda.
2. Buka berkas `analisis_stunting.ipynb`.
3. Jalankan sel kode secara berurutan untuk melihat hasil analisis dan grafik visualisasi interaktif.

### Menjalankan Script Python (.py)
Eksekusi langsung script python melalui terminal atau command prompt:
```bash
python extracted/analisis_stunting.py
```
