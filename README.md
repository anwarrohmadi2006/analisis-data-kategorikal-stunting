# Analisis Data Kategorikal - Hubungan Pendapatan & Sanitasi terhadap Stunting

Repositori ini berisi laporan analisis data kategorikal, kode pemrograman Python, dan Jupyter Notebook mengenai pengaruh pendapatan keluarga (*Income*) dan kondisi sanitasi (*Sanitation*) terhadap status stunting pada balita.

## Identitas Penulis
- **Nama**: Anwar Rohmadi
- **NIM**: 247411027
- **Mata Kuliah**: Analisis Data Kategorikal dengan Python
- **Topik**: Regresi Logistik & Analisis Data Biner

## Daftar Isi Repositori
1. **[analisis_stunting.ipynb](analisis_stunting.ipynb)**: Jupyter Notebook yang memuat seluruh proses analisis data dari pemuatan data, uji asosiasi (korelasi Point Bi-Serial), pemodelan Regresi Logistik (Model A & Model B), Probit Model pembanding, hingga visualisasi hasil.
2. **[extracted/analisis_stunting.py](extracted/analisis_stunting.py)**: Kode pemrograman Python mandiri yang dapat dijalankan secara langsung.
3. **[extracted/laporan_analisis_stunting.md](extracted/laporan_analisis_stunting.md)**: Dokumen laporan analisis stunting format Markdown lengkap dengan rumus LaTeX.
4. **[extracted/laporan_analisis_stunting.pdf](extracted/laporan_analisis_stunting.pdf)**: Dokumen laporan resmi format PDF siap cetak dengan font Times New Roman, teks rata kanan-kiri (justified), layout bersih bebas dari garis pembatas horizontal (strips), dan persamaan matematika ter-render dengan sempurna.

## Metode Analisis
- **Uji Korelasi Point Bi-Serial**: Mengukur hubungan asosiasi antara variabel independen kontinu dengan variabel dependen biner.
- **Regresi Logistik Biner**: Pemodelan peluang stunting menggunakan fungsi link logit.
- **Regresi Probit**: Model alternatif pembanding dengan fungsi link probit (distribusi normal standar).
- **ROC-AUC**: Mengukur kemampuan diskriminasi dan evaluasi performa model klasifikasi.

## Cara Menjalankan Kode
Anda dapat menjalankan script python secara mandiri:
```bash
python extracted/analisis_stunting.py
```
Atau membuka notebook `analisis_stunting.ipynb` di Jupyter Lab / Google Colab.
