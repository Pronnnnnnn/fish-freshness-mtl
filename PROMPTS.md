# Prompt Bertahap untuk Claude Code

Letakkan `SPEC.md` di akar proyek terlebih dahulu, lalu commit:

```
git add SPEC.md
git commit -m "Add technical specification for the experiment"
```

Kerjakan tujuh tahap di bawah **berurutan**. Setiap tahap diakhiri satu
commit. Jangan menggabungkan dua tahap dalam satu prompt.

---

## TAHAP 0 — Pembersihan hasil lama

```
Baca SPEC.md di akar proyek terlebih dahulu.

Hasil eksperimen lama tidak lagi sah karena mekanisme pembagian data akan
diubah dari tingkat citra ke tingkat kelompok waktu. Kosongkan isi
05_Checkpoints/ dan 06_Results/, serta hapus manifest dan split lama di
02_Manifests/, tetapi pertahankan foldernya beserta .gitkeep bila ada.

Jangan menyentuh 04_Src/, 03_Notebooks/, atau 01_Dataset/ pada tahap ini.

Setelah itu tunjukkan isi ketiga folder tersebut supaya saya bisa
memverifikasi.
```

Commit: `git commit -am "Clear stale checkpoints, results, and manifests"`

---

## TAHAP 1 — Kelompok waktu pada manifest

```
Baca SPEC.md, khususnya bagian 2.

Tambahkan kolom time_group ke manifest yang dihasilkan build_manifest()
di 04_Src/manifest_utils.py, mengikuti aturan pada SPEC.md bagian 2.

Yang penting:
- Kunci kelompok dibentuk dari nama folder + IMG_YYYYMMDD_HHMMSS, sufiks
  angka setelah detik dibuang.
- Ada tepat 3 berkas di dataset yang namanya tidak mengikuti pola tersebut.
  Berkas seperti itu harus diberi kelompok tersendiri berisi satu citra,
  dan kode tidak boleh gagal karenanya.

Tambahkan juga fungsi untuk meringkas statistik kelompok: jumlah kelompok
unik, jumlah kelompok yang berisi lebih dari satu citra, rata-rata citra per
kelompok jamak, dan persentase citra yang berada dalam kelompok jamak.

Jangan mengubah kolom lain yang sudah ada dan jangan menyentuh berkas lain.
```

**Verifikasi sebelum commit.** Jalankan pembuatan manifest, lalu pastikan
angkanya cocok dengan SPEC.md bagian 2: 2.841 kelompok unik, 474 kelompok
jamak, sekitar 46% citra dalam kelompok jamak, total 4.390 citra.

Commit: `git commit -am "Add time_group column to the manifest"`

---

## TAHAP 2 — Pembagian data tingkat kelompok

```
Baca SPEC.md, khususnya bagian 3.

Ganti isi 04_Src/split_utils.py agar pembagian dilakukan pada tingkat
time_group, bukan pada tingkat citra seperti sekarang. Seluruh citra dalam
satu kelompok waktu harus masuk ke subset yang sama.

Ikuti seluruh ketentuan pada SPEC.md bagian 3, termasuk:
- rasio 70/15/15, stratifikasi atas combined_class
- bilangan pengacak pembagian 2026, terpisah dari seed pelatihan
- hasil disimpan sekali ke 02_Manifests/split_manifest.csv lalu dibaca ulang

Tambahkan fungsi verifikasi yang memeriksa ketiga hal pada SPEC.md bagian 3:
tidak ada time_group yang muncul di lebih dari satu subset, ke-24 kombinasi
kelas terwakili di ketiga subset, dan total citra 4.390. Fungsi ini harus
melempar error bila salah satu tidak terpenuhi.

Sesuaikan juga notebook 03_Notebooks/02_train_val_test_split.ipynb agar
memakai fungsi baru ini dan menjalankan verifikasinya.
```

**Verifikasi sebelum commit.** Jalankan notebook 02 sampai selesai, pastikan
verifikasinya lolos, dan periksa proporsi citra akhir tidak jauh dari
70/15/15. Selisih beberapa persen wajar.

Commit: `git commit -am "Split at time-group level to prevent near-duplicate leakage"`

---

## TAHAP 3 — Reproducibility

```
Baca SPEC.md, khususnya bagian 8.

Lengkapi set_seed() di 04_Src/train.py agar menyentuh seluruh sumber
keacakan sesuai SPEC.md bagian 8, termasuk mode deterministik cuDNN.

Berikan generator ber-seed dan worker_init_fn pada DataLoader di
_make_loaders(), serta generator pada WeightedRandomSampler di
make_balanced_sampler() (04_Src/dataset.py).

Pastikan seed pelatihan tidak menyentuh pembagian data dengan cara apa pun.

Jangan mengubah nilai hyperparameter apa pun.
```

Commit: `git commit -am "Seed all randomness sources for reproducible runs"`

---

## TAHAP 4 — Model C dan pemetaan 24 kelas

```
Baca SPEC.md, khususnya bagian 4 dan 9.

Tambahkan Model C, yaitu model berkepala tunggal dengan 24 keluaran yang
memprediksi kombinasi spesies dan kesegaran sekaligus. Model ini adalah
pembanding, bukan skema yang diteliti.

Yang perlu ditambahkan:
1. Kelas model di 04_Src/models.py, memakai backbone dan susunan kepala
   yang sama dengan model lain, hanya berbeda jumlah keluaran.
2. Kolom combined_idx di manifest, dibentuk dengan rumus pada SPEC.md
   bagian 4. Perhatikan bahwa kesegaran menempati digit satuan agar urutan
   ordinalnya terjaga. Jangan membentuknya dari urutan alfabetis.
3. Fungsi pemetaan balik dari combined_idx menjadi (species_idx, freshness_idx).
4. Fungsi pelatihan train_flat24 di 04_Src/train.py, mengikuti pola
   train_single_task yang sudah ada.

Kriteria pemantauan Model C mengikuti SPEC.md bagian 9: rata-rata tak
berbobot F1 makro spesies dan kesegaran SETELAH pemetaan balik, bukan
F1 24 kelas.

Jangan mengubah Model A, B, maupun MTL pada tahap ini.
```

**Verifikasi sebelum commit.** Periksa bahwa pemetaan baliknya benar,
misalnya `combined_idx=5` menghasilkan `species_idx=1, freshness_idx=2`.

Commit: `git commit -am "Add Model C flat 24-class baseline with label mapping"`

---

## TAHAP 5 — Weight decay UW dan verifikasi loss

```
Baca SPEC.md, khususnya bagian 7.

Di train_multitask() pada 04_Src/train.py, parameter log-varians
UncertaintyWeighting saat ini menerima weight decay yang sama dengan bobot
jaringan. Ini harus dikecualikan menggunakan parameter group terpisah
seperti contoh pada SPEC.md bagian 7.

Pastikan pengecualian ini hanya berlaku bagi parameter strategi loss, dan
seluruh bobot jaringan tetap memakai weight decay 1e-5.

Model A, B, C, D-EW, dan D-DWA tidak memiliki parameter tambahan sehingga
tidak terpengaruh.

Setelah itu, verifikasi bahwa implementasi ketiga strategi loss di
04_Src/loss_weighting.py sudah sesuai SPEC.md bagian 6 dan 7, khususnya
nilai awal log-varians 0 dan temperatur DWA 2,0. Laporkan bila ada yang
tidak sesuai, tetapi jangan mengubah nilainya tanpa saya konfirmasi.
```

**Verifikasi sebelum commit.** Cetak `optimizer.param_groups` dan pastikan
ada dua grup dengan `weight_decay` 1e-5 dan 0,0.

Commit: `git commit -am "Exclude UW log-variance parameters from weight decay"`

---

## TAHAP 6 — Metrik, penyimpanan hasil, dan perbandingan

```
Baca SPEC.md, khususnya bagian 5, 10, 11, dan 12.

Tiga hal yang perlu dikerjakan.

Pertama, simpan logit mentah data uji ke 06_Results untuk setiap model dan
seed, sesuai SPEC.md bagian 5, agar probabilitas dapat dihitung belakangan
tanpa mengulang pengujian.

Kedua, ubah penyimpanan hasil ke format tabel panjang sesuai SPEC.md bagian
11, sehingga nilai per seed tersimpan utuh dan bukan hanya rata-ratanya.

Ketiga, tulis ulang 04_Src/comparison.py mengikuti SPEC.md bagian 12.
Hapus fungsi exceeds_seed_variation beserta seluruh pemakaiannya, dan ganti
dengan perhitungan selisih berpasangan antar-seed. Kriteria pelaporannya
adalah konsistensi tanda pada ketiga seed, tanpa ambang numerik apa pun.

Sediakan ketiga tahap perbandingan pada SPEC.md bagian 12, termasuk
perbandingan Model C terhadap ketiga varian MTL yang sebelumnya belum ada.

Sesuaikan pula notebook 04 dan 05 agar memakai fungsi baru ini.
```

Commit: `git commit -am "Store per-seed results and compare via paired differences"`

---

## TAHAP 7 — Waktu inferensi dan Grad-CAM

```
Baca SPEC.md, khususnya bagian 13 dan 14.

Dua hal yang perlu dikerjakan.

Pertama, sesuaikan measure_inference_time_ms() di 04_Src/evaluate.py agar
mengikuti seluruh ketentuan SPEC.md bagian 13, terutama bahwa yang diukur
hanya satu forward pass dengan citra yang sudah berada di memori GPU, serta
pelaporan rata-rata beserta simpangan baku. Tambahkan perhitungan rasio
penghematan terhadap gabungan Model A dan Model B.

Kedua, sesuaikan 04_Src/gradcam_utils.py dan notebook 06 agar mengikuti
SPEC.md bagian 14, terutama:
- pemilihan instans seed berdasarkan Joint Accuracy pada data VALIDASI,
  dengan mengambil instans yang paling mendekati nilai tengah, bukan yang
  tertinggi, dan bukan berdasarkan data uji
- pengambilan 48 citra secara acak berstrata, 2 citra per kombinasi kelas
- target layer berupa peta fitur spasial terakhir sebelum global average
  pooling

Catat nomor seed yang dipakai ke dalam keluaran agar analisis dapat diulang.
```

Commit: `git commit -am "Fix inference timing protocol and Grad-CAM sampling"`

---

## Setelah ketujuh tahap selesai

Sebelum menjalankan 18 pelatihan penuh, lakukan uji cepat:

```
Baca SPEC.md.

Jalankan uji cepat menyeluruh memakai subset kecil, sekitar 100 citra dan
2 epoch saja, melewati seluruh alur untuk keenam konfigurasi model: pemuatan
data, pelatihan, evaluasi metrik, penyimpanan hasil, perbandingan, dan satu
kali Grad-CAM.

Tujuannya hanya memastikan tidak ada error, bukan menghasilkan angka yang
bermakna. Laporkan bila ada tahap yang gagal.

Jangan mengubah hyperparameter apa pun untuk uji ini selain jumlah epoch
dan ukuran subset, dan jangan menimpa berkas hasil yang sebenarnya.
```

Kuota Colab terbatas, sehingga menemukan kesalahan pada run ke-14 dari 18
jauh lebih mahal daripada menemukannya dalam uji dua menit.

Setelah uji cepat lolos, jalankan Model A dan B terlebih dahulu (6 run),
ukur selisih tingkat kesulitan kedua tugas, baru lanjutkan ke Model C dan
ketiga varian D.
