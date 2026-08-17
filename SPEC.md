# SPEC.md — Spesifikasi Teknis Eksperimen

Berkas ini adalah sumber kebenaran tunggal untuk implementasi. Setiap
keputusan di bawah sudah dikunci pada proposal Tugas Akhir dan **tidak boleh
diubah** tanpa instruksi eksplisit. Jika ada konflik antara kode dan berkas
ini, berkas ini yang benar.

---

## 0. Aturan yang berlaku untuk semua perubahan

- **Jangan mengubah nilai hyperparameter apa pun.** Seluruh nilai pada
  bagian 6 adalah variabel kontrol penelitian.
- **Jangan mengubah struktur folder** (`01_Dataset`, `02_Manifests`,
  `03_Notebooks`, `04_Src`, `05_Checkpoints`, `06_Results`, `07_Paper`).
- **Jangan menambah dependensi baru** di luar yang sudah ada di
  `requirements.txt` kecuali diminta.
- **Jangan merapikan atau merefaktor kode yang tidak diminta.**
- Komentar dan docstring ditulis dalam bahasa Inggris, mengikuti gaya
  yang sudah ada.

---

## 1. Dataset dan lingkungan

- Dataset FFE: 4.390 citra, 8 spesies × 3 tingkat kesegaran = 24 kombinasi.
- Sumber: Google Drive, folder `fish-freshness-mtl`, berisi
  `8_fish_3_freshness.zip`. Zip diekstrak ke `01_Dataset` saat runtime Colab.
- Struktur folder di dalam zip: satu folder per kombinasi, dinamai
  `"Species - Freshness"`, contoh `"Chanos Chanos - Fresh"`.
- Catatan: satu folder memuat spasi ganda (`"Nibea Albiflora -  Highly Fresh"`).
  Sudah ditangani `parse_folder_name`.

---

## 2. Penamaan berkas dan kelompok waktu (WAJIB)

Nama berkas mengikuti pola kamera Android: `IMG_YYYYMMDD_HHMMSS.jpg`,
sebagian dengan sufiks angka: `IMG_YYYYMMDD_HHMMSS_N.jpg`.

**Fakta terverifikasi dari dataset** (dihitung ulang langsung dari
`01_Dataset/8_fish_3_freshness`, aturan kunci = folder + prefiks
`IMG_YYYYMMDD_HHMMSS`):

- 3.226 kelompok waktu unik.
- 491 kelompok memuat lebih dari satu citra, rata-rata 3,37 citra/kelompok.
- Kelompok jamak mencakup 1.655 citra atau 37,7% dataset.
- **3 berkas tidak mengikuti pola** (2 di `Oreochromis Mossambicus - Highly
  Fresh`, 1 di `Oreochromis Niloticus - Fresh`). Ketiganya berstempel waktu
  lima digit, bukan enam: `IMG_20191110_05505_6`, `IMG_20191111_05504_31`,
  `IMG_20191112_05410_41`.

> Catatan revisi: draf awal SPEC mencantumkan 2.841 kelompok / 474 jamak /
> 2.019 citra (46,0%). Angka tersebut tidak dapat direproduksi dengan aturan
> mana pun yang diuji (per-detik maupun per-menit) dan tidak konsisten secara
> internal: 2.841 − 474 = 2.367 kelompok tunggal, sehingga 2.367 + 2.019 =
> 4.386, bukan 4.390. Angka di atas adalah hasil hitung ulang dan itulah yang
> dipakai sebagai sasaran verifikasi. Klaim "3 berkas menyimpang" terverifikasi
> tepat, termasuk lokasi foldernya.

**Dampak yang terukur pada pembagian lama (tingkat citra):** 311 kelompok
waktu terbelah antar-subset, 173 di antaranya terbelah antara latih dan uji,
sehingga **222 citra uji (33,7%) memiliki kembaran di data latih**. Inilah
alasan seluruh hasil sebelum perbaikan ini dinyatakan tidak sah.

**Aturan pembentukan `time_group`:**

1. Kunci kelompok = `folder + "/" + IMG_YYYYMMDD_HHMMSS` (sufiks dibuang).
2. Berkas yang namanya **tidak terurai** oleh pola tersebut diberi kelompok
   tersendiri berisi satu citra (gunakan nama berkas lengkap sebagai kunci).
   Kode **tidak boleh gagal** karena ketiga berkas ini.
3. Satu kelompok waktu selalu berada dalam satu kombinasi kelas, sehingga
   stratifikasi tetap dilakukan atas `combined_class`.

---

## 3. Pembagian data (WAJIB, paling kritis)

- Rasio 70% latih, 15% validasi, 15% uji.
- **Pembagian dilakukan pada tingkat `time_group`, BUKAN tingkat citra.**
  Seluruh citra dalam satu kelompok masuk ke subset yang sama.
- Stratifikasi atas `combined_class` (24 kelas).
- **Dibuat SATU KALI**, disimpan ke `02_Manifests/split_manifest.csv`,
  lalu dibaca ulang oleh seluruh pelatihan. Jangan pernah membuat ulang
  pembagian di dalam loop pelatihan.
- **Bilangan pengacak pembagian = 2026**, terpisah dan berbeda dari seed
  pelatihan (42, 43, 44). Ini wajib: jika sama, komposisi subset ikut
  berubah antar-seed dan seluruh perbandingan menjadi tidak sah.
- Karena ukuran kelompok berbeda-beda, proporsi citra akhir tidak akan
  tepat 70/15/15. Selisih beberapa persen wajar dan tidak perlu dipaksakan.

**Verifikasi wajib setelah pembagian dibuat:**
- Tidak ada satu pun `time_group` yang muncul di lebih dari satu subset.
- Ke-24 kombinasi kelas terwakili di ketiga subset.
- Total citra = 4.390.

---

## 4. Enam konfigurasi model

| Model | Skema | Kepala | Strategi loss |
|-------|-------|--------|---------------|
| A | Single-task | 1 × 8 (spesies) | — |
| B | Single-task | 1 × 3 (kesegaran) | — |
| C | Flat 24 kelas | 1 × 24 (gabungan) | — |
| D-EW | Multi-task | 2 (8 + 3) | Equal Weighting |
| D-UW | Multi-task | 2 (8 + 3) | Uncertainty Weighting |
| D-DWA | Multi-task | 2 (8 + 3) | Dynamic Weight Averaging |

6 konfigurasi × 3 seed = **18 run**.

**Model C — pemetaan label 24 kelas:**

```
combined_idx = species_idx * 3 + freshness_idx
species_idx  = combined_idx // 3
freshness_idx = combined_idx % 3
```

Kesegaran menempati digit satuan agar urutan ordinalnya (0=Highly Fresh,
1=Fresh, 2=Not Fresh) terjaga setelah pemetaan balik — syarat agar QWK
bermakna. **Jangan** membentuk indeks 24 kelas dari urutan alfabetis
`combined_class`.

Prediksi Model C dipetakan balik menjadi prediksi spesies dan kesegaran
sebelum evaluasi, sehingga seluruh metriknya dihitung dengan cara yang
persis sama dengan model MTL.

---

## 5. Keluaran model

- `classification head` menghasilkan **logit mentah**, tanpa Softmax.
  `nn.CrossEntropyLoss` sudah memuat `log_softmax` di dalamnya; menambahkan
  Softmax akan menerapkannya dua kali.
- Prediksi kelas = `argmax` atas logit.
- Softmax hanya dipakai bila probabilitas benar-benar dibutuhkan
  (saat ini tidak ada metrik yang membutuhkannya).
- **Simpan logit mentah data uji ke `06_Results`** agar probabilitas dapat
  dihitung belakangan tanpa mengulang pengujian.
- Grad-CAM mengambil gradien dari **logit**, bukan probabilitas.

---

## 6. Hyperparameter (variabel kontrol — JANGAN DIUBAH)

| Parameter | Nilai |
|-----------|-------|
| Backbone | `tf_efficientnetv2_b0` (timm), pretrained ImageNet |
| Strategi transfer | Fine-tuning seluruh lapisan, tanpa pembekuan |
| Ukuran masukan | 224 × 224 × 3 |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Laju pembelajaran awal | 1e-4 |
| Maksimum epoch | 100 |
| Ukuran batch | 32 |
| Dropout | 0,2 |
| Weight decay | 1e-5 |
| **Weight decay parameter UW** | **0** (lihat bagian 7) |
| **Temperatur DWA (T)** | **2,0** |
| Early stopping patience | 15 epoch |
| Random seed pelatihan | 42, 43, 44 |
| Random seed pembagian data | 2026 |

**Augmentasi (hanya data latih, daring per pengambilan):**
- `RandomHorizontalFlip(p=0.5)`
- `RandomRotation(degrees=15)`
- `ColorJitter(brightness=0.2, contrast=0.2)`
- Tidak ada pembalikan vertikal, tidak ada penggeseran rona.
- Data validasi dan uji hanya `Resize(224,224)` + `Normalize`.

**Penyeimbangan kelas:** `WeightedRandomSampler` dengan bobot berbanding
terbalik terhadap frekuensi `combined_class` (24 kombinasi), `replacement=True`,
`num_samples=len(train_df)`. **Hanya pada data latih.** Validasi dan uji
mengikuti distribusi asli.

---

## 7. Weight decay untuk Uncertainty Weighting (WAJIB)

Parameter log-varians UW (`log_var_species`, `log_var_freshness`) **harus
dikecualikan dari weight decay** menggunakan parameter group terpisah:

```python
optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "weight_decay": cfg.weight_decay},
    {"params": loss_strategy.parameters(), "weight_decay": 0.0},
], lr=cfg.lr)
```

Alasan: fungsi objektif UW sudah memuat suku regularisasi `s1 + s2`.
Jika weight decay ikut dikenakan, kedua parameter tertarik ke nol dari dua
arah sekaligus, bobot kedua tugas terdorong menjadi seragam bukan karena
mekanisme UW, dan UW menyerupai EW secara semu. Temuan "ketiganya setara"
akan menjadi artefak kode, bukan temuan penelitian.

Nilai awal `log_var` = 0 (bobot awal = 1), sehingga D-UW berangkat dari
titik yang sama dengan D-EW.

Model A, B, C, D-EW, dan D-DWA tidak memiliki parameter tambahan, jadi
parameter group tunggal sudah cukup.

---

## 8. Reproducibility (WAJIB)

`set_seed(seed)` harus menyentuh seluruh sumber keacakan:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

DataLoader harus menerima `generator` ber-seed dan `worker_init_fn` agar
urutan pengambilan sampel identik pada pengulangan yang sama.
`WeightedRandomSampler` juga harus menerima `generator`.

Seed pelatihan **hanya** mengubah inisialisasi bobot, pengacakan sampler,
dan urutan DataLoader. Seed pelatihan **tidak boleh** menyentuh pembagian
data.

---

## 9. Kriteria early stopping dan pemilihan checkpoint

Bobot yang disimpan adalah bobot pada epoch dengan nilai pemantau
**validasi** tertinggi.

| Model | Metrik pemantau |
|-------|-----------------|
| A | F1 makro spesies |
| B | F1 makro kesegaran |
| C | rata-rata tak berbobot F1 makro spesies dan kesegaran, **setelah pemetaan balik 24 → (spesies, kesegaran)** |
| D-* | rata-rata tak berbobot F1 makro kedua kepala |

Model C **tidak** dipantau dengan F1 24 kelas. Penyeragaman ini diperlukan
agar pemilihan bobot pada Model C dan D didasarkan pada besaran yang sama,
sebab keduanya nanti dibandingkan pada metrik yang sama.

*Joint Accuracy* **tidak** dipakai sebagai kriteria pemilihan, karena secara
bawaan menguntungkan rumusan label gabungan. Ia tetap dilaporkan sebagai
metrik hasil.

Kriteria ini ditetapkan di muka dan tidak boleh diubah setelah melihat hasil.

---

## 10. Metrik evaluasi

Dihitung pada data uji, untuk setiap model dan setiap seed.

**Tugas spesies (nominal, 8 kelas):** accuracy, precision makro, recall makro,
F1 makro, MCC, Cohen's Kappa.

**Tugas kesegaran (ordinal, 3 kelas):** accuracy, precision makro, recall makro,
F1 makro, MCC, **QWK** (menggantikan Cohen's Kappa).

**Simultan:** Joint Accuracy.

**Efisiensi:** jumlah parameter, waktu inferensi.

Model A dan B menghasilkan metrik satu tugas. Model C dan D-* menghasilkan
metrik kedua tugas plus Joint Accuracy.

Matriks konfusi dibuat per tugas untuk melihat letak kesalahan, terutama
apakah dua spesies *Oreochromis* sering tertukar dan apakah kesalahan
kesegaran meloncat dua tingkat atau hanya bergeser satu tingkat.

**Hierarki pelaporan:** metrik utama adalah F1 makro dan accuracy (spesies),
F1 makro dan QWK (kesegaran), serta Joint Accuracy (simultan). Sisanya
metrik pendukung.

---

## 11. Penyimpanan hasil (WAJIB)

Hasil disimpan dalam **format tabel panjang**, satu baris per kombinasi
model × seed, ke `06_Results/`:

```
model, seed, task, metric, value
```

atau setara, yang penting **nilai per seed tersimpan utuh**, bukan hanya
rata-ratanya. Selisih berpasangan pada bagian 12 dihitung dari tabel ini.
Jika hanya rata-rata yang disimpan, informasi pasangannya hilang permanen.

---

## 12. Perbandingan model (WAJIB — menggantikan kriteria lama)

Perbandingan dilakukan melalui **selisih berpasangan antar-seed**, bukan
dengan mempertentangkan dua rata-rata beserta simpangan baku terpisah.

Karena seluruh model dilatih dengan seed yang sama (42, 43, 44) dan
dievaluasi pada data uji yang sama, untuk setiap seed dapat dihitung selisih
kinerja antara dua konfigurasi. Ketiga selisih itu dilaporkan sebagai
rata-rata beserta simpangan bakunya.

**Kriteria pelaporan:** suatu konfigurasi dilaporkan sebagai kandidat terbaik
apabila selisih berpasangannya **bertanda sama pada ketiga seed** (unggul
pada seluruh pengulangan tanpa kecuali). Besaran rata-rata dan simpangan baku
selisih dilaporkan berdampingan.

**JANGAN menetapkan ambang numerik** seperti "rata-rata selisih > simpangan
baku". Ambang semacam itu tidak punya dasar teoretis pada n=3 dan terbaca
seolah uji signifikansi. **Hapus fungsi `exceeds_seed_variation` dan seluruh
pemakaiannya.**

**Tiga tahap perbandingan:**

1. **STL vs MTL** (indikasi negative transfer)
   A vs D-EW, A vs D-UW, A vs D-DWA (akurasi spesies)
   B vs D-EW, B vs D-UW, B vs D-DWA (akurasi kesegaran)
   → 6 titik pembanding

2. **Flat 24 kelas vs MTL** (rumusan persoalan)
   C vs D-EW, C vs D-UW, C vs D-DWA
   pada akurasi spesies, akurasi kesegaran, dan Joint Accuracy
   → 3 titik pembanding

3. **Antar strategi pembobotan**
   D-EW vs D-UW vs D-DWA

Uji statistik formal (paired t-test) **tidak digunakan** meskipun selisih
berpasangan tersedia, karena n=3 membuat kekuatan statistiknya tidak memadai.

---

## 13. Protokol waktu inferensi

- GPU yang sama untuk seluruh model, `model.eval()`, `batch_size = 1`.
- Didahului beberapa iterasi pemanasan agar keadaan GPU stabil.
- `torch.cuda.synchronize()` sebelum dan sesudah pencatatan. Tanpa ini yang
  tercatat hanya waktu penjadwalan operasi, bukan waktu penyelesaiannya.
- Dilaporkan sebagai rata-rata beserta simpangan baku atas sejumlah iterasi.
- **Yang diukur hanya satu kali forward pass** dengan citra yang sudah berada
  di memori GPU. Waktu pemuatan berkas, pra-pemrosesan, dan pemindahan data
  dari memori utama ke GPU **tidak ikut dihitung**.

**Pembanding efisiensi:** Model D dibandingkan terhadap **gabungan Model A
dan Model B**, bukan salah satunya, sebab alternatif nyata dari satu model
MTL adalah sepasang model tugas tunggal yang dijalankan berurutan.

Laporkan pula rasio penghematan:
```
speedup = (t_A + t_B) / t_D
```

---

## 14. Grad-CAM

- Dijalankan sebagai **analisis lanjutan (post-hoc)** terhadap model D
  terbaik, yang baru diketahui setelah perbandingan. Tidak dipakai untuk
  menentukan model maupun menyetel parameter apa pun.
- **Pemilihan instans seed:** instans yang kinerjanya paling mendekati nilai
  tengah di antara ketiga seed menurut **Joint Accuracy pada data VALIDASI**,
  bukan data uji. Data uji harus tetap murni untuk evaluasi akhir. Nomor seed
  yang dipakai dicatat.
- **Sampel citra:** acak berstrata atas kombinasi spesies × kesegaran,
  **2 citra per kombinasi = 48 citra**. Ditetapkan sebelum eksperimen,
  bukan dipilih menurut pertimbangan peneliti.
- **Target layer:** peta fitur spasial terakhir pada backbone, yaitu keluaran
  sesaat sebelum global average pooling.
- Model dalam mode `eval()` agar dropout mati dan peta panas bersifat tetap.
- Gradien diambil dari **logit** masing-masing kepala secara terpisah,
  menghasilkan dua peta panas per citra.
- Pada MTL, `backward()` untuk kedua kepala memerlukan `retain_graph=True`
  pada backward pertama, atau forward ulang.

---

## 15. Keterbatasan yang sudah diakui (jangan "diperbaiki" oleh kode)

- Identitas individu ikan **tidak dapat direkonstruksi** dari nama berkas.
  Yang dikendalikan hanya kebocoran antar kelompok waktu. Kinerja yang
  dilaporkan mencerminkan generalisasi terhadap kelompok pemotretan yang
  tidak dilihat saat pelatihan, bukan terhadap individu ikan yang baru.
- Akuisisi dilakukan dalam tiga jendela waktu, masing-masing enam hari
  berurutan. Empat spesies muncul di dua jendela.
- Label kesegaran ditetapkan dari protokol lama penyimpanan, bukan dari
  pengukuran laboratorium (TVB-N).
