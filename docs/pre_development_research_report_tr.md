# Geliştirme Öncesi Araştırma Raporu

**Proje:** SE3513 Machine Learning — Fake Job Posting Detection
**Veri seti:** Real / Fake Job Posting Prediction (EMSCAD), Kaggle, version 1
**Tarih:** 2026-11-25
**Güncelleme:** 2026-11-26 (Bölüm 6 ve 7, Adım 2'den sonra güncellendi)
**Amaç:** Bir sonraki geliştirme aşamasından önce bu veri seti üzerinde yapılmış herkese açık çalışmaları incelemek, güvenilir bulguları hatalı olanlardan ayırmak ve izleyeceğimiz metodolojiyi belirlemek.

---

## 1. Özet

Veri setinin Kaggle Code sayfasında en üst sırada yer alan 11 public notebook'u, hem kodlarını hem de raporladıkları çıktıları okuyarak inceledik. Temel bulgular şunlardır:

1. **Öne çıkarılan skorların çoğu güvenilir değil.** On bir notebook'tan dördü, data leakage veya implementation bug'ları nedeniyle şişirilmiş sonuçlar raporluyor (raporlanan fake-class F1 0.93 ile 0.9996 arasında). Diğer birçoğu yalnızca accuracy raporluyor; %4.8'lik bir pozitif oranda bu metrik bilgi vermez.
2. **Train/test ayrımı doğru yapılmış notebook'lar yaklaşık 0.74–0.82 fake-class F1'e ulaşıyor** ve bunu near-duplicate ilanların ayıklanmadığı *random* bir split üzerinde yapıyor. Bu senaryo bizimkinden daha kolaydır.
3. **Mevcut modelimiz (grouped, de-duplicated test setinde fake-class F1 0.80, ROC AUC 0.973)**, daha katı bir değerlendirme altında, en iyi temiz public sonuçlarla **zaten rekabet edebilir düzeyde**.
4. Bu notebook'lardaki bazı fikirler kendi kurulumumuzda test edilmeye değer (Bölüm 5). Bazı uygulamalardan ise kaçınılmalı (Bölüm 4).
5. Kendi sonuçlarımız **validation'da yalnızca 128 fake ilan bulunan tek bir split'ten** geliyor. Deneyler arasındaki birkaç F1 puanlık farklar gürültü sınırları içinde kalıyor. Bu nedenle bir sonraki aşama, değerlendirme protokolünü güçlendirerek başlıyor (Bölüm 6).

---

## 2. Yöntem Seçimini Etkileyen Veri Seti Özellikleri

| Özellik | Değer | Sonuç |
|---|---|---|
| Satır sayısı (ham / de-duplication sonrası) | 17,880 / 17,598 | Küçük veri seti; klasik linear modeller güçlü baseline olmaya devam ediyor |
| Fake ilanlar | 866 (%4.84) | Ciddi class imbalance; accuracy kullanışlı bir metrik değil |
| Near-duplicate gruplar (description TF-IDF üzerinde cosine radius 0.10) | 12,372 grup, en büyüğü 414 satır | Random split, aynı ilanın kopyalarını hem train'e hem test'e yerleştirir |
| Missing value'lar | `salary_range` %84, `department` %65, `required_education` %45, `benefits` %40 | Bir alanın boş olması kendi başına sinyal taşıyabilir |
| Maskelenmiş token'lar | Metin `#URL_<hash>#` gibi placeholder'lar içeriyor | Genellikle cleaning sırasında siliniyor; bunun yerine feature olarak sayılabilir |
| Kaynak ve dönem | Tek kaynak, 2012–2014 | Güncel ilanlara genellenebilirlik garanti değil |

---

## 3. Public Notebook'ların İncelenmesi

Notebook'lar yalnızca numarayla, sıralamadaki yerlerine göre belirtilmiştir. "Temiz" ifadesi train/test ayrımının doğru yapıldığı anlamına gelir; near-duplicate'lerin ele alındığı anlamına gelmez.

| # | Yaklaşım | Raporlanan fake-class sonucu | Değerlendirme |
|---|---|---|---|
| 1 | Bag-of-words / TF-IDF + Multinomial NB; dense NN | NB F1 0.33 ve 0.00; NN F1 0.64 | Başlıkta %97 accuracy öne çıkarılıyor. `TfidfVectorizer(max_df=1)` bir integer alıyor ve bu yalnızca tek bir dokümanda geçen terimleri tutuyor; bu bir implementation bug'ı. |
| 2 | CountVectorizer + LR; GloVe + NN; frozen DistilBERT + LR | ROC AUC 0.85 / 0.81 / 0.77 | ROC AUC, skorlar yerine 0/1 hard prediction'lardan hesaplanmış. DistilBERT yalnızca ilk 2,000 satıra, 60 token'a truncate edilerek uygulanmış. **İyi uygulama:** mean ± std ile stratified 5-fold CV. |
| 3 | Description üzerinde Bi-LSTM | Raporlanmamış | Metin 40 token'a truncate edilmiş, class weighting yok, yalnızca tek bir metin alanı kullanılmış. |
| 4 | Tüm metin alanları + `location` üzerinde 1–3-gram bag-of-words; LR / RF / SVC / XGBoost | LR recall 0.72 | En iyi basit ve temiz sonuç. De-duplication olmadan random split. |
| 5 | Bag-of-words + POS tag'leri; çeşitli classifier'lar | F1 0.9996 | **Leakage:** minority class, train/test split'ten *önce* majority class'ın boyutuna (13,937 satır) kadar with replacement oversample edilmiş; bu yüzden aynı ilanın kopyaları her iki tarafta da yer alıyor. |
| 6 | TF-IDF + Multinomial NB | F1 0.49 (recall 0.32) | Vectoriser, split'ten önce tüm veri üzerinde fit edilmiş (hafif leakage). |
| 7 | Alan bazlı TF-IDF + uzunluk/kelime sayısı feature'ları + one-hot categorical'lar; LR | ROC AUC 0.73 | ROC AUC hard prediction'lardan hesaplanmış. Kelime sayısı feature'ında bug var (`len(str(x.split()))` listenin string temsilinin karakterlerini sayıyor). **İyi fikirler:** başlık için ayrı vectoriser ve metin uzunluğu feature'ları. |
| 8 | Label-encode edilmiş metin kolonları; LR / KNN / RF | Yalnızca accuracy | **Leakage:** oversampling ve birleştirme sonrasında satır index'i feature olarak kullanılmış, bu yüzden index label'ı kodluyor. Serbest metin integer olarak label-encode edilmiş. |
| 9 | TF-IDF + SMOTE + RF; LSTM; GRU; Word2Vec + Bi-LSTM | F1 0.79 / 0.76 / 0.74 / 0.82 | **İncelenenler arasındaki en temiz metodoloji:** önce stratified split, resampling yalnızca training verisine uygulanmış. De-duplication olmadan random split. |
| 10 | Class bazlı kelime frekansı feature'ları + RF | Recall 0.83 | **Target leakage:** class bazlı kelime frekansları (ve NB olasılıkları), split'ten önce test label'ları dahil tüm veri seti üzerinde hesaplanmış. |
| 11 | GloVe ortalamaları + label-encode edilmiş categorical'lar; XGBoost + SMOTETomek | CV F1 0.93 | **Bug'lar ve leakage:** `str(data['text'])` tüm kolonu tek bir string'e dönüştürüyor, bu yüzden her satır aynı metni alıyor. Hesaplanan GloVe feature'ları modele hiç verilmiyor. SMOTETomek cross-validation'dan önce uygulanmış, bu yüzden sentetik örnekler fold'lar arasında sızıyor. |

### Mevcut modelimizle karşılaştırma

| Kurulum | Split | Fake-class F1 |
|---|---|---|
| Temiz public notebook'lar (#4, #9) | Random, near-duplicate'ler ayıklanmamış | ~0.74–0.82 |
| Bizim class-weighted Linear SVM'imiz (TF-IDF + structured features) | Description benzerliğine göre grouped, de-duplicated | **0.80** (precision 0.86, recall 0.75) |

---

## 4. Kaçınacağımız Uygulamalar

| Gözlemlenen hata | Bu proje için kural |
|---|---|
| Split'ten önce oversampling / SMOTE (#5, #8, #11) | Her türlü resampling yalnızca training fold'u içinde, bir `Pipeline` içinde yapılır. |
| Vectoriser'ları fit etmek veya label istatistiklerini tüm veri üzerinde hesaplamak (#6, #10) | Fit edilen her adım (TF-IDF, scaler'lar, encoder'lar, frekans tabloları) yalnızca training fold'u üzerinde fit edilir. |
| 0/1 prediction'lardan ROC AUC (#2, #7) | Ranking metrikleri `decision_function` veya `predict_proba` skorlarından hesaplanır. |
| Ana metrik olarak accuracy (#1, #5, #8) | Ana metrikler fake-class PR-AUC, F1, precision ve recall'dur. |
| Near-duplicate'ler içeren random split (tümü) | Grouped split korunur; gruplar fold sınırlarını asla aşmaz. |
| Tek split, variance tahmini yok (çoğu) | mean ± std ile repeated grouped cross-validation; test setinde bootstrap CI'lar. |
| Neural modeller için metnin aşırı truncate edilmesi (#2, #3) | Transformer kullanılırsa, tüm training seti ve uygulanabilir en uzun sequence length kullanılır. |

---

## 5. Test Edilmek Üzere Benimsenen Fikirler

Her fikir, Bölüm 6'daki protokol altında mevcut baseline'a karşı bir ablation olarak test edilecektir. Bir fikir, yalnızca sağladığı iyileşme fold'dan fold'a değişimi aşarsa korunur.

| Fikir | Kaynak | Gerekçe |
|---|---|---|
| Word n-gram'lar (1–2, 1–3) ve tüm metin alanları | #4 | En güçlü temiz basit sonuç. Önceki `requirements` ablation'ımız yalnızca unigram kullanıyordu. |
| Character n-gram'lar (`char_wb`, 3–5) | Hiçbir notebook'ta yok | Yazım farklılıklarına ve formatlama kalıntılarına karşı dayanıklı. |
| Alan başına ayrı TF-IDF (title, company profile, description) | #7 | Boş company profile gibi alana özgü sinyalleri korur. |
| Metin uzunluğu, kelime sayısı, boş alan göstergeleri | #7 (bug düzeltilerek) | Fake ilanlar uzunluk ve doluluk açısından farklılık gösteriyor. |
| `location`'dan çıkarılan ülke | #4, #5 (orada ölçülmemiş) | Muhtemelen güçlü bir sinyal. Modelin fraud ipuçları yerine coğrafyayı öğrenme riski tartışılacak. |
| Maskelenmiş token sayıları (`#URL_`, `#EMAIL_`, `#PHONE_`) | Veride gözlemlendi | Genellikle cleaning'de atılıyor; iletişim bilgisi yoğun scam ilanlarına işaret edebilir. |
| mean ± std raporlamalı CV | #2, #7 | Küçük farkları değerlendirmek için gerekli. |
| Tüm veri üzerinde sentence embedding'ler / DistilBERT | #2 (kötü uygulanmış) | TF-IDF'in kaçırdığı anlamsal ipuçlarını yakalar (örn. "pay a registration fee"). |

**Benimsenmeyenler:** Sparse metin üzerinde SMOTE (#9'da class weighting'e göre kazanç yok ve kendi random-oversampling deneyimiz F1'i düşürdü); sıfırdan eğitilen recurrent network'ler (veri çok az; #3 veya #9'da avantaj yok).

---

## 6. Planlanan Metodoloji

### Adım 1 — Değerlendirme altyapısı
- Mevcut grouped test setini (2,597 ilan, 131 fake) final değerlendirmeye kadar dokunulmadan tut.
- Kalan development verisi (training + validation, 15,001 ilan) üzerinde, mevcut description-similarity gruplarını kullanarak **repeated `StratifiedGroupKFold`** (5 fold × 3 repeat) çalıştır.
- Decision threshold'u **her training fold'unun içinde** (inner grouped CV) seç, böylece raporlanan F1 ölçüldüğü fold üzerinde ayarlanmamış olur.
- Her model için şunların mean ± std değerlerini raporla: fake-class PR-AUC (average precision), ROC AUC, F1, precision, recall.
- Final test değerlendirmesi için **group-level bootstrap %95 confidence interval'ları** raporla.
- Mevcut final modeli bu protokol altında yeniden ölç ve **baseline** olarak sabitle.

### Adım 2 — Feature engineering
Bölüm 5'teki fikirlerin, her seferinde tek bir değişiklik yapılarak ablation'ları.

**Tamamlandı** (`feature_engineering.ipynb`). Seçilen konfigürasyon: word 1–2 gram + `char_wb` 3–5 gram, `sublinear_tf=True`, `min_df=2`, `C=1`. Development CV PR-AUC 0.825 → 0.871 (15/15 fold'da daha iyi), F1 0.748 → 0.789 (14/15). Meta feature'lar, location, ekstra categorical ve binary alanlar karar kuralını geçemedi.

### Adım 3 — Model genişletmeleri
Adım 2 konfigürasyonu yeni baseline'dır. Önce iki kontrol yapılacak:
- **Yeni fold'larda tekrar kontrol.** Adım 2 konfigürasyonu, Adım 1 baseline'ı ile hiç kullanılmamış bir CV seed'inden üretilen fold'larda bir kez daha karşılaştırılacak. Adım 2, aynı fold'larda 14 varyant arasından en iyisini seçti; kazanç yeni fold'larda da korunuyorsa bu seçimin bir ürünü değildir.
- **Label noise kontrolü.** Her iki label'ı birden içeren 8 similarity grubu (73 ilan, 13 fake) incelenecek ve bunların training'den çıkarılmasının etkisi ölçülecek (sensitivity analysis). Label'lar değiştirilmeyecek.

Adaylar çalıştırmadan önce sabitlenir ve sonuçlar görüldükten sonra yeni varyant eklenmez: SVM'in probability calibration'ı (`CalibratedClassifierCV`); sentence-embedding feature'ları; out-of-fold metin skorları ve structured features üzerinde gradient boosting (LightGBM, güçlü regularization ile).

### Adım 4 — Opsiyonel transformer fine-tuning
GPU süresi elverirse, class weighting ile DistilBERT, aynı protokol altında değerlendirilecek.

### Adım 5 — Final model ve rapor
Modeli development CV üzerinde seç, threshold'u sabitle, held-out test setinde **bir kez** değerlendir ve group-level bootstrap confidence interval'larını raporla. Ek olarak:
- **Yorumlama.** SVM coefficient'ları (veya SHAP) ile genel dolandırıcılık işaretlerini (ör. "registration fee", "wire transfer") şirket adları gibi veri setine özgü izlerden ayır.
- **Shortcut testi.** Şirket adları maskelendiğinde veya `company_profile` çıkarıldığında performans düşüşünü ölç. Büyük bir düşüş, modelin dolandırıcılık işaretlerini öğrenmek yerine şirketleri ezberlediği anlamına gelir.
- **Hata analizi.** Modelin en emin olduğu hatalı tahminleri incele; bir kısmı label hatası olabilir.
- **Limitations.** CV–test farkını ve Bölüm 7'deki kısıtları yukarıdaki ölçümlerle birlikte raporla.

---

## 7. Riskler ve Kısıtlar

| Risk | Bildiklerimiz | Önlem | Ne zaman |
|---|---|---|---|
| **Küçük pozitif sınıf** | 855 fake ilan var, ancak yalnızca 518 bağımsız similarity grubu (development verisinde 436); herhangi tek bir split yüksek variance'a sahip | 15 fold üzerinde mean ± std, paired comparison, sıkı karar kuralı, test setinde bootstrap CI; düşük variance'lı modeller tercih edilir | Tüm adımlar |
| **Selection bias** | Adım 2, aynı fold'larda 14 varyant arasından en iyisini seçti; bu yüzden CV skoru biraz iyimser | Yeni fold'larda tekrar kontrol; aday listelerini çalıştırmadan önce sabitleme; test setini bir kez kullanma; CV–test farkını raporlama | Adım 3 ve 5 |
| **Tek kaynak ve dönem** | Tek platform, 2012–2014; tarih kolonu yok, bu yüzden zamana göre split mümkün değil | Bu veriyle çözülemez. Yorumlama ve shortcut testi modelin neye dayandığını gösterir; limitations bölümünde açıkça belirtilir | Adım 5 |
| **Label noise** | Birden fazla ilan içeren 1,767 grubun 8'inde label'lar karışık (73 ilan, 13 fake); birebir aynı description'lara sahip bir küme çelişkili label'lar taşıyor | Label'ları değiştirmeden elle inceleme ve sensitivity analysis; emin olunan hataların analizi | Adım 3 ve 5 |
| **Location feature'ları** | Adım 2'de ölçülebilir kazanç yok; benimsenmedi | Ek bir işlem gerekmiyor | — |
