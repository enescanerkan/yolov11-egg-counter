# Egg Counter Program

Bu belge, yumurta sayma uygulamasının çalışma prensibini, sınıfları ve fonksiyonları detaylı bir şekilde açıklamaktadır. Kod, bir konveyör bant üzerindeki yumurtaları algılamak, takip etmek ve saymak için geliştirilmiştir. İşlevselliği sağlamak için YOLO nesne algılama modeli ve OpenCV kullanılmıştır.

---

## Program Akışı

1. **Video Açma ve Model Yükleme**: Giriş videosu (konveyör bant üzerindeki yumurtaları içeren) işlenir ve YOLO modeli kullanılarak yumurta algılama yapılır.
2. **Yumurta Takibi ve Sayma**: Her tespit edilen yumurta bir kimlik (ID) alır. Çizgiyi geçen yumurtalar kontrol edilir ve bir kez sayılır.
3. **Sonuçların Kaydedilmesi**: Toplam sayım sonucu ekrana ve bir dosyaya yazılır. İşlenmiş video kaydedilir.

---

## Sınıflar ve Fonksiyonlar

### EggCounter Sınıfı

Konveyör bant üzerindeki yumurtaları algılamak, takip etmek ve saymak için geliştirilmiş bir sınıftır.

#### **init**(self, line_position, direction)

- **Amaç**: `EggCounter` sınıfını başlatır.
- **Parametreler**:
  - `line_position`: Çizginin x-koordinatı (yumurta sayma işlemi için referans alınır).
  - `direction`: Sayma yönü, `"right_to_left"` (sağdan sola) veya `"left_to_right"` (soldan sağa).
- **Değişkenler**:
  - `self.count`: Toplam sayılan yumurta sayısı.
  - `self.tracked_objects`: Takip edilen nesneler (ID ve pozisyon bilgileri).
  - `self.object_id`: Benzersiz nesne kimlikleri için sayaç.
  - `self.counted_objects`: Sayılmış nesnelerin ID'lerini saklayan küme.

#### get_total_count(self)

- **Amaç**: Toplam sayılan yumurta sayısını döndürür.
- **Dönüş Değeri**: `self.count`

#### save_count_to_file(self, filename="count_result.txt")

- **Amaç**: Toplam sayıyı bir dosyaya kaydeder.
- **Parametreler**:
  - `filename`: Dosya adı (varsayılan olarak `count_result.txt`).

#### process_frame(self, frame, detections)

- **Amaç**: Her video karesini işleyerek tespit edilen nesneleri takip eder ve çizgiyi geçen yumurtaları sayar.
- **Parametreler**:
  - `frame`: Video karesi (OpenCV formatında).
  - `detections`: YOLO modelinden alınan tespit sonuçları [(x1, y1, x2, y2, confidence), ...].
- **İşlem Adımları**:
  1. Algılanan nesnelerin merkez noktalarını hesaplar.
  2. Önceki kareden en yakın nesneyi eşleştirir (Öklid mesafesi ile).
  3. Çizgi geçiş kontrolü yapar ve gerekli durumlarda sayımı artırır.
  4. Her nesne için sınırlayıcı kutular ve kimlik bilgisi çizer.
  5. Çizgi ve toplam sayım bilgisini görüntüye ekler.
- **Dönüş Değeri**: İşlenmiş video karesi.

---

## Ana Program

`if __name__ == "__main__":` bloğunda yer alır ve şu işlemleri gerçekleştirir:

1. **Video ve Model Hazırlığı**

   - Giriş videosu ve YOLO modeli yüklenir.
   - Çıkış videosu için yazıcı (`VideoWriter`) ayarlanır.

2. **EggCounter Nesnesinin Başlatılması**

   - Çizgi konumu ve yön bilgisi ile `EggCounter` sınıfının bir örneği oluşturulur.

3. **Ana İşlem Döngüsü**

   - Videodan her kare alınır ve YOLO modeli ile işlenir.
   - Algılanan nesneler `process_frame` fonksiyonuna gönderilir.
   - İşlenmiş kare ekranda gösterilir ve çıkış videosuna kaydedilir.
   - `q` tuşuna basılarak döngü sonlandırılabilir.

4. **Sonuçların Kaydedilmesi**

   - Toplam sayım ekrana ve bir dosyaya yazılır.
   - Kaynaklar serbest bırakılır.

---

## Notlar

1. **Nesne Takibi ve Eşleştirme**

   - Algılanan her nesne, merkez noktası üzerinden takip edilir.
   - Öklid mesafesi ile en yakın nesne eşleştirilir.
   - 50 piksel eşik değeri kullanılır.

2. **Çizgi Geçiş Kontrolü**

   - Çizgiyi geçen nesnelerin pozisyonları önceki ve şimdiki karelerde kontrol edilir.
   - Sayılmış nesnelerin ID'leri bir kümede saklanarak tekrar sayılmaları önlenir.

3. **YOLO Tespitleri**

   - YOLO modelinden gelen tespitler [(x1, y1, x2, y2, confidence)] formatında alınır.
   - 0.5 güven eşiğinin altındaki tespitler göz ardı edilir.

---

## Kullanım Senaryosu

- Konveyör bant üzerindeki yumurta gibi hareketli nesneleri saymak için kullanılır.
- Endüstriyel otomasyon ve üretim hatlarında nesne takibi ve sayımı amacıyla uygulanabilir.
- Yumurtaların büyüklüğüne göre sınıflandırma yapılabilir. Atanan benzersiz ID'ler kameralarla takip edilerek yumurtaları S, M, L boylara göre sınıflandırabiliriz.
