# Egg Detection and Counting on Conveyor Belt
(Konveyör Bant Üzerinde Yumurta Tespiti ve Sayımı)

A computer vision project that performs real-time egg detection, segmentation, and counting using YOLOv11m-seg model on conveyor belt footage.
(Konveyör bant görüntüleri üzerinde YOLOv11m-seg modeli kullanarak gerçek zamanlı yumurta tespiti, segmentasyonu ve sayımı yapan bir bilgisayarlı görü projesi.)

## Project Overview
(Proje Genel Bakış)

This project implements an automated system for detecting, segmenting, and counting eggs moving on a conveyor belt. It uses the YOLOv11m-seg model for accurate egg segmentation and a custom tracking algorithm for precise counting.
(Bu proje, konveyör bant üzerinde hareket eden yumurtaları tespit eden, segmente eden ve sayan otomatik bir sistem uygulamaktadır. Hassas yumurta segmentasyonu için YOLOv11m-seg modeli ve hassas sayım için özel bir takip algoritması kullanır.)

## Labeling in the CVAT.ai 
![etiket](https://github.com/user-attachments/assets/50135c91-5237-4368-ad0f-0bcad874da67)


## Videos:(Not: Video kaliteleri Github'a yüklemek için düşürüldü.)

https://github.com/user-attachments/assets/cfe99aea-c888-482f-a31e-354be1ddce43 



https://github.com/user-attachments/assets/0b1dd88c-b3b7-4822-8244-bae7139247c6




https://github.com/user-attachments/assets/fa2cf22a-76c6-4508-9a67-e00dd5f9c424



## Features
(Özellikler)

- Real-time egg detection and segmentation (Gerçek zamanlı yumurta tespiti ve segmentasyonu)
- Accurate counting system with direction awareness (Yön farkındalıklı hassas sayım sistemi)
- Object tracking to prevent double counting (Çift sayımı önlemek için nesne takibi)
- Video processing and output generation (Video işleme ve çıktı oluşturma)
- Result logging and visualization (Sonuç kaydetme ve görselleştirme)

## Technologies Used
(Kullanılan Teknolojiler)

- Python 3.12
- YOLOv11m-seg
- OpenCV
- Ultralytics
- CVAT.ai (for annotation / etiketleme için)
- SMPlayer (for frame extraction / kare çıkarımı için)

## Dataset Preparation
(Veri Seti Hazırlama)

1. **Frame Extraction** (Kare Çıkarma): 
   Used SMPlayer to extract individual frames from conveyor belt footage
   (Konveyör bant görüntülerinden tek tek kareleri çıkarmak için SMPlayer kullanıldı)

2. **Annotation** (Etiketleme): 
   - Used CVAT.ai for segmentation annotation (Segmentasyon etiketlemesi için CVAT.ai kullanıldı)
   - Annotated approximately 1000 eggs across 50 images (50 görüntüde yaklaşık 1000 yumurta etiketlendi)
   - Dataset split: 80% training, 20% validation (Veri seti bölünmesi: %80 eğitim, %20 doğrulama)

3. **Image Processing** (Görüntü İşleme): 
   Used `image_rename.py` for consistent dataset organization
   (Tutarlı veri seti organizasyonu için `image_rename.py` kullanıldı)

## Model Training
(Model Eğitimi)

Training parameters on Kaggle (Kaggle üzerinde eğitim parametreleri):
- Epochs: 150 
- Batch size: 8 
- Workers: 8 
- Image size: 640x640 

## Project Structure
(Proje Yapısı)

```
project-root/
├── src/
│   ├── counting.py           # Main counting implementation (Ana sayım uygulaması)
│   ├── segment_egg.py        # Segmentation implementation (Segmentasyon uygulaması)
│   ├── segment_trying.py     # Experimental segmentation code (Deneysel segmentasyon kodu)
│   └── image_rename.py       # Dataset organization utility (Veri seti düzenleme aracı)
|   └── README_counting.md    # Readme for counting.py (Counting.py Readme'si)
|   └── egg_video/
|      ├── egg_video.mp4      # Main video
├── README.md

```

## Key Components
(Ana Bileşenler)

### EggCounter Class (`counting.py`)
(EggCounter Sınıfı)

The main implementation includes (Ana uygulama şunları içerir):
- Object tracking system using unique IDs (Benzersiz kimlikler kullanan nesne takip sistemi)
- Distance-based object matching (Mesafe tabanlı nesne eşleştirme)
- Line crossing detection (Çizgi geçiş tespiti)
- Count validation to prevent duplicates (Çift sayımı önlemek için sayım doğrulama)

Key features (Temel özellikler):
```python
class EggCounter:
    def __init__(self, line_position, direction="right_to_left"):
        self.line_position = line_position  # Çizgi pozisyonu
        self.direction = direction          # Yön
        self.count = 0                      # Sayaç
        self.tracked_objects = {}           # Takip edilen nesneler
        self.object_id = 0                  # Nesne kimliği
        self.counted_objects = set()        # Sayılan nesneler
```

### Counting Logic
(Sayım Mantığı)

The system employs three-step verification (Sistem üç adımlı doğrulama kullanır):
1. Previous position check (Önceki pozisyon kontrolü)
2. Current position check (Mevcut pozisyon kontrolü)
3. ID verification to prevent double counting (Çift sayımı önlemek için kimlik doğrulama)

## Installation
(Kurulum)

```bash
# Clone the repository (Depoyu klonlayın)
git clone [https://github.com/enescanerkan/EggCounter-YOLOv11]

# Install dependencies (Bağımlılıkları yükleyin)
pip install opencv-python
pip install ultralytics
```

## Usage
(Kullanım)

1. Place your video file in the project directory (Video dosyanızı proje dizinine yerleştirin)
2. Update the video path in `counting.py` (`counting.py` içindeki video yolunu güncelleyin):
   ```python
   input_video_path = "path_to_your_video.mp4"
   ```
3. Run the counter (Sayacı çalıştırın):
   ```bash
   python counting.py
   ```

## Results
(Sonuçlar)

The system outputs (Sistem çıktıları):
- Processed video with visualization (Görselleştirmeli işlenmiş video)
- Real-time count display (Gerçek zamanlı sayım gösterimi)
- Final count saved to text file (Metin dosyasına kaydedilen final sayım)

