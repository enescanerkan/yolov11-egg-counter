from ultralytics import YOLO

# Modeli yükle
model_path = r"C:\Users\Monster\Desktop\best.pt"
model = YOLO(model_path)

# Videoyu çalıştır ve sonuçları kaydet
video_path = r"C:\Users\Monster\Desktop\Conveyor1_egg.mp4"

# Modeli video üzerinde çalıştır (GPU kullanarak)
results = model.predict(
    source=video_path,   # İşlenecek video
    conf=0.5,            # Güven eşiği
    save=True,           # Sonuçları kaydet
    device='cuda'        # GPU kullanımı
)

print("Sonuçlar kaydedildi.")
