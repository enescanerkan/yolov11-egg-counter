import cv2
import numpy as np
from ultralytics import YOLO

"""
EggCounter sınıfı, konveyör bant üzerindeki yumurtaları sayar.
Belirli bir çizgiyi geçen yumurtaları tespit eder ve takip eder.
Her yumurtayı yalnızca bir kez sayar.
"""


class EggCounter:
    def __init__(self, line_position, direction="right_to_left"):
        """
        EggCounter sınıfının başlatıcı metodu.

        Parametreler:
        line_position (int): Sayım çizgisinin x koordinatı
        direction (str): Sayım yönü ("right_to_left" veya "left_to_right")
        """
        self.line_position = line_position
        self.direction = direction
        self.count = 0  # Toplam sayım
        self.tracked_objects = {}  # Takip edilen nesnelerin bilgileri
        self.object_id = 0  # Benzersiz nesne ID'leri için sayaç
        self.counted_objects = set()  # Sayılmış nesnelerin ID'lerini tutan küme

    def get_total_count(self):
        """
        Toplam sayım değerini döndürür.
        """
        return self.count

    def save_count_to_file(self, filename="count_result.txt"):
        """
        Sayım sonucunu dosyaya kaydeder.
        """
        with open(filename, 'w') as f:
            f.write(f"Toplam sayılan yumurta: {self.count}")

    def process_frame(self, frame, results):
        """
        Video karesini işler ve yumurta segmentasyonlarını yapar.

        Parametreler:
        frame: İşlenecek video karesi
        results: YOLO modelinden gelen segmentasyon sonuçları

        Return:
        İşlenmiş video karesi
        """
        # YOLO sonuçlarını çiz (segmentasyon dahil)
        annotated_frame = results[0].plot()

        updated_tracked_objects = {}

        # Her bir tespit için işlem yap
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Tespit kutusunun koordinatlarını ve merkez noktasını hesapla
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf[0].item()

                if confidence < 0.5:  # Güven eşiği kontrolü
                    continue

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # En yakın eşleşen nesneyi bul
                matched_id = None
                min_distance = float('inf')

                for obj_id, obj_data in self.tracked_objects.items():
                    prev_center_x, prev_center_y = obj_data["center"]
                    # Öklid mesafesi hesapla
                    distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
                    if distance < 50 and distance < min_distance:  # 50 piksel mesafe eşiği
                        matched_id = obj_id
                        min_distance = distance

                if matched_id is not None:
                    # Eşleşen nesnenin önceki pozisyonunu al
                    prev_center_x = self.tracked_objects[matched_id]["center"][0]

                    # Çizgi geçiş kontrolü yap
                    if self.direction == "right_to_left":
                        if prev_center_x > self.line_position and center_x <= self.line_position:
                            if matched_id not in self.counted_objects:
                                self.count += 1
                                self.counted_objects.add(matched_id)
                    else:  # left_to_right durumu
                        if prev_center_x < self.line_position and center_x >= self.line_position:
                            if matched_id not in self.counted_objects:
                                self.count += 1
                                self.counted_objects.add(matched_id)

                    # Nesne bilgilerini güncelle
                    updated_tracked_objects[matched_id] = {
                        "center": (center_x, center_y),
                        "box": (x1, y1, x2, y2)
                    }
                else:
                    # Yeni nesne oluştur
                    self.object_id += 1
                    updated_tracked_objects[self.object_id] = {
                        "center": (center_x, center_y),
                        "box": (x1, y1, x2, y2)
                    }

                # ID'yi göster
                object_id = matched_id if matched_id is not None else self.object_id
                cv2.putText(annotated_frame, f"ID: {object_id}", (x1, y1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Takip edilen nesneleri güncelle
        self.tracked_objects = updated_tracked_objects

        # Sayım çizgisini çiz
        cv2.line(annotated_frame, (self.line_position, 0),
                 (self.line_position, annotated_frame.shape[0]), (0, 0, 255), 3)

        # Toplam sayımı göster
        cv2.putText(annotated_frame, f"Count: {self.count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return annotated_frame


if __name__ == "__main__":
    """
    Ana program döngüsü.
    Video girişini açar, YOLO modelini yükler ve yumurta sayımını gerçekleştirir.
    İşlenmiş videoyu hem gösterir hem de kaydeder.
    """
    # Video girişini aç
    input_video_path = r"C:\Users\Monster\Desktop\egg_dataset\Conveyor1_egg.mp4"
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Hata: Video açılamadı!")
        exit()

    # Video özelliklerini al ve çıktı video yazıcısını ayarla
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Sayaç ve model ayarları
    line_position = 300
    direction = "right_to_left"
    egg_counter = EggCounter(line_position, direction)

    try:
        # Modeli yükle (segmentasyon modunda)
        model = YOLO(r"C:\Users\Monster\Desktop\egg_dataset\best.pt")
        print("Model başarıyla yüklendi.")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        exit()

    print("Video işleniyor...")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\nVideo işleme tamamlandı.")
            final_count = egg_counter.get_total_count()
            print(f"Son sayım: {final_count} yumurta")
            egg_counter.save_count_to_file()
            break

        # YOLO sonuçlarını al
        try:
            results = model(frame)

            # Kareyi işle ve sonuçları göster
            processed_frame = egg_counter.process_frame(frame, results)

            # Görüntüyü göster ve kaydet
            cv2.imshow("Egg Counter", processed_frame)
            out.write(processed_frame)

            frame_count += 1
            if frame_count % 30 == 0:  # Her 30 karede bir ilerleme göster
                print(f"İşlenen kare sayısı: {frame_count}", end='\r')

        except Exception as e:
            print(f"Kare işlenirken hata oluştu: {e}")
            continue

        # 'q' tuşuna basılırsa döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nKullanıcı tarafından sonlandırıldı.")
            break

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print(f"\nToplam sayılan yumurta: {egg_counter.get_total_count()}")