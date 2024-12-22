import cv2
import numpy as np
from ultralytics import YOLO


class EggCounter:
    def __init__(self, line_position, direction="right_to_left"):
        """
        Başlangıç değerlerini ayarla
        """
        self.line_position = line_position
        self.direction = direction
        self.count = 0
        self.tracked_objects = {}
        self.object_id = 0
        self.counted_objects = set()

    def process_frame(self, frame, results):
        """
        Frame'i işle ve hem box hem de segmentasyon maskelerini göster
        """
        updated_tracked_objects = {}

        # Orijinal frame'in bir kopyasını al (maskeleri göstermek için)
        overlay = frame.copy()

        # Her tespit için
        for i, det in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            confidence = det.conf[0].item()

            if confidence > 0.5:  # Güven eşiği kontrolü
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Nesne takibi ve sayma mantığı (önceki gibi)
                matched_id = None
                min_distance = float('inf')

                for obj_id, obj_data in self.tracked_objects.items():
                    prev_center_x, prev_center_y = obj_data["center"]
                    distance = ((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2) ** 0.5
                    if distance < 50 and distance < min_distance:
                        matched_id = obj_id
                        min_distance = distance

                if matched_id is not None:
                    prev_center_x = self.tracked_objects[matched_id]["center"][0]

                    if self.direction == "right_to_left":
                        if prev_center_x > self.line_position and center_x <= self.line_position:
                            if matched_id not in self.counted_objects:
                                self.count += 1
                                self.counted_objects.add(matched_id)
                    else:
                        if prev_center_x < self.line_position and center_x >= self.line_position:
                            if matched_id not in self.counted_objects:
                                self.count += 1
                                self.counted_objects.add(matched_id)

                    updated_tracked_objects[matched_id] = {
                        "center": (center_x, center_y),
                        "box": (x1, y1, x2, y2)
                    }
                else:
                    self.object_id += 1
                    updated_tracked_objects[self.object_id] = {
                        "center": (center_x, center_y),
                        "box": (x1, y1, x2, y2)
                    }

                # Segmentasyon maskesini göster (eğer varsa)
                if hasattr(results[0], 'masks') and results[0].masks is not None:
                    mask = results[0].masks[i].data[0].cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    color_mask = np.zeros_like(frame)
                    color_mask[mask > 0.5] = [0, 255, 0]  # Yeşil renk
                    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)

                # Bounding box ve ID'yi göster
                object_id = matched_id if matched_id is not None else self.object_id
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(overlay, f"ID: {object_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        self.tracked_objects = updated_tracked_objects

        # Sayım çizgisini çiz
        cv2.line(overlay, (self.line_position, 0),
                 (self.line_position, frame.shape[0]), (0, 255, 0), 2)

        # Sayacı göster
        cv2.putText(overlay, f"Count: {self.count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return overlay


if __name__ == "__main__":
    # Video girişini aç
    cap = cv2.VideoCapture(r"C:\Users\Monster\Desktop\egg_dataset\Conveyor1_egg.mp4")

    # Video yazıcıyı ayarla
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Sayaç ayarları
    line_position = 300
    direction = "right_to_left"
    egg_counter = EggCounter(line_position, direction)

    # YOLO modelini yükle
    model = YOLO(r"C:\Users\Monster\Desktop\egg_dataset\best.pt")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO tahminlerini al (şimdi tüm sonuçları geçiriyoruz)
        results = model(frame)

        # Frame'i işle
        processed_frame = egg_counter.process_frame(frame, results)

        # Sonuçları göster ve kaydet
        cv2.imshow("Egg Counter", processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nToplam sayılan yumurta: {egg_counter.count}")