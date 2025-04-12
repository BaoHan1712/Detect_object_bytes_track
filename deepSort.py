import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from supervision.draw.color import Color


# Hàm vẽ regions
def draw_polygon(frame, points, color=(0, 255, 0), thickness=1):
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(frame, points[i], points[i + 1], color, thickness)
        else:
            cv2.line(frame, points[i], points[0], color, thickness)
    return frame

video_path = r"data\v1.mp4"
cap = cv2.VideoCapture(video_path)

target_width, target_height = 1280, 720

# List các vùng theo dõi
regions = [
    [(331, 523), (728, 179), (1167, 256), (989, 709)],
    [(170, 577), (474, 313), (133, 133), (-8, 266)]
]

model = YOLO("yolo11l.pt")

tracker = sv.ByteTrack()

# Tạo các đối tượng vùng theo dõi (Zone)
zones = []
for region in regions:
    polygon = np.array(region)
    zone = sv.PolygonZone(polygon=polygon)
    zones.append(zone)

class_names = model.model.names

# Tạo annotator cho khung
box_annotator = sv.BoxAnnotator(thickness=1)

# Annotator cho các vùng
zone_annotators = []
for zone in zones:
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=Color.GREEN,
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )
    zone_annotators.append(zone_annotator)

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Hết video hoặc lỗi đọc frame.")
        break

    frame = cv2.resize(frame, (target_width, target_height))
    
    # Detect objects
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Tracking
    detections = tracker.update_with_detections(detections)
    
    # Annotate từng object
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
        tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else None
        class_name = class_names[class_id] if class_id >= 0 else "Unknown"

        label = f"#{tracker_id} {class_name}" if tracker_id is not None else class_name

        # Vẽ khung
        frame = box_annotator.annotate(
            scene=frame,
            detections=sv.Detections(
                xyxy=np.array([box]),
                class_id=np.array([class_id]),
                tracker_id=np.array([tracker_id]) if tracker_id is not None else None
            )
        )

        # Vẽ nhãn
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )
    
    # Vẽ các vùng theo dõi
    for i, region in enumerate(regions):
        # Vẽ đường viền cho region
        frame = draw_polygon(frame, region)
        
        # Lọc detections nằm trong zone
        zone_detections = zones[i].trigger(detections=detections)
        
        # Vẽ annotate cho zone
        frame = zone_annotators[i].annotate(scene=frame)
        zone_center = np.mean(region, axis=0).astype(int)
        cv2.putText(
            frame, 
            f"Zone {i+1}: {zone_detections.sum()} objects", 
            (zone_center[0] - 50, zone_center[1]), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 0, 255), 
            2
        )
    cv2.imshow("DeepSORT Tracking", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
