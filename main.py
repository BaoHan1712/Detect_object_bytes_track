import cv2
from ultralytics import solutions

video_path = r"data\v1.mp4"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# List các vùng theo dõi
regions = [
    [(331, 523), (728, 179), (1167, 256), (989, 709)],
    [(170, 577), (474, 313), (133, 133), (-8, 266)]
]

target_width, target_height = 1280, 720


# Khởi tạo trackzones
trackzones = []
for region in regions:
    trackzone = solutions.TrackZone(
        region=region,
        model="yolo11l.pt",
        tracker="bytetrack.yaml",
        verbose=False,
        conf=0.55
    )
    trackzones.append(trackzone)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Hết video hoặc lỗi đọc frame.")
        break

    frame = cv2.resize(frame, (target_width, target_height))

    # Áp dụng tracking cho từng vùng
    for trackzone in trackzones:
        _ = trackzone.trackzone(frame)  # Vẽ trực tiếp lên frame

    cv2.imshow("Tracking Result", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
