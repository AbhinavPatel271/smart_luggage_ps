import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5m_Objects365.pt", source="github")

image_path = "../utilities/objects.jpeg"
image = cv2.imread(image_path)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = model(image_rgb)

for result in results.xyxy[0]:  # Results in (x1, y1, x2, y2, confidence, class)
    x1, y1, x2, y2, conf, cls = map(int, result)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite("yolov5m_object365.jpg", image)
