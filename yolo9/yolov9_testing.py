from ultralytics import YOLO
import cv2

model = YOLO("yolov9e.pt")
image_path = "../utilities/objects.jpeg"
image = cv2.imread(image_path)

results = model(image)
annotated_image = results[0].plot()

cv2.imwrite("yolov9e_detection.jpg", annotated_image)

