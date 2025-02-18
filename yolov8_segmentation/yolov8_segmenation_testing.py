from ultralytics import YOLO
import cv2

model = YOLO("yolov8x-seg.pt")
image_path = "../utilities/objects.jpeg"
image = cv2.imread(image_path)

results = model(image)
annotated_image = results[0].plot()

cv2.imwrite("yolov8x_segmentation.jpg", annotated_image)

