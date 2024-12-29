from ultralytics import YOLO

model = YOLO('yolov8x') 

results = model.predict('input_videos/08fd33_4.mp4', save = True) 

print(results[0])

print("=======================")


# logs the data from all the bounding boxes created by the model
for box in results[0].boxes:
    print(box) 