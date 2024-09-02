
from ultralytics import YOLO
import torch
import cv2
import supervision as sv
torch.cuda.is_available()
deploy_Model = YOLO("D:/PYTHON/Manhole Cover Detection/runs/detect/train3/weights/best.pt")
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
box_annot = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)


while True:
    ret, frame = cam.read()
    result = deploy_Model(frame)[0]
    detection = sv.Detections.from_yolov8(result)
    labels = [
        f"{deploy_Model.names[class_id]} {confidence:0.01f}"
        for _, confidence, class_id, _
        in detection
    ]
    frame = box_annot.annotate(scene=frame, detections=detection, labels=labels)
    cv2.imshow("Predictor", frame)

    if(cv2.waitKey(30)==27):
        break