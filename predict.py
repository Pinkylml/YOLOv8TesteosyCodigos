from ultralytics import YOLO
import os
import cv2
# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('ASL_15epoch.pt')  # load a custom model
path_img='images/C9_jpg.rf.0499bc1602cddf18027320fbc97ae10e.jpg'
image=cv2.imread(path_img)
results = model(path_img, conf=0.3,show=True) 
for r in results:
    bos=r.boxes
    box=bos.xyxy
    numpy=box.numpy()
    print(box)
    print(numpy)
    #print(r.masks)
    print("clase",bos.cls)

#x1=numpy[0][0]
#y1=numpy[0][1]
#x2=numpy[0][2]
#y2=numpy[0][3]
#cropped_region = image[int(y1):int(y2), int(x1):int(x2)]
#cv2.imwrite('/home/dell/Documentos/YOLOv7Test/YOLOv8/Detecciones/image.png', cropped_region)