from ultralytics import YOLO
import os
# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/dell/Documentos/YOLOv7Test/YOLOv8/YOLOv8Segmentation50epoch.pt')  # load a custom model
detected=0
no_detected=0
for image in os.listdir('/home/dell/Documentos/YOLOv7Test/YOLOv8/dataset_crestas/no'):
    path_img=os.path.join('/home/dell/Documentos/YOLOv7Test/YOLOv8/dataset_crestas/no',image) 
    results = model(path_img, save=True, conf=0.2,save_crop=True,imgsz=640)  
    for r in results:
        flag=r.boxes
    if bool(flag):
        print("Hubo detecciones")
        detected+=1
    else:
        print("No hubo detecciones")
        no_detected+=1

cantidad=len(os.listdir('/home/dell/Documentos/YOLOv7Test/YOLOv8/dataset_crestas/no'))
print(f"Hay {cantidad} imagenes de las cuales")
porcentaje_validas=(detected*100)/cantidad
porcentaje_novalidas=(no_detected*100)/cantidad
print("Se detectaron: ",detected,str(porcentaje_validas))
print("No detectaron: ",no_detected,str(porcentaje_novalidas))