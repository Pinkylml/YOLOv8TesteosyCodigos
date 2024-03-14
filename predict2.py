from ultralytics import YOLO
import os
import cv2
import errno
# Load a model
#model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/home/dell/Documentos/YOLOv7Test/YOLOv8/YOLOv8Model50Epoch.pt')  # load a custom model
directorio_actual = os.path.dirname(__file__) 
print(directorio_actual)
path_detections = directorio_actual + '/Detecciones'
try:
    os.mkdir(path_detections)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
def recortar(numpy,path_img,imagen):
    x1=numpy[0][0]
    y1=numpy[0][1]
    x2=numpy[0][2]
    y2=numpy[0][3]
    cropped_region = imagen[int(y1):int(y2), int(x1):int(x2)]
    nombre_sin_extension = os.path.splitext(os.path.basename(path_img))[0]
    print(nombre_sin_extension)
    path_to_detected_img = os.path.join(path_detections, nombre_sin_extension)
    print(path_to_detected_img)
    try:
        os.mkdir(path_to_detected_img)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    cv2.imwrite(f'{path_to_detected_img}/{nombre_sin_extension}detection.png', cropped_region)


def detectar(path):

    image=cv2.imread(path)
    results = model(path, conf=0.3) 
    for r in results:
        bos=r.boxes
        box=bos.xyxy
        numpy=box.numpy()
        print(box)
        print(numpy)
        #print(r.masks)
    if bool(bos):
      recortar(numpy,path,image)
    else:
        print("No hubo detecciones")

#detectar('/home/dell/Documentos/YOLOv7Test/YOLOv8/images/5.jpg')
for image in os.listdir('/home/dell/Documentos/YOLOv7Test/YOLOv8/images'):
    path_to_image=os.path.join('/home/dell/Documentos/YOLOv7Test/YOLOv8/images',image)
    detectar(path_to_image)