from ultralytics import YOLO
from PIL import Image
import os
import errno
import cv2
import numpy as np
# Load a model
#model = YOLO('yolov8n-seg.pt')  
model = YOLO('/media/jeffersonc/DiscoRespaldo/YOLOv8/YOLOv8Segmentation50epoch.pt')  # load a custom model
path_img='/media/jeffersonc/DiscoRespaldo/YOLOv8/images/5.jpg'
# Predict with the model
results = model(path_img,imgsz=640,show=True) 
directorio_actual = os.path.dirname(__file__) 
print(directorio_actual)
path_detections = directorio_actual + '/DeteccionesConRecortes'
nombre_sin_extension = os.path.splitext(os.path.basename(path_img))[0]
print(nombre_sin_extension)
path_to_croped_img=os.path.join(path_detections,nombre_sin_extension)
try:
    os.mkdir(path_detections)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise



def recorteRectangular(numpyArrayCOrd,imageToCrop):
    cv2.imshow('Mi Imagen', imageToCrop)
    x1=numpyArrayCOrd[0][0]
    y1=numpyArrayCOrd[0][1]
    x2=numpyArrayCOrd[0][2]
    y2=numpyArrayCOrd[0][3]
    cropped_region = imageToCrop[int(y1):int(y2), int(x1):int(x2)]
    # Crear una máscara para identificar píxeles negros
    mask = (cropped_region == [0, 0, 0]).all(axis=2)

    # Cambiar el fondo negro a blanco
    cropped_region[mask] = [255, 255, 255]

    cv2.imshow('Recortada', cropped_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cropped_region

def recorteContorno(path_to_image,path_to_txt,outhput_path,name,coordenadasRecorte):
    # Ruta de la imagen original
    image_path = path_to_image

    # Ruta del archivo .txt con las coordenadas
    txt_path = path_to_txt

    # Cargar la imagen original
    image = cv2.imread(image_path)

    # Inicializar una máscara vacía
    mask = np.zeros_like(image)

    # Cargar las coordenadas desde el archivo .txt
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # Iterar a través de las líneas del archivo
    for line in lines:
        # Dividir la línea en partes y omitir la clase (primer número)
        parts = line.strip().split()[1:]
        
        # Convertir las partes en una lista de puntos (x, y)
        points = [(int((float(parts[i])+0.01) * image.shape[1]), int((float(parts[i+1])-0.01) * image.shape[0])) for i in range(0, len(parts), 2)]
        
        # Convertir la lista de puntos en un formato compatible con OpenCV
        points = np.array(points, dtype=np.int32)
        
        # Dibujar el polígono en la máscara
        cv2.fillPoly(mask, [points], (255, 255, 255))

    # Recortar la región de interés de la imagen original usando la máscara
    result = cv2.bitwise_and(image, mask)
    result=recorteRectangular(coordenadasRecorte,result)
    output_image=os.path.join(outhput_path,f'{name}segmented3.png')
    cv2.imwrite(output_image, result)

for r in results:
    blag=r.boxes
print(bool(blag)) 
if bool(blag):  
    for r in results:
        r.save_crop(path_to_croped_img)
        path_to_txt=os.path.join(path_to_croped_img,f'{nombre_sin_extension}.txt')
        r.save_txt(path_to_txt)
        name_img_croped=os.listdir(f'{path_to_croped_img}/dedo')
        print(name_img_croped)
        carpet='dedo'
        path_img2=os.path.join(path_to_croped_img,carpet,name_img_croped[0])
        print(path_img2)
        box=r.boxes
        box=box.xyxy
        coord=box.numpy()
        recorteContorno(path_img,path_to_txt,path_to_croped_img,nombre_sin_extension,coord)
