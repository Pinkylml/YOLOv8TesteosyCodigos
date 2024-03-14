from ultralytics import YOLO
from PIL import Image
import os
import errno
import cv2
import numpy as np
directorio_actual = os.path.dirname(__file__) 
print(directorio_actual)
path_detections = directorio_actual + '/RecortesDatasetCrestas'
path_detections2 = path_detections+'/Validas'
path_detections3 = path_detections+'/NoValidas'
path_labels=path_detections2+'/lables'
path_labels2=path_detections3+'/lables'
#nombre_sin_extension = os.path.splitext(os.path.basename(path_img))[0]
#print(nombre_sin_extension)
#path_to_croped_img=os.path.join(path_detections,nombre_sin_extension)
try:
    os.mkdir(path_detections)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.mkdir(path_detections2)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.mkdir(path_detections3)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.mkdir(path_labels)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
try:
    os.mkdir(path_labels2)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
model = YOLO('/home/dell/Documentos/YOLOv7Test/YOLOv8/YOLOv8Segmentation50epoch.pt') 
detected=0
no_detected=0
def recorte(path_to_image,path_to_txt,outhput_path,name):
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
        points = [(int(float(parts[i]) * image.shape[1]), int(float(parts[i+1]) * image.shape[0])) for i in range(0, len(parts), 2)]
        
        # Convertir la lista de puntos en un formato compatible con OpenCV
        points = np.array(points, dtype=np.int32)
        
        # Dibujar el polígono en la máscara
        cv2.fillPoly(mask, [points], (255, 255, 255))

    # Recortar la región de interés de la imagen original usando la máscara
    result = cv2.bitwise_and(image, mask)
    output_image=os.path.join(outhput_path,f'{name}segmented.png')
    cv2.imwrite(output_image, result)


def predictandSegment(path_img):
    global detected
    global no_detected
    results = model(path_img,imgsz=640) 
    rutasSplit=path_img.split('/')
    carpetaContenedora=rutasSplit[len(rutasSplit)-2]
    if carpetaContenedora=='si':
        out_path=path_detections2
        path_to_txt=path_labels
    elif carpetaContenedora=='no':
        out_path=path_detections3
        path_to_txt=path_labels2
    else:
        print('Algo salio mal')
    for r in results:
        blag=r.boxes
    if bool(blag):  
        nombre_sin_extension = os.path.splitext(os.path.basename(path_img))[0]
        detected+=1
        for r in results:
            path_to_txt2=os.path.join(path_to_txt,f'{nombre_sin_extension}.txt')
            r.save_txt(path_to_txt2)
            recorte(path_img,path_to_txt2,out_path,nombre_sin_extension)
    else:
        no_detected+=1

for carpeta in os.listdir('/home/dell/Documentos/YOLOv7Test/YOLOv8/dataset_crestas'):
    contenedora_ruta=os.path.join('/home/dell/Documentos/YOLOv7Test/YOLOv8/dataset_crestas',carpeta)
    for imagen in os.listdir(contenedora_ruta):
        imagen_ruta=os.path.join(contenedora_ruta,imagen)   
        predictandSegment(imagen_ruta)
    cantidad=len(os.listdir(contenedora_ruta))
    print(f"Hay {cantidad} imagenes en {carpeta} de las cuales.......")
    porcentaje_validas=(detected*100)/cantidad
    porcentaje_novalidas=(no_detected*100)/cantidad
    print("Se detectaron: ",detected,str(porcentaje_validas))
    print("No detectaron: ",no_detected,str(porcentaje_novalidas))
    