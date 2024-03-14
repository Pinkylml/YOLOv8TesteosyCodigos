import cv2
import numpy as np

# Ruta de la imagen original
image_path = '/home/dell/Documentos/YOLOv7Test/YOLOv8/images/20220720_000735_thumb_png.rf.8ccb4eb352db96f9b7a6d0bfc272c5b0.jpg'

# Ruta del archivo .txt con las coordenadas
txt_path = '/home/dell/Documentos/YOLOv7Test/YOLOv8/runs/segment/predict10/labels/20220720_000735_thumb_png.rf.8ccb4eb352db96f9b7a6d0bfc272c5b0.txt'

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

# Guardar la imagen resultante
cv2.imwrite('imagen_recortadaprueba2.jpg', result)


