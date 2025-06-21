# hough_line_detection_real.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Detección de Líneas con Transformada de Hough

Este script detecta líneas en una imagen usando la Transformada de Hough probabilística
(cv2.HoughLinesP). Las líneas detectadas se filtran para conservar únicamente aquellas
que pasan cerca del punto A (391, 200).

Autor: Flavio Pérez
Año: 2025
"""


def distancia_punto_a_linea(p, a, b):
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    if np.all(a == b):
        return np.linalg.norm(p - a)
    else:
        line_vec = b - a
        p_vec = p - a
        t = np.dot(p_vec, line_vec) / np.dot(line_vec, line_vec)
        t = np.clip(t, 0, 1)
        proj = a + t * line_vec
        return np.linalg.norm(p - proj)

def detectar_lineas_hough(imagen_path, centro_a=(391, 200), mostrar=True, guardar_salida=False):
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen '{imagen_path}'")
        return 0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=30, maxLineGap=10)
    output = img.copy()
    count = 0

    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            dist = distancia_punto_a_linea(centro_a, (x1,y1), (x2,y2))
            if dist < 10:
                cv2.line(output, (x1,y1), (x2,y2), (0,255,0), 2)
                count += 1

    cv2.circle(output, centro_a, 5, (0,0,255), -1)
    cv2.putText(output, "A", (centro_a[0]+5, centro_a[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    if mostrar:
        plt.figure(figsize=(10,6))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title(f"Líneas detectadas cerca del punto A: {count}")
        plt.axis("off")
        plt.show()

    if guardar_salida:
        salida_path = imagen_path.replace('.', '_lineas_hough.')
        cv2.imwrite(salida_path, output)

    return count

if __name__ == "__main__":
    ruta_imagen = "./bloque-motor.jpg"
    centro_a = (391, 200)
    num_lineas = detectar_lineas_hough(ruta_imagen, centro_a)
    print(f"Líneas detectadas cerca del punto A: {num_lineas}")
