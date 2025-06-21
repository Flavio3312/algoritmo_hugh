
import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Detección de Círculos con Transformada de Hough

Este script detecta círculos en una imagen usando la Transformada de Hough
para círculos (cv2.HoughCircles). Se verifica si el centro de algún círculo
detectado está cerca del punto A (391, 200).

Autor: Flavio Pérez
Año: 2025
"""


def detectar_circulo_hough(imagen_path, centro_a=(391, 200), mostrar=True, guardar_salida=False):
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen '{imagen_path}'")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)  # Suavizado más fuerte

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,     # Canny threshold
        param2=20,      # Umbral para acumulador: más bajo = más detecciones
        minRadius=20,
        maxRadius=60
    )

    output = img.copy()
    centro_detectado = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            dist = np.linalg.norm(np.array([x, y]) - np.array(centro_a))
            if dist < 20:
                cv2.circle(output, (x, y), r, (0, 255, 0), 3)
                cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(output, "C", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                centro_detectado = (x, y)
                break

    if centro_detectado is None:
        print("No se detectó círculo cerca del punto A.")
    else:
        print(f"Círculo detectado en: {centro_detectado}")

    cv2.circle(output, centro_a, 5, (255, 0, 0), -1)
    cv2.putText(output, "A", (centro_a[0] + 5, centro_a[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    if mostrar:
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.title("Detección de círculo con Hough")
        plt.axis("off")
        plt.show()

    if guardar_salida:
        salida_path = imagen_path.replace('.', '_circulo_hough.')
        cv2.imwrite(salida_path, output)

    return centro_detectado


if __name__ == "__main__":
    ruta_imagen = "./bloque-motor.jpg"
    centro_a = (391, 200)
    centro_circulo = detectar_circulo_hough(ruta_imagen, centro_a)
    if centro_circulo is not None:
        print(f"Círculo detectado en: {centro_circulo}")
    else:
        print("No se detectó ningún círculo cerca del punto A.")
