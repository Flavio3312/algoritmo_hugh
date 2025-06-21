# Transformada de Hough

Este trabajo  aplica la **Transformada de Hough** para detectar **líneas** y **círculos** en una imagen (bloque de motor), utilizando OpenCV.

Se trabaja sobre un punto conocido (`punto A = (391, 200)`) para validar si los elementos detectados pasan cerca de ese centro.

---

## Archivos

- `hough_line_detection_real.py`: Detecta líneas rectas con `HoughLinesP`.
- `hough_circle_detection_real.py`: Detecta círculos con `HoughCircles`.

Ambos scripts filtran los resultados en función de su proximidad al punto A.

---

## Uso

```bash
python hough_line_detection_real.py
python hough_circle_detection_real.py
