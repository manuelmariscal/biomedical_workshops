# Biomedical Workshops: Gait Analysis for Biomedical Engineers

## Descripción

Este repositorio contiene el código y los recursos necesarios para realizar un taller sobre el análisis de la marcha utilizando visión por computadora. Los participantes aprenderán a capturar y analizar los movimientos de la marcha en tiempo real utilizando herramientas como TensorFlow y MediaPipe.

## Estructura del Repositorio

```bash
biomedical_workshops/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── gait_analysis.py
└── main.py
```

## Requisitos

- Python 3.x
- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- SciPy

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone <URL del repositorio>
    cd biomedical_workshops
    ```

2. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Modo Completo de Análisis de la Marcha

Para capturar video, realizar el seguimiento de la marcha y analizar los datos en tiempo real, ejecuta el siguiente comando:

```bash
python main.py --source 0
```

Proporciona la fuente de video (0 para cámara de laptop, URL para cámara de celular) cuando se te solicite.

### Detalles Técnicos

El análisis incluye la captura de datos de movimiento durante 30 segundos, guardando los puntos de interés y visualizando una "movie" del movimiento de los puntos rastreados entre el segundo 5 y el 25, con los siguientes segmentos:

- `left_ankle` a `left_knee`
- `left_knee` a `left_hip`
- `left_hip` a `center_of_gravity`
- `center_of_gravity` a `left_shoulder`
- `left_shoulder` a `left_elbow`
- `left_elbow` a `left_wrist`
- `right_ankle` a `right_knee`
- `right_knee` a `right_hip`
- `right_hip` a `center_of_gravity`
- `center_of_gravity` a `right_shoulder`
- `right_shoulder` a `right_elbow`
- `right_elbow` a `right_wrist`

Los datos de las posiciones de los puntos se suavizan utilizando un filtro de Savitzky-Golay para reducir el ruido.

## Recursos y Material Adicional

- [Documentación de MediaPipe](https://google.github.io/mediapipe/)
- [Documentación de OpenCV](https://opencv.org/)
- [Documentación de TensorFlow](https://www.tensorflow.org/)

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus mejoras o sugerencias.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

Este archivo `README.md` proporciona una guía clara y detallada sobre cómo usar los scripts y recursos del repositorio para el taller de análisis de la marcha, así como información técnica adicional sobre el procesamiento y visualización de los datos.
