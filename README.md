# Biomedical Workshops: Gait Analysis for Biomedical Engineers

## Descripción

Este repositorio contiene el código y los recursos necesarios para realizar un taller sobre el análisis de la marcha utilizando visión por computadora. Los participantes aprenderán a capturar y analizar los movimientos de la marcha en tiempo real utilizando herramientas como TensorFlow y MediaPipe.

## Estructura del Repositorio

```bash
biomedical_workshops/
├── README.md
├── requirements.txt
├── src/
│   ├── capture_video.py
│   ├── gait_tracking.py
│   ├── realtime_gait_analysis.py
└── main.py
```

## Requisitos

- Python 3.x
- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- Matplotlib

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone <https://github.com/manuelmariscal/biomedical_workshops.git>
    cd biomedical_workshops
    ```

2. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

### Modo de Captura de Video

Para capturar video desde la cámara de tu laptop, ejecuta el siguiente comando:

```bash
python main.py --mode capture --source 0
```

Para capturar video desde la cámara de tu celular, proporciona la URL de la cámara:

```bash
python main.py --mode capture --source <URL>
```

Presiona `q` para salir de la visualización.

### Modo de Seguimiento de la Marcha

Para realizar el seguimiento de la marcha en tiempo real, ejecuta el siguiente comando:

```bash

python main.py --mode track --source 0
```

Para usar la cámara de tu celular, proporciona la URL de la cámara:

```bash
python main.py --mode track --source <URL>
```

Presiona `q` para salir de la visualización.

### Modo de Análisis en Tiempo Real de la Marcha

Para realizar un análisis en tiempo real de los datos de la marcha capturados, ejecuta el siguiente comando:

```bash
python main.py --mode analyze --source 0
```

Para usar la cámara de tu celular, proporciona la URL de la cámara:

```bash
python main.py --mode analyze --source <URL>
```

Presiona `q` para salir de la visualización.

## Recursos y Material Adicional

- [Documentación de MediaPipe](https://google.github.io/mediapipe/)
- [Documentación de OpenCV](https://opencv.org/)
- [Documentación de TensorFlow](https://www.tensorflow.org/)

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o envía un pull request con tus mejoras o sugerencias.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

Este archivo README proporciona una guía clara y detallada sobre cómo usar los scripts y recursos del repositorio para el taller de análisis de la marcha.