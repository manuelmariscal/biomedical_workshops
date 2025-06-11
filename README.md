# Biomedical Workshops

## Análisis de movimiento en tiempo real

Este repositorio contiene una herramienta para capturar y analizar movimientos utilizando la cámara del equipo o un dispositivo móvil. El seguimiento se realiza con Mediapipe y OpenCV y está orientado a estudiantes e investigadores de Ingeniería Biomédica. Los datos pueden procesarse en tiempo real o de forma posterior a partir de archivos guardados.

### Características

- **Seguimiento en tiempo real** usando Mediapipe y OpenCV.
- **Captura de video** desde la cámara del equipo o de un teléfono móvil.
- **Extracción de puntos clave** (tobillos, rodillas, caderas, hombros, codos, muñecas y centro de gravedad).
- **Guardado de datos** en archivos CSV para análisis posterior.
- **Análisis de movimientos** con herramientas de correlación y visualización.
- **Matriz de correlaciones** y mapas de calor del esqueleto.
- **Interfaz gráfica opcional** mediante Dash.
- **Múltiples modos de operación**:
  - **track**: captura y seguimiento en tiempo real.
  - **analyze**: análisis de datos previamente guardados.
  - **corr**: cálculo y visualización de correlaciones entre movimientos.

### Repository Structure

```bash
biomedical_workshops/
├── main.py           # interfaz por consola
├── dash_app.py       # interfaz gráfica con Dash
├── README.md
├── requirements.txt
├── tests/            # pruebas automatizadas
└── src/
    ├── tracking.py
    ├── corr.py
    └── analysis.py
```

### Instalación

1. **Clona el repositorio**:

    ```bash
    git clone https://github.com/manuelmariscal/biomedical_workshops.git
    cd biomedical_workshops
    ```

2. **Crea un entorno virtual** y actívalo:

    ```bash
    python -m venv bioworkshops
    # En Windows:
    bioworkshops\Scripts\activate
    # En Unix o macOS:
    source bioworkshops/bin/activate
    ```

3. **Instala las dependencias**:

    ```bash
    pip install -r requirements.txt
    ```

### Uso

La herramienta funciona en tres modos: `track`, `analyze` y `corr`. Puedes elegir el modo con el argumento `--mode`.

#### Modo de seguimiento (`track`)

Para ejecutar el seguimiento en tiempo real:

```bash
python main.py --mode track --source 0
```

Donde `0` es la cámara por defecto del equipo. También puedes indicar una URL o el índice de otra cámara.

#### Modo de análisis (`analyze`)

Para analizar datos almacenados:

```bash
python main.py --mode analyze --data_folder data
```

Este modo procesa los archivos CSV de la carpeta indicada y genera gráficos y métricas.

#### Modo de correlación (`corr`)

Para calcular y visualizar correlaciones entre movimientos:

```bash
python main.py --mode corr --data_folder data
```

El programa calcula matrices de correlación y muestra mapas de calor del esqueleto utilizando los archivos CSV encontrados en `data_folder`.

#### Argumentos principales

- `--mode`: modo de operación (`track`, `analyze` o `corr`).
- `--source`: fuente de video (por ejemplo `0` para la cámara local). Sólo en el modo `track`.
- `--data_folder`: carpeta donde se guardan o leen los archivos CSV. Se usa en `analyze` y `corr`.

### Análisis de datos

La aplicación guarda las coordenadas de los puntos clave en archivos CSV dentro de `data_folder`. A partir de ellos se generan distintas visualizaciones según el modo elegido:

1. **Modo `track`**:
   - Muestra el esqueleto en tiempo real.
   - Guarda las coordenadas para revisarlas más adelante.

2. **Modo `analyze`**:
   - Grafica los movimientos registrados.
   - Genera animaciones para comparar ejecuciones válidas e inválidas.
   - Calcula correlaciones entre movimientos.

3. **Modo `corr`**:
   - Muestra matrices de correlación entre pares de movimientos.
   - Dibuja un mapa de calor del esqueleto con las correlaciones promedio.
   - Las gráficas se muestran juntas para facilitar la interpretación.

### Ejemplo rápido

Para ejecutar el modo `corr` analizando los archivos dentro de `data`:

```bash
python main.py --mode corr --data_folder data
```

### Interfaz gráfica con Dash

Puedes lanzar una pequeña aplicación web para elegir el modo de ejecución sin utilizar la consola:

```bash
python dash_app.py
```

El servidor se inicia en `http://127.0.0.1:8050/` y desde allí podrás seleccionar el modo, la carpeta de datos y el número máximo de frames.

### Ejecutar pruebas

El repositorio incluye pruebas con `pytest`. Para ejecutarlas:

```bash
pytest
```

### Contribuciones

¡Se agradecen las contribuciones! Abre un issue o un pull request y procura incluir pruebas cuando sea posible.

### Licencia

Este proyecto está disponible bajo la licencia MIT.

### Agradecimientos

Este proyecto utiliza las siguientes bibliotecas:

- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [SciPy](https://www.scipy.org/)

---

Para más información visita el [repositorio del proyecto](https://github.com/manuelmariscal/biomedical_workshops).
