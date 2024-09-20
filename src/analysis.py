# src/analysis.py

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

class MovementAnalysis:
    def __init__(self, data_folder='data'):
        """
        Inicializa la clase para el análisis de movimientos.
        """
        self.data_folder = data_folder
        self.datasets = []
        self.load_datasets()

    def load_datasets(self):
        """
        Carga todos los archivos CSV de datos en la carpeta especificada.
        """
        # Obtener la ruta absoluta del directorio de datos
        data_folder_path = os.path.abspath(self.data_folder)
        csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
        if not csv_files:
            print(f"No se encontraron archivos CSV en la carpeta especificada: {data_folder_path}")
            return

        valid_count = 0
        invalid_count = 0
        unknown_count = 0

        for file in csv_files:
            data = self.load_data_from_csv(file)
            if data is not None:
                basename = os.path.basename(file)
                # Corregimos el orden de las condiciones
                if 'INVALIDO' in basename:
                    label = 'INVALIDO'
                    invalid_count += 1
                    identifier = f'Invalido - {invalid_count}'
                elif 'VALIDO' in basename:
                    label = 'VALIDO'
                    valid_count += 1
                    identifier = f'Valido - {valid_count}'
                else:
                    label = 'DESCONOCIDO'
                    unknown_count += 1
                    identifier = f'Desconocido - {unknown_count}'
                self.datasets.append((file, data, label, identifier))
        print(f"Se cargaron {len(self.datasets)} conjuntos de datos.")

    def load_data_from_csv(self, filename):
        """
        Carga los datos de un archivo CSV.
        """
        try:
            df = pd.read_csv(filename)
            # Asegurar que los datos sean numéricos y manejar valores faltantes
            df = df.apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
            return None

    def calculate_correlations(self):
        """
        Calcula las correlaciones entre cada par de movimientos.
        """
        if not self.datasets:
            print("No hay conjuntos de datos para analizar.")
            return None

        if len(self.datasets) < 2:
            print("Se requieren al menos dos conjuntos de datos para calcular correlaciones.")
            return None

        correlation_results = []

        for i, (file1, data1, label1, _) in enumerate(self.datasets):
            for j, (file2, data2, label2, _) in enumerate(self.datasets):
                if i < j:  # Evitar duplicados y autocorrelación
                    key = f"{os.path.basename(file1)} vs {os.path.basename(file2)}"
                    corr = self.compute_correlation(data1, data2)
                    correlation_results.append({
                        'Par': key,
                        'Correlacion': corr,
                        'Etiqueta1': label1,
                        'Etiqueta2': label2
                    })

        if not correlation_results:
            print("No se encontraron pares de datos para calcular correlaciones.")
            return None

        correlation_df = pd.DataFrame(correlation_results)
        print("Tabla de correlaciones:")
        print(correlation_df)
        return correlation_df

    def compute_correlation(self, data1, data2):
        """
        Calcula la correlación promedio entre dos conjuntos de datos.
        """
        if data1 is None or data2 is None:
            return np.nan
        cols = [col for col in data1.columns if '_x' in col or '_y' in col]
        if not cols:
            return np.nan
        min_length = min(len(data1), len(data2))
        if min_length == 0:
            return np.nan

        data1_positions = data1[cols].iloc[:min_length]
        data2_positions = data2[cols].iloc[:min_length]

        correlations = []
        for col in cols:
            if col in data1_positions.columns and col in data2_positions.columns:
                col1 = data1_positions[col]
                col2 = data2_positions[col]
                # Excluir NaN para correlación
                valid_idx = col1.notna() & col2.notna()
                if valid_idx.any():
                    corr = col1[valid_idx].corr(col2[valid_idx])
                    if not np.isnan(corr):
                        correlations.append(corr)

        if correlations:
            return np.mean(correlations)
        else:
            return np.nan

    def plot_movements(self):
        """
        Grafica y anima todos los movimientos.
        """
        if not self.datasets:
            print("No hay conjuntos de datos para graficar.")
            return

        # Organizar los movimientos por categoría
        valid_movements = []
        invalid_movements = []
        unknown_movements = []

        for filename, data, label, identifier in self.datasets:
            if label == 'VALIDO':
                valid_movements.append((filename, data, label, identifier))
            elif label == 'INVALIDO':
                invalid_movements.append((filename, data, label, identifier))
            else:
                unknown_movements.append((filename, data, label, identifier))

        animations = []  # Para almacenar las animaciones y evitar que se recolecten como basura

        # Figura 1: Movimientos válidos
        if valid_movements:
            ani = self.animate_movements(valid_movements, title='Animación de Movimientos Válidos', colormap='Greens')
            animations.append(ani)

        # Figura 2: Movimientos inválidos
        if invalid_movements:
            ani = self.animate_movements(invalid_movements, title='Animación de Movimientos Inválidos', colormap='Reds')
            animations.append(ani)

        # Figura 3: Comparación de movimientos válidos e inválidos
        if valid_movements and invalid_movements:
            combined_movements = valid_movements + invalid_movements
            ani = self.animate_movements(combined_movements, title='Comparación de Movimientos Válidos e Inválidos', colormap=None)
            animations.append(ani)

        # Mostrar todas las animaciones al mismo tiempo
        plt.show()

    def animate_movements(self, movements, title, colormap=None):
        """
        Crea una animación de los movimientos proporcionados.
        """
        from matplotlib.animation import FuncAnimation
        from matplotlib.cm import get_cmap

        # Preparar los datos para la animación
        fig, ax = plt.subplots(figsize=(8, 6))

        max_frames = max(len(data) for _, data, _, _ in movements if data is not None)

        # Obtener los keypoints y conexiones
        keypoints = [col[:-2] for col in movements[0][1].columns if '_x' in col]
        connections = [
            ('left_ankle', 'left_knee'),
            ('left_knee', 'left_hip'),
            ('left_hip', 'left_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_ankle', 'right_knee'),
            ('right_knee', 'right_hip'),
            ('right_hip', 'right_shoulder'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_hip', 'right_hip'),
            ('left_shoulder', 'right_shoulder'),
        ]

        # Asignar colores
        colors = []
        if colormap:
            cmap = get_cmap(colormap)
            num_movements = len(movements)
            # Utilizar tonos más oscuros (valores más altos en el colormap)
            colors = [cmap(0.6 + 0.4 * i / max(num_movements - 1, 1)) for i in range(num_movements)]
        else:
            # Asignar colores según la etiqueta con tonos más oscuros
            colors = []
            valid_indices = [i for i, (_, _, label, _) in enumerate(movements) if label == 'VALIDO']
            invalid_indices = [i for i, (_, _, label, _) in enumerate(movements) if label == 'INVALIDO']
            num_valid = len(valid_indices)
            num_invalid = len(invalid_indices)
            cmap_valid = get_cmap('Greens')
            cmap_invalid = get_cmap('Reds')
            for i, (_, _, label, _) in enumerate(movements):
                if label == 'VALIDO':
                    idx = valid_indices.index(i)
                    color = cmap_valid(0.6 + 0.4 * idx / max(num_valid - 1, 1))
                    colors.append(color)
                elif label == 'INVALIDO':
                    idx = invalid_indices.index(i)
                    color = cmap_invalid(0.6 + 0.4 * idx / max(num_invalid - 1, 1))
                    colors.append(color)
                else:
                    colors.append('blue')

        # Crear las líneas para los esqueletos
        skeleton_lines = []
        for i, (filename, data, label, identifier) in enumerate(movements):
            lines = []
            for _ in connections:
                line, = ax.plot([], [], color=colors[i], linewidth=2)
                lines.append(line)
            skeleton_lines.append((lines, data, label, identifier))

        # Configurar los ejes basados en los datos
        all_x = []
        all_y = []
        for _, data, _, _ in movements:
            for col in data.columns:
                if '_x' in col:
                    all_x.extend(data[col].dropna().tolist())
                elif '_y' in col:
                    all_y.extend(data[col].dropna().tolist())

        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            ax.set_xlim(min_x - 20, max_x + 20)
            ax.set_ylim(max_y + 20, min_y - 20)  # Invertir el eje y para coordenadas de imagen
        else:
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)

        ax.set_xlabel('Posición X')
        ax.set_ylabel('Posición Y')
        ax.set_title(title)
        # Crear la leyenda
        handles = []
        labels = []
        for i, (_, _, label, identifier) in enumerate(movements):
            handle, = ax.plot([], [], color=colors[i], linewidth=2, label=identifier)
            handles.append(handle)
            labels.append(identifier)
        ax.legend(handles, labels)
        plt.tight_layout()

        def init():
            for lines, _, _, _ in skeleton_lines:
                for line in lines:
                    line.set_data([], [])
            return [line for lines, _, _, _ in skeleton_lines for line in lines]

        def animate(frame):
            for idx, (lines, data, label, identifier) in enumerate(skeleton_lines):
                if frame < len(data):
                    # Obtener las posiciones de los keypoints en este frame
                    positions = {}
                    for keypoint in keypoints:
                        x_col = f'{keypoint}_x'
                        y_col = f'{keypoint}_y'
                        if x_col in data.columns and y_col in data.columns:
                            x = data.iloc[frame][x_col]
                            y = data.iloc[frame][y_col]
                            if not np.isnan(x) and not np.isnan(y):
                                positions[keypoint] = (x, y)
                    # Actualizar las líneas del esqueleto
                    for line_idx, (p1, p2) in enumerate(connections):
                        if p1 in positions and p2 in positions:
                            x1, y1 = positions[p1]
                            x2, y2 = positions[p2]
                            lines[line_idx].set_data([x1, x2], [y1, y2])
                        else:
                            lines[line_idx].set_data([], [])
                else:
                    # Si no hay más frames, limpiar las líneas
                    for line in lines:
                        line.set_data([], [])
            return [line for lines, _, _, _ in skeleton_lines for line in lines]

        ani = FuncAnimation(fig, animate, frames=max_frames, init_func=init, blit=True, interval=50)

        # En lugar de plt.show(), retornamos la animación
        return ani

    def run_analysis(self):
        """
        Ejecuta todo el análisis: calcula correlaciones y grafica movimientos.
        """
        correlation_df = self.calculate_correlations()
        if correlation_df is not None and not correlation_df.empty:
            # Calcular correlaciones promedio por categoría
            valid_vs_valid = correlation_df[(correlation_df['Etiqueta1'] == 'VALIDO') & (correlation_df['Etiqueta2'] == 'VALIDO')]
            avg_corr_valid_vs_valid = valid_vs_valid['Correlacion'].mean()

            invalid_vs_invalid = correlation_df[(correlation_df['Etiqueta1'] == 'INVALIDO') & (correlation_df['Etiqueta2'] == 'INVALIDO')]
            avg_corr_invalid_vs_invalid = invalid_vs_invalid['Correlacion'].mean()

            valid_vs_invalid = correlation_df[
                ((correlation_df['Etiqueta1'] == 'VALIDO') & (correlation_df['Etiqueta2'] == 'INVALIDO')) |
                ((correlation_df['Etiqueta1'] == 'INVALIDO') & (correlation_df['Etiqueta2'] == 'VALIDO'))
            ]
            avg_corr_valid_vs_invalid = valid_vs_invalid['Correlacion'].mean()

            print("\nCorrelaciones promedio:")
            print(f"VALIDO vs VALIDO: {avg_corr_valid_vs_valid}")
            print(f"INVALIDO vs INVALIDO: {avg_corr_invalid_vs_invalid}")
            print(f"VALIDO vs INVALIDO: {avg_corr_valid_vs_invalid}")
        else:
            print("No se pudieron calcular correlaciones.")

        self.plot_movements()
