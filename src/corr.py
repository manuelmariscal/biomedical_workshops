# corr.py

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_datasets(data_folder='data'):
    """
    Carga todos los archivos CSV de datos en la carpeta especificada.
    """
    data_folder_path = os.path.abspath(data_folder)
    csv_files = glob.glob(os.path.join(data_folder_path, '*.csv'))
    datasets = []

    if not csv_files:
        print(f"No se encontraron archivos CSV en la carpeta especificada: {data_folder_path}")
        return datasets

    valid_count = 0
    invalid_count = 0
    unknown_count = 0

    for file in csv_files:
        data = load_data_from_csv(file)
        if data is not None:
            basename = os.path.basename(file)
            # Determinar la etiqueta basada en el nombre del archivo
            if 'INVALIDO' in basename.upper():
                label = 'INVALID'
                invalid_count += 1
                identifier = f'INVALID - {invalid_count}'
            elif 'VALIDO' in basename.upper():
                label = 'VALID'
                valid_count += 1
                identifier = f'VALID - {valid_count}'
            else:
                label = 'UNKNOWN'
                unknown_count += 1
                identifier = f'UNKNOWN - {unknown_count}'
            datasets.append((identifier, data))
    print(f"Se cargaron {len(datasets)} conjuntos de datos.")
    return datasets

def load_data_from_csv(filename):
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

def trim_datasets_to_smallest(datasets):
    """
    Recorta todos los conjuntos de datos al tamaño del número de frames del archivo más pequeño.
    """
    min_length = min(len(data) for _, data in datasets)
    trimmed_datasets = []
    for name, data in datasets:
        trimmed_data = data.iloc[:min_length].reset_index(drop=True)
        trimmed_datasets.append((name, trimmed_data))
    return trimmed_datasets

def compute_correlation_matrix(data1, data2):
    """
    Calcula la matriz de correlación entre dos conjuntos de datos.
    """
    cols = [col for col in data1.columns if '_x' in col or '_y' in col]
    data1_positions = data1[cols]
    data2_positions = data2[cols]

    corr_matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for col1 in cols:
        for col2 in cols:
            series1 = data1_positions[col1]
            series2 = data2_positions[col2]
            # Excluir NaN para correlación
            valid_idx = series1.notna() & series2.notna()
            if valid_idx.any():
                corr = series1[valid_idx].corr(series2[valid_idx])
                corr_matrix.loc[col1, col2] = corr
            else:
                corr_matrix.loc[col1, col2] = np.nan

    return corr_matrix

def abbreviate_keypoint(keypoint):
    """
    Abrevia el nombre de un punto clave.
    """
    parts = keypoint.split('_')
    abbreviations = {
        'nose': 'N',
        'left': 'L',
        'right': 'R',
        'shoulder': 'S',
        'elbow': 'E',
        'wrist': 'W',
        'hip': 'H',
        'knee': 'K',
        'ankle': 'A',
        'eye': 'E',
        'ear': 'Ea',
        'x': 'x',
        'y': 'y'
    }
    abbreviated_parts = [abbreviations.get(part, part[0].upper()) for part in parts]
    return ''.join(abbreviated_parts)

def plot_correlation_matrices(datasets):
    """
    Genera y grafica las matrices de correlación para cada par de movimientos.
    """
    num_pairs = len(datasets) * (len(datasets) - 1) // 2
    cols = 3  # Número de columnas en la matriz de subplots
    rows = (num_pairs + cols - 1) // cols  # Calcular filas necesarias

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    axes = axes.flatten()

    pair_idx = 0
    cmap_blue = plt.cm.Blues  # Mapa de color con tonos azules

    for i, (name1, data1) in enumerate(datasets):
        for j, (name2, data2) in enumerate(datasets):
            if i < j:
                corr_matrix = compute_correlation_matrix(data1, data2)
                im = axes[pair_idx].imshow(corr_matrix.values.astype(float), cmap=cmap_blue, vmin=-1, vmax=1)
                keypoint_labels = [abbreviate_keypoint(col) for col in corr_matrix.columns]
                axes[pair_idx].set_xticks(range(len(keypoint_labels)))
                axes[pair_idx].set_yticks(range(len(keypoint_labels)))
                axes[pair_idx].set_xticklabels(keypoint_labels, rotation=90, fontsize=8)
                axes[pair_idx].set_yticklabels(keypoint_labels, fontsize=8)
                axes[pair_idx].set_title(f'{name1} vs {name2}', fontsize=10)
                fig.colorbar(im, ax=axes[pair_idx], fraction=0.046, pad=0.04)
                pair_idx += 1

    # Eliminar subplots no utilizados
    for idx in range(pair_idx, len(axes)):
        fig.delaxes(axes[idx])

    fig.suptitle('Matrices de Correlación entre Pares de Movimientos', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()  # No mostrar aún para poder mostrar ambas figuras al mismo tiempo
    return fig  # Retornar la figura

def compute_average_keypoint_correlations(datasets):
    """
    Calcula las correlaciones promedio por punto clave entre todos los pares de movimientos.
    """
    keypoint_corrs_sum = {}
    keypoint_corrs_count = {}

    for i, (_, data1) in enumerate(datasets):
        for j, (_, data2) in enumerate(datasets):
            if i < j:
                cols = [col for col in data1.columns if '_x' in col or '_y' in col]
                min_length = min(len(data1), len(data2))
                data1_positions = data1[cols].iloc[:min_length]
                data2_positions = data2[cols].iloc[:min_length]
                for col in cols:
                    series1 = data1_positions[col]
                    series2 = data2_positions[col]
                    valid_idx = series1.notna() & series2.notna()
                    if valid_idx.any():
                        corr = series1[valid_idx].corr(series2[valid_idx])
                        if not np.isnan(corr):
                            if col in keypoint_corrs_sum:
                                keypoint_corrs_sum[col] += corr
                                keypoint_corrs_count[col] += 1
                            else:
                                keypoint_corrs_sum[col] = corr
                                keypoint_corrs_count[col] = 1

    avg_keypoint_corrs = {k: keypoint_corrs_sum[k] / keypoint_corrs_count[k] for k in keypoint_corrs_sum}
    return avg_keypoint_corrs

def plot_skeleton_heatmap(avg_keypoint_corrs):
    """
    Grafica el esqueleto con un mapa de calor basado en las correlaciones promedio por punto clave.
    """
    # Obtener los keypoints y conexiones
    keypoints = list(set([col[:-2] for col in avg_keypoint_corrs.keys()]))
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

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 6))

    # Simular posiciones promedio de los keypoints
    positions = {}
    total_positions = {}
    count_positions = {}
    for _, data in datasets:
        for kp in keypoints:
            x_col = f'{kp}_x'
            y_col = f'{kp}_y'
            if x_col in data.columns and y_col in data.columns:
                x_vals = data[x_col].dropna()
                y_vals = data[y_col].dropna()
                if not x_vals.empty and not y_vals.empty:
                    avg_x = x_vals.mean()
                    avg_y = y_vals.mean()
                    if kp in total_positions:
                        total_positions[kp][0] += avg_x
                        total_positions[kp][1] += avg_y
                        count_positions[kp] += 1
                    else:
                        total_positions[kp] = [avg_x, avg_y]
                        count_positions[kp] = 1
    for kp in total_positions:
        positions[kp] = (total_positions[kp][0] / count_positions[kp], total_positions[kp][1] / count_positions[kp])

    # Normalizar las correlaciones para el mapa de calor
    corrs = []
    for kp in keypoints:
        x_corr = avg_keypoint_corrs.get(f'{kp}_x', 0)
        y_corr = avg_keypoint_corrs.get(f'{kp}_y', 0)
        corrs.append((x_corr + y_corr) / 2)
    corrs = np.array(corrs)
    vmin = -1  # Valor mínimo para normalización
    vmax = 1   # Valor máximo para normalización
    norm_corrs = (corrs - vmin) / (vmax - vmin)
    keypoint_colors = {kp: plt.cm.Blues(norm_corrs[i]) for i, kp in enumerate(keypoints)}

    # Dibujar keypoints
    for kp, (x, y) in positions.items():
        color = keypoint_colors.get(kp, (0, 0, 0, 1))
        ax.scatter(x, y, color=color, s=200, edgecolors='black')  # Tamaño aumentado y borde para mejor visibilidad
        ax.text(x, y - 5, abbreviate_keypoint(kp), fontsize=10, ha='center', va='top')  # Etiquetas abreviadas

    # Dibujar conexiones
    for p1, p2 in connections:
        if p1 in positions and p2 in positions:
            x1, y1 = positions[p1]
            x2, y2 = positions[p2]
            ax.plot([x1, x2], [y1, y2], color='darkgray', linewidth=4)

    # Ajustar los límites de los ejes
    all_x = [pos[0] for pos in positions.values()]
    all_y = [pos[1] for pos in positions.values()]
    ax.set_xlim(min(all_x) - 20, max(all_x) + 20)
    ax.set_ylim(max(all_y) + 20, min(all_y) - 20)  # Invertimos los límites del eje Y

    # Añadir colorbar
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Correlación promedio')

    ax.set_xlabel('Posición X')
    ax.set_ylabel('Posición Y')
    ax.set_title('Mapa de calor del esqueleto basado en correlaciones promedio')
    plt.tight_layout()
    # plt.show()  # No mostrar aún para poder mostrar ambas figuras al mismo tiempo
    return fig  # Retornar la figura

def main(data_folder='data'):
    global datasets  # Hacer datasets global para usar en otras funciones
    datasets = load_datasets(data_folder)
    if not datasets:
        return

    # Recortar los conjuntos de datos al tamaño del archivo más pequeño
    datasets = trim_datasets_to_smallest(datasets)

    # Generar y mostrar las matrices de correlación
    fig_corr_matrices = plot_correlation_matrices(datasets)

    # Calcular correlaciones promedio por punto clave
    avg_keypoint_corrs = compute_average_keypoint_correlations(datasets)

    # Generar y mostrar el mapa de calor del esqueleto
    fig_skeleton_heatmap = plot_skeleton_heatmap(avg_keypoint_corrs)

    # Mostrar ambas figuras al mismo tiempo
    plt.show()

# Eliminamos la siguiente línea para que el script no se ejecute al ser importado
# if __name__ == "__main__":
#     main()
