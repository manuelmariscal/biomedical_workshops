# main.py

import argparse
import os
import sys

# Asegurar que el directorio 'src' y el directorio actual están en el path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.dirname(__file__))

from src.tracking import Tracking
from src.analysis import MovementAnalysis  # Si aún usas MovementAnalysis
import corr  # Importamos corr.py

def main():

    # ASCII ART de bienvenida
    print(r"""                                                                            
    (           (  (                  )         )                  
    ( )\ (        )\))(   '     (    ( /(      ( /(                  
    )((_))\   (  ((_)()\ )  (   )(   )\()) (   )\())  (   `  )   (   
    ((_)_((_)  )\ _(())\_)() )\ (()\ ((_)\  )\ ((_)\   )\  /(/(   )\  
    | _ )(_) ((_)\ \((_)/ /((_) ((_)| |(_)((_)| |(_) ((_)((_)_\ ((_) 
    | _ \| |/ _ \ \ \/\/ // _ \| '_|| / / (_-<| ' \ / _ \| '_ \)(_-< 
    |___/|_|\___/  \_/\_/ \___/|_|  |_\_\ /__/|_||_|\___/| .__/ /__/ 
                                                        |_|         
    """)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs innecesarios de TensorFlow

    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Análisis de Movimiento para Ingenieros Biomédicos')
    parser.add_argument('--mode', type=str, choices=['track', 'analyze', 'corr'], default='track',
                        help='Modo de operación: "track" para captura de datos, "analyze" para análisis de datos, "corr" para correlación de datos')
    parser.add_argument('--source', type=str, default='0',
                        help='Fuente de video: 0 para cámara de laptop, URL para cámara externa (solo en modo "track")')
    parser.add_argument('--data_folder', type=str, default='data',
                        help='Carpeta donde se almacenan los archivos CSV de datos (solo en modos "analyze" y "corr")')
    args = parser.parse_args()

    if args.mode == 'track':
        # Ejecutar el código de tracking
        tracking = Tracking(source=args.source)
        tracking.run_analysis()
    elif args.mode == 'analyze':
        # Ejecutar el código de análisis
        analysis = MovementAnalysis(data_folder=args.data_folder)
        analysis.run_analysis()
    elif args.mode == 'corr':
        # Ejecutar el código de correlación utilizando corr.py
        corr.main(data_folder=args.data_folder)
    else:
        print('Modo no reconocido. Usa "track", "analyze" o "corr".')

if __name__ == "__main__":
    main()
