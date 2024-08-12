import argparse
from src.gait_analysis_new import GaitAnalysis
import os

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprimir logs innecesarios de TensorFlow

    # Configuración de argumentos
    parser = argparse.ArgumentParser(description='Gait Analysis for Biomedical Engineers')
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for laptop camera, URL for mobile camera')
    parser.add_argument('--test', action='store_true', help='Use saved data for testing')
    args = parser.parse_args()

    # Determinar la fuente de video: cámara o URL
    source = int(args.source) if args.source.isdigit() else args.source

    # Crear una instancia de GaitAnalysis
    ga = GaitAnalysis(source=source, test=args.test)

    # Ejecutar el análisis de marcha
    ga.run_analysis(duration=30)

    # Guardar y mostrar la película del análisis
    ga.save_and_display_movie(start_time=5, end_time=25)

if __name__ == "__main__":
    main()
