import argparse
from src.analysis_keras import GaitAnalysis
import os

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Para suprimir logs de TensorFlow
    parser = argparse.ArgumentParser(description='Gait Analysis for Biomedical Engineers')
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for laptop camera, URL for mobile camera')
    parser.add_argument('--test', action='store_true', help='Use saved data for testing')
    args = parser.parse_args()
    
    source = int(args.source) if args.source.isdigit() else args.source
    ga = GaitAnalysis(source=source, test=args.test)
    ga.run_analysis(duration=30)
    ga.save_and_display_movie(start_time=5, end_time=25)

if __name__ == "__main__":
    main()
