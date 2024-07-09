import argparse
from src.capture_video import VideoCapture
from src.gait_tracking import GaitTracking
from src.realtime_gait_analysis import RealTimeGaitAnalysis

def main():
    parser = argparse.ArgumentParser(description='Gait Analysis for Biomedical Engineers')
    parser.add_argument('--mode', type=str, choices=['capture', 'track', 'analyze'], required=True,
                        help='Mode of operation: capture, track, analyze')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: 0 for laptop camera, URL for mobile camera')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    if args.debug:
        print(f"Running in {args.mode} mode with source: {source}")

    if args.mode == 'capture':
        vc = VideoCapture(source)
        vc.start_capture()
    elif args.mode == 'track':
        gt = GaitTracking(source)
        gt.start_tracking()
    elif args.mode == 'analyze':
        rta = RealTimeGaitAnalysis(source)
        rta.analyze_gait()
    else:
        print("Invalid mode selected")

if __name__ == "__main__":
    main()
