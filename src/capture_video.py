import cv2
import mediapipe as mp

class GaitTracking:
    def __init__(self, source=0):
        """
        Inicializa el seguimiento de la marcha.
        :param source: Fuente de video (0 para cámara de laptop, URL para cámara de celular).
        """
        self.source = source
        self.cap = None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def start_tracking(self):
        """
        Inicia el seguimiento de la marcha.
        """
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise ValueError("Error al abrir la fuente de video")
            
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error al capturar el frame")
                        break
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.pose_landmarks:
                        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    cv2.imshow('Gait Tracking', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"Error durante el seguimiento de la marcha: {e}")
        finally:
            self.release_capture()

    def release_capture(self):
        """
        Libera la captura de video.
        """
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
