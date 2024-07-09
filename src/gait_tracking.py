import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def track_gait():
    cap = cv2.VideoCapture(0)  # Captura desde la cámara por defecto
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()  # Leer un frame de la cámara
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir a RGB
            results = pose.process(image)  # Procesar la imagen con MediaPipe Pose
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convertir de nuevo a BGR
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # Dibujar las landmarks
            cv2.imshow('Gait Tracking', image)  # Mostrar el video con las landmarks
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Salir si se presiona 'q'
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_gait()
