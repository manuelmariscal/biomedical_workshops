import cv2
import numpy as np

class MotionAnalysis:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.tracker = cv2.TrackerKCF_create()
        self.tracking = False
        self.bbox = None

    def run_analysis(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            # Aplicar sustracción de fondo para detectar movimiento
            fgmask = self.fgbg.apply(frame)

            # Aplicar operaciones morfológicas para reducir ruido
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)

            # Encontrar contornos en la máscara de primer plano
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 1000:
                    # Si el área del contorno es suficientemente grande, considerarlo un objeto
                    x, y, w, h = cv2.boundingRect(contour)

                    if not self.tracking:
                        # Inicializar seguimiento
                        self.bbox = (x, y, w, h)
                        self.tracker.init(frame, self.bbox)
                        self.tracking = True
                    else:
                        # Actualizar seguimiento
                        self.tracking, self.bbox = self.tracker.update(frame)
                        if self.tracking:
                            x, y, w, h = map(int, self.bbox)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostrar el resultado
            cv2.imshow('Motion Tracking', frame)

            # Salir del programa cuando se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Liberar recursos
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analysis = MotionAnalysis(source='your_video.mp4')  # Cambia 'your_video.mp4' por la ruta de tu video
    analysis.run_analysis()
