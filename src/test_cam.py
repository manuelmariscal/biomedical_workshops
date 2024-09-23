import cv2

cap = cv2.VideoCapture(1)  # Usa el índice correcto para DroidCam

# Configurar resolución (ajústalo a las resoluciones disponibles en DroidCam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)  # Ajusta el ancho a 640 o según lo que prefieras
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)  # Ajusta la altura a 480 o según lo que prefieras

if not cap.isOpened():
    print("No se puede abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se puede recibir frame (final de transmisión?)")
        break

    # Mostrar el frame
    cv2.imshow('DroidCam Stream', frame)

    # Presiona 'q' para salir del loop
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
