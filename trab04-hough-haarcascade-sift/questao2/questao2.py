# Alunos:
# - Gustavo Baroni Bruder
# - Ana Carolina da Silva

import cv2 as cv2
from matplotlib import pyplot as plt


carros_cascade = cv2.CascadeClassifier("dataset/cars.xml")
carros_video_cap = cv2.VideoCapture("dataset/cars.avi")

qtd_frames = 0

while carros_video_cap.isOpened():
    ret, frame = carros_video_cap.read()
    if not ret:
        break

    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fator_escala = 1.03
    min_vizinhos = 3
    carros = carros_cascade.detectMultiScale(frame_cinza, fator_escala, min_vizinhos)

    for (x, y, largura, altura) in carros:
        cor_verde_rgb = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + largura, y + altura), cor_verde_rgb, 2)

    qtd_frames += 1
    plt.title("Frame #" + str(qtd_frames) + "\nQuantidade de carros identificados: " + str(len(carros)))
    plt.imshow(frame, 'gray')
    plt.show()

    if cv2.waitKey(1) == ord('q'):
        break

carros_video_cap.release()
cv2.destroyAllWindows()
