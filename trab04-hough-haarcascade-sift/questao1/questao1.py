# Alunos:
# - Gustavo Baroni Bruder
# - Ana Carolina da Silva

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def identificar_iris(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_borrada = cv2.medianBlur(imagem_cinza, 5)
    _, imagem_binaria = cv2.threshold(imagem_borrada, 100, 255, cv2.THRESH_BINARY)

    circulos_iris = cv2.HoughCircles(imagem_binaria, cv2.HOUGH_GRADIENT, dp=2, minDist=2000, param1=15, param2=25, minRadius=80, maxRadius=200)
    circulos_iris = np.uint16(np.around(circulos_iris))

    mascara_iris = np.zeros_like(imagem)

    for circulo_iris in circulos_iris[0, :]:
        cor_branca_rgb = (255, 255, 255)
        cv2.circle(mascara_iris, (circulo_iris[0], circulo_iris[1]), circulo_iris[2], cor_branca_rgb, -1)

    imagem_iris_cortada_fundo = cv2.bitwise_and(imagem, mascara_iris)
    return imagem_iris_cortada_fundo


def remover_pupila(imagem):
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem_borrada = cv2.medianBlur(imagem_cinza, 5)
    _, imagem_binaria = cv2.threshold(imagem_borrada, 45, 255, cv2.THRESH_BINARY)

    circulos = cv2.HoughCircles(imagem_binaria, cv2.HOUGH_GRADIENT, dp=2, minDist=500, param1=15, param2=25, minRadius=15, maxRadius=70)
    circulos = np.uint16(np.around(circulos))

    mascara_pupila = np.full_like(imagem, 255)

    for circulo in circulos[0, :]:
        cor_preta_rgb = (0, 0, 0)
        cv2.circle(mascara_pupila, (circulo[0], circulo[1]), circulo[2], cor_preta_rgb, -1)

    imagem_iris_cortada_pupila = cv2.bitwise_and(imagem, mascara_pupila)
    return imagem_iris_cortada_pupila


diretorio_dataset = "dataset"

for arquivo in os.listdir(diretorio_dataset):
    diretorio_imagem = os.path.join(diretorio_dataset, arquivo)
    imagem = cv2.imread(diretorio_imagem)

    imagem_iris = identificar_iris(imagem)
    imagem_sem_pupila = remover_pupila(imagem_iris)

    plt.subplot(1, 1, 1)
    plt.title(arquivo + " - Apenas iris")
    plt.imshow(imagem_sem_pupila)
    plt.show()
