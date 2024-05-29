# Alunos:
# - Gustavo Baroni Bruder
# - Ana Carolina da Silva

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


ORB = cv.ORB_create()
BF = cv.BFMatcher(cv.NORM_L2, crossCheck=True)


def calcular_correspondencias_kp(imagem1, imagem2):
    kp1, des1 = ORB.detectAndCompute(imagem1, None)
    kp2, des2 = ORB.detectAndCompute(imagem2, None)

    correspondencias = BF.match(des1, des2)
    correspondencias = sorted(correspondencias, key=lambda _: _.distance)

    return kp1, kp2, correspondencias


def calcular_homografia(correspondencias, kp1, kp2):
    pontos_origem = np.float32([kp1[m.queryIdx].pt for m in correspondencias]).reshape(-1, 2)
    pontos_destino = np.float32([kp2[m.trainIdx].pt for m in correspondencias]).reshape(-1, 2)

    M, _ = cv.findHomography(pontos_origem, pontos_destino, cv.RANSAC, 5.0)

    return M


def calcular_canvas(imagem1, imagem2, M):
    h1, w1 = imagem1.shape[:2]
    h2, w2 = imagem2.shape[:2]

    cantos_imagem1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    cantos_imagem1_deformados = cv.perspectiveTransform(cantos_imagem1, M)
    cantos = np.concatenate((cantos_imagem1_deformados, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)), axis=0)

    [x_min, y_min] = np.int32(cantos.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(cantos.max(axis=0).ravel() + 0.5)
    distancia_translacao = [-x_min, -y_min]

    return x_min, y_min, x_max, y_max, distancia_translacao


def obter_juncao_imagens(imagem1, imagem2, M, x_min, y_min, x_max, y_max, distancia_translacao):
    h_out = y_max - y_min
    w_out = x_max - x_min

    M_translate = np.array([[1, 0, distancia_translacao[0]], [0, 1, distancia_translacao[1]], [0, 0, 1]])
    M = M_translate @ M
    imagem1_deformada = cv.warpPerspective(imagem1, M, (w_out, h_out))

    imagem2_transladada = cv.warpAffine(imagem2, np.float32([[1, 0, distancia_translacao[0]], [0, 1, distancia_translacao[1]]]), (w_out, h_out))
    canvas = imagem2_transladada.copy()

    canvas[imagem1_deformada > 0] = imagem1_deformada[imagem1_deformada > 0]

    juncao_imagens = cv.addWeighted(canvas, 0.5, imagem2_transladada, 0.5, 0)

    return juncao_imagens


def exibir_resultado(resultado):
    plt.imshow(resultado)
    plt.axis('off')
    plt.show()


imagens = ['dataset/estante1.png', 'dataset/estante2.png', 'dataset/estante3.png']
juncao_imagens = None

for i in range(len(imagens) - 1):
    if juncao_imagens is None:
        imagem1 = cv.imread(imagens[i], cv.IMREAD_COLOR)
    else:
        imagem1 = juncao_imagens

    imagem2 = cv.imread(imagens[i + 1], cv.IMREAD_COLOR)

    kp1, kp2, correspondencias = calcular_correspondencias_kp(imagem1, imagem2)
    M = calcular_homografia(correspondencias, kp1, kp2)
    x_min, y_min, x_max, y_max, distancia_translacao = calcular_canvas(imagem1, imagem2, M)
    juncao_imagens = obter_juncao_imagens(imagem1, imagem2, M, x_min, y_min, x_max, y_max, distancia_translacao)

exibir_resultado(juncao_imagens)
