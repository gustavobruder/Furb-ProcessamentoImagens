# Alunos:
# - Gustavo Baroni Bruder
# - Ana Carolina da Silva

import cv2 as cv
import matplotlib.pyplot as plt


IMAGEM_50KM = "50km"
IMAGEM_LOMBADA = "Lombada"
IMAGEM_PARE = "Pare"

ALGORITMO_SIFT = "SIFT"
ALGORITMO_ORB = "ORB"


def obter_algoritmo(nome_algoritmo):
    algoritmo = None

    if nome_algoritmo == ALGORITMO_SIFT:
        algoritmo = cv.SIFT_create()
    elif nome_algoritmo == ALGORITMO_ORB:
        algoritmo = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2,
                                  scoreType=cv.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

    return algoritmo


def abrir_imagem(diretorio_imagem):
    imagem = cv.imread(diretorio_imagem)
    imagem = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)
    imagem_cinza = cv.cvtColor(imagem, cv.COLOR_RGB2GRAY)
    return imagem, imagem_cinza


def formatar_nome_imagem_entrada(tipo_imagem):
    return str.lower(tipo_imagem)


def formatar_nome_imagem_teste(tipo_imagem, indice):
    return f'imagem{tipo_imagem}_{indice:02}'


def exibir_resultados(nome_algoritmo, tipo_imagem, correspondencias_por_imagem):
    print(f"\n{nome_algoritmo}:\n")

    for indice_imagem in range(1, 12):
        print(f"{formatar_nome_imagem_teste(tipo_imagem, indice_imagem)} = {correspondencias_por_imagem[indice_imagem - 1]}")


def obter_correspondencias(nome_algoritmo, tipo_imagem):
    algoritmo = obter_algoritmo(nome_algoritmo)

    imagem_entrada, imagem_entrada_cinza = abrir_imagem(f"dataset/entradas/{formatar_nome_imagem_entrada(tipo_imagem)}.jpg")
    entrada_kp, entrada_desc = algoritmo.detectAndCompute(imagem_entrada_cinza, None)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    correspondencias_por_imagem = []

    for indice_imagem in range(1, 12):
        imagem_teste, imagem_teste_cinza = abrir_imagem(f"dataset/dados/{formatar_nome_imagem_teste(tipo_imagem, indice_imagem)}.jpg")
        teste_kp, teste_desc = algoritmo.detectAndCompute(imagem_teste_cinza, None)
        correspondencias = bf.knnMatch(entrada_desc, teste_desc, k=2)

        correspondencias_boas = []
        for m, n in correspondencias:
            if m.distance < 0.75 * n.distance:
                correspondencias_boas.append([m])

        correspondencias_por_imagem.append(len(correspondencias_boas))

        result = cv.drawMatchesKnn(imagem_entrada, entrada_kp, imagem_teste, teste_kp, correspondencias_boas, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) # knn
        plt.rcParams['figure.figsize'] = [14.0, 7.0]
        plt.imshow(result)
        plt.show()

    exibir_resultados(nome_algoritmo, tipo_imagem, correspondencias_por_imagem)


print(f"\nResultados para imagens de 50KM:")

obter_correspondencias(ALGORITMO_SIFT, IMAGEM_50KM)
obter_correspondencias(ALGORITMO_ORB, IMAGEM_50KM)

print(f"\nResultados para imagens de LOMBADA:")

obter_correspondencias(ALGORITMO_SIFT, IMAGEM_LOMBADA)
obter_correspondencias(ALGORITMO_ORB, IMAGEM_LOMBADA)

print(f"\nResultados para imagens de PARE:")

obter_correspondencias(ALGORITMO_SIFT, IMAGEM_PARE)
obter_correspondencias(ALGORITMO_ORB, IMAGEM_PARE)
