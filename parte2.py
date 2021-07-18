''' Parte 2 - Carregamento da imagem '''

import cv2
import numpy as np
import os

# Carrega a foto de uma cerveja e faz o pre-processamento para a comparacao


def loadBeerPhoto(fileName):
    img = cv2.imread(os.path.abspath(os.getcwd()) +
                     "/Fotos/{0}".format(fileName), cv2.IMREAD_UNCHANGED)

    # Como as imagens estao sendo gravadas em 4032x3024, reduzimos para 30% do tamanho
    scale = 30
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def dedectaBorda(imagem, alto):
    filtrada = cv2.GaussianBlur(imagem, ((21, 21) if alto else (3, 3)), 0)
    #filtrada = cv2.medianBlur(imagem, 7)
    otsuValue, limiarizada = cv2.threshold(
        filtrada, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    edge = cv2.Canny(limiarizada, 100, 200)
    return edge, otsuValue


def extractBeerArea(imagem, cinza):
    original = imagem.copy()
    borda, limiarOtsu = dedectaBorda(cinza, True)
    imgFilled = np.zeros_like(borda)
    contornos = cv2.findContours(
        borda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
    maxContorno = max(contornos, key=cv2.contourArea)

    cv2.drawContours(imgFilled, [maxContorno], 0, 255, thickness=cv2.FILLED)

    return imgFilled, limiarOtsu


if __name__ == "__main__":
    # Parte 2
    # Iniciamos a segunda parte lendo a imagem da cerveja

    beerPhoto, grayBeerPhoto = loadBeerPhoto("nova3370.png")

    beerArea, limiarOtsu = extractBeerArea(beerPhoto, grayBeerPhoto)

    rAnd = cv2.bitwise_and(beerPhoto, beerPhoto, mask=beerArea)

    cv2.imshow("Original", beerPhoto)
    cv2.imshow("grayBeerPhoto", grayBeerPhoto)
    cv2.imshow("beerArea", beerArea)
    cv2.imshow("rAnd", rAnd)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
