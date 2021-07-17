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

    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def threshImg(image):
    cinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtrada = cv2.GaussianBlur(cinza, (201, 201), 0)
    t, limiarizada = cv2.threshold(
        filtrada, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    print('otsu value:{}'.format(t))
    masked = cv2.bitwise_and(image, image, mask=limiarizada)

    limiar = []
    limiar.append(filtrada)
    limiar.append(limiarizada)
    limiar.append(masked)

    for i in range(len(limiar)):
        cv2.imshow(str(i), limiar[i])

    cv2.waitKey(0)


def extractBeerArea(image):
    original = image.copy()
    imgthreshelded = threshImg(image)
    masked = cv2.bitwise_and(image, image, mask=imgthreshelded)


if __name__ == "__main__":
    # Parte 2
    # Iniciamos a segunda parte lendo a imagem da cerveja

    beerPhoto = loadBeerPhoto("nova3368.png")
    threshImg(beerPhoto)
    # extractBeerArea(beerPhoto)
    cv2.destroyAllWindows()
