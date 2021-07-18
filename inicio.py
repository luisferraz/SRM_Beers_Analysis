'''! Processamento da tabela SRM'''
# * 1 - Carregar a imagem da tabela do computador
# * 2 - processar cada cor do SRM
# *      -> Carregar a imagem da cor
# *      -> Converter RGB para LAB (ou ja carregar em LAB - ver se é possivel)
# *      -> Salvar SRM como chave + Array LAB como valor
# * 3 - Mapear os estilos e valor SRM MIN/MAX da planilha
# * 4 - Criar estrutura de comparação:
# *      -> Estilo da cerveja
# *      -> ARRAY SRM para valor MIN
# *      -> ARRAY SRM para valor MAX
# *

#from parte2 import getDominantLABColor
import cv2
import os
import csv
import numpy as np

# Classe que mapeia um nivel SRM a partir da imagem que o representa e salva seus valores convertidos para LAB


def getDominantLABColor(img):
    img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data, 1, None, criteria, 10, flags)

    dominantColor = centers[0].astype(np.float32)
    return dominantColor


class SRMColor (object):
    def __init__(self, index):
        # Valor SRM da Cor
        self.index = index
        # A cor (uma tupla LAB carregada da imagem)
        self.colorValue = self.processSRMColor(index)

    def processSRMColor(self, index):
        filename = os.path.abspath(os.getcwd()) + \
            "/SRM_BASE/SRM_{0}.PNG".format(index)
        if not os.path.isfile(filename):
            return None

        img = cv2.imread(filename)
        dominantColor = getDominantLABColor(img)
        return dominantColor


# Classe que compoe um estilo de cerveja, com seu nome, SRM Minimo e Maximo, utilizado para a comparacao


class BeerStyle(object):
    def __init__(self, name, min_srm, max_srm):
        self.name = name
        self.minSRM = SRMColor(round(float(min_srm)))
        self.maxSRM = SRMColor(round(float(max_srm)))

# Carrega o arquivo CSV com os estilos de Cerveja em um dicionario


def loadBeersStyles():
    file = open(os.path.abspath(os.getcwd()) +
                "/SRM_BASE/SRM Beers Values.csv")
    csvFile = csv.DictReader(file, delimiter=";")
    return csvFile


# * Carrega a foto de uma cerveja, retornando a imagem original e em escala de cinza
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

# * Rotina para deteccao de borda
# * Utiliza um filtro Gaussiano com escalar alto ou baixo
# * Para então por um Threshold utilizando o algoritmo de Otsu
# * Por fim, executa o algoritmo de Canny para encontrar a borda da imagem que representa a area ocupada pela cerveja


def dedectaBorda(imagem, alto):
    filtrada = cv2.GaussianBlur(imagem, ((21, 21) if alto else (3, 3)), 0)
    #filtrada = cv2.medianBlur(imagem, 7)
    otsuValue, limiarizada = cv2.threshold(
        filtrada, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    edge = cv2.Canny(limiarizada, 100, 200)
    return edge, otsuValue

# * Rotina que recebe uma imagem original


def extractBeerArea(imagem, cinza):
    borda, limiarOtsu = dedectaBorda(cinza, True)
    imgFilled = np.zeros_like(borda)
    contornos = cv2.findContours(
        borda, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
    maxContorno = max(contornos, key=cv2.contourArea)

    cv2.drawContours(imgFilled, [maxContorno], 0, 255, thickness=cv2.FILLED)

    return imgFilled, limiarOtsu


if __name__ == "__main__":
    # ? PARTE 1 - PROCESSAMENTO DA BASE SRM
    # * Principais estilos de cerveja e seus valores SRM Minimo e Máximo

    # Carregamos o arquivo CSV com os estilos de cerveja
    base_file = loadBeersStyles()

    # Entao, para cada estilo, criamos um parametro de comparacao - SRM MIN e SRM MAX
    BeerStyles = [BeerStyle(line["Estilo"], line["MIN_SRM"],
                            line["MAX_SRM"]) for line in base_file]

    [print(i.name, i.minSRM.index, i.minSRM.colorValue,
           i.maxSRM.index, i.maxSRM.colorValue) for i in BeerStyles]
