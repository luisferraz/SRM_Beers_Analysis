import cv2
import os
import csv
import numpy as np

# * Função recebe uma imagem e retorna a sua cor dominante no colorspace LAB


def getDominantColor(img):
    #img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data, 1, None, criteria, 10, flags)

    dominantColor = centers[0].astype(np.float32)

    return dominantColor


# * Classe que mapeia um nivel SRM a partir da imagem que o representa e salva sua cor para comparacao
class SRMColor (object):
    def __init__(self, index):
        # Valor SRM da Cor
        self.index = index
        # A cor (uma tupla BGR carregada da imagem)
        self.colorValue = self.processSRMColor(index)

    def processSRMColor(self, index):
        filename = os.path.abspath(os.getcwd()) + \
            "/SRM_BASE/SRM_{0}.PNG".format(index)
        if not os.path.isfile(filename):
            return None

        img = cv2.imread(filename)
        dominantColor = getDominantColor(img)
        return dominantColor


# * Classe que compoe um estilo de cerveja, com seu nome e SRM, que sera utilizado para a comparacao
class BeerStyle(object):
    def __init__(self, name, min_srm, max_srm):
        self.name = name
        self.SRM = SRMColor(round((float(max_srm) + float(min_srm)) / 2))

# * Salva uma imagem em disco na pasta imagensGeradas
def salvaImg(img, name):
    directory = os.path.join (os.path.abspath(os.getcwd()), "imagensGeradas")
    if not os.path.exists(directory):
        os.makedirs(directory);
    name = directory + "/{0}.png".format(name)
    cv2.imwrite(name, img)

# * Carrega o arquivo CSV com os estilos de Cerveja em um dictionary
def loadBeersStyles():
    file = open(os.path.abspath(os.getcwd()) +
                "/SRM_BASE/SRM Beers Values.csv")
    csvFile = csv.DictReader(file, delimiter=";")
    return csvFile


# * Carrega a foto de uma cerveja, retornando a imagem original e em escala de cinza
def carregaFotoCerveja(fileName):
    img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)

    # Como as imagens estao sendo gravadas em 4032x3024, reduzimos para 20% do tamanho
    scale = 20
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return resized, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


# * Rotina para aplicacao do filtro 
# * Utiliza um filtro Gaussiano com escalar alto ou baixo (parametrizavel) ou filtro de Mediana
# * Retorna a imagem filtrada
def filtraImg(imagem, escalar, filtro):
    if (filtro == "Gaussiano"):
        filtrada = cv2.GaussianBlur(imagem, escalar, 0)
    else:
        filtrada = cv2.medianBlur(imagem, 7)
    
    return filtrada



# * Rotina que executa um Threshold utilizando o algoritmo de Otsu
# * Retorna o Limiar que o algoritmo Otsu utilizou e a imagem limiarizada
def limiarizaImg (img):
    otsuValue, limiarizada = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return otsuValue, limiarizada

# * Por fim, executa o algoritmo de Canny para encontrar a borda da imagem que representa a area ocupada pela cerveja
# * Retorna a imagem com a detecção da borda
def detectaBorda(img):
    edge = cv2.Canny(img, 100, 200)
    return edge

# * Rotina que recebe uma imagem que passou pelo processo de deteccao de borda e retorna
# * uma mascara com a area de interesse que
def extractBeerArea(imagem):
    imgFilled = np.zeros_like(imagem)
    contornos = cv2.findContours(
        imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = contornos[0] if len(contornos) == 2 else contornos[1]
    maxContorno = max(contornos, key=cv2.contourArea)

    cv2.drawContours(imgFilled, [maxContorno], 0, 255, thickness=cv2.FILLED)

    return imgFilled

def xyz2srgb(tristimulus, scale=True):
   m = np.matrix([[ 3.2404542, -1.5371385, -0.4985314],
                     [-0.9692660,  1.8760108,  0.0415560],
                     [ 0.0556434, -0.2040259,  1.0572252]])

   XYZ = np.matrix(tristimulus).T
   if scale:
      XYZ = XYZ / 100.0

   rgbLinear = m * XYZ

   sRGB = rgbLinear

   index = np.where(rgbLinear <= 0.0031308)
   if len(index[0]) != 0:
      sRGB[index] = rgbLinear[index] * 12.92

   index = np.where(rgbLinear > 0.0031308)
   if len(index[0]) != 0:
      a = 0.055
      sRGB[index] = (1 + a) * np.array(rgbLinear[index])**(1 / 2.4) - a

   sRGB = np.clip(sRGB, 0, 1)

   return sRGB[0,0], sRGB[1,0], sRGB[2,0]

#* Funcao que calcula a distancia entre duas cores
#* Recebe as cores RGB
#* Converte as cores para LAB (colorspace device independent)
#* Calcula a distancia Euclidiana entre as duas cores
#* Retorna o valor
def deltaE(color1, color2, maxCount=255):
    if type(color1).__module__ != np.__name__:
        c1 = np.asarray(color1)
    else:
        c1 = color1

    if type(color2).__module__ != np.__name__:
        c2 = np.asarray(color2)
    else:
        c2 = color2

    # Convert the provided colors to single-precision floating-point
    # values in the range [0,1] for computation
    c1 = c1.astype(np.float32) / maxCount
    c2 = c2.astype(np.float32) / maxCount

    # Check that the dimensional shape of the provided colors are the same
    dimensions1 = np.shape(c1)
    dimensions2 = np.shape(c2)
    if dimensions1 != dimensions2:
        raise ValueError('Provided datasets must have the same shape')

    # Make sure the provided colors are presented as an image to the color
    # conversion routine
    if len(dimensions1) == 1 and dimensions1[-1] == 3:
        c1 = np.asarray(c1).reshape((1, 1, 3))
        c2 = np.asarray(c2).reshape((1, 1, 3))
    elif len(dimensions1) != 3 or dimensions1[-1] != 3:
        msg = 'Provided colors must be either a 3-element vector or 3xn array'
        raise ValueError(msg)

    # Convert provided sRGB colors to L*a*b* space
    lab1 = cv2.cvtColor(c1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(c2, cv2.COLOR_BGR2LAB)

    # Compute the delta E for each of the provided color pairs
    dE = np.sqrt(np.sum((lab1 - lab2)**2, -1))

    # Return the delta E image and the average delta E value -or- the scalar
    # delta E for a pair of provided color triplets

    if dE.size == 1:
        return float(dE[0, 0])
    else:
        return dE, float(np.mean(dE))

#* Rotina para o calculo da diferenca entre a cor da area da cerveja e cor de cada estilo
#* Devolve um dictionary com o estilo, a cor da area da cerveja, cor de comparação e a diferença euclidiana entre as cores
def calculaDiferenca(valor, beerStyle: BeerStyle):
    nome = beerStyle.name
    cor1 = np.asarray(valor).reshape((1, 1, 3))
    cor2 = np.asarray(beerStyle.SRM.colorValue).reshape((1, 1, 3))
    cor1 = xyz2srgb (cv2.cvtColor(cor1, cv2.COLOR_BGR2XYZ))
    cor2 = xyz2srgb (cv2.cvtColor(cor2, cv2.COLOR_BGR2XYZ))
    diff = deltaE(cor1, cor2, 255)
    return dict(zip(["Estilo", "Valor", "Comparativo", "Diferenca"], [nome, valor, beerStyle.SRM.colorValue, diff]))

#* Funcao que, dada a lista com todos os dictionaries compostos pela função calculaDiferenca
#* Retorna o dictionary que possua o menor valor de "value"
def buscaMenorDiferenca(lista, value):
    return min(lista, key=lambda x: x[value])

#* Funcao para processamento das fotos que estao no diretorio Fotos
#* Carrega cada uma delas e executa os procedimentos para pre-processamento e analise,
#* salva as imagens intermediarias do processo e descreve o resultado
def processaComparaFotos (BeerStyles):
    for dirpath, _, files in os.walk (os.path.join(os.getcwd() + "/Fotos")):
        print (dirpath)
        for file in files:
            #? Começamos carregando a foto original e em escala de cinza do copo de cerveja 
            fotoCerveja, fotoCervejaCinza = carregaFotoCerveja(os.path.join(dirpath, file))
            salvaImg(fotoCervejaCinza, "gray_" + file)
            
            #? Entao filtramos a imagem, utilizando primeiro um filtro gaussiano com escalar baixo (3, 3)
            gauss_lo = filtraImg(fotoCervejaCinza, (3, 3), "Gaussiano")
            salvaImg(gauss_lo, "gaussiano_3x3_" + file)

            #? Gaussiano com escalar alto (21, 21)
            gauss_hi = filtraImg(fotoCervejaCinza, (21,21), "Gaussiano")
            salvaImg(gauss_hi, "gaussiano_21x21_" + file)

            #? E filtro por Mediana
            mediana = filtraImg(fotoCervejaCinza, None, "Mediana")
            salvaImg(mediana, "mediana_" + file)

            #? Vamos prosseguir com a imagem com filtro Gaussiano com escalar alto
            #? Passamos agora por um Threshold (limiarizacao) que utiliza o algoritmo de Otsu
            limiarOtsu, limiarizada = limiarizaImg(gauss_hi)
            salvaImg (limiarizada, "otsu_{0}".format(limiarOtsu) + file)

            #? Passamos entao pelo dectector de bordas que utiliza o algoritmo Canny
            #? Espera-se que a imagem retornada contenha a borda da area da cerveja
            borda = detectaBorda(limiarizada)
            salvaImg (borda, "canny_" + file)

            #? Utilizamos a borda detectada para construir a parte da imagem que vamos utilizar 
            mascaraInterese = extractBeerArea(borda)
            salvaImg (mascaraInterese, "mask_" + file)

            #? Por fim, aplicamos a mascara com a area de interesse na imagem original 
            #? para extrair a parte util para a analise
            areaInteresse = cv2.bitwise_and(fotoCerveja, fotoCerveja, mask=mascaraInterese)
            salvaImg (areaInteresse, "ROI_" + file)

            # ? PARTE 3 - COMPARACAO DA COR DA FOTO COM A BASE E DEFINICAO DO ESTILO
            # * Define a cor dominante no colorspace LAB (device independant) da area da cerveja
            # * Compara a cor dominante com as cores da base SRM para cada estilo, usando Distancia Euclidiana (DeltaE)
            # * Retorna os Estilos de cerveja provaveis para a imagem, em relação a proximidade das cores

            corDominante = getDominantColor(areaInteresse)

            diferencas = [calculaDiferenca(corDominante, i) for i in BeerStyles]

            provavelEstilo = buscaMenorDiferenca(diferencas, 'Diferenca')

            print(file, provavelEstilo["Estilo"], provavelEstilo["Diferenca"])


if __name__ == "__main__":
    # ? PARTE 1 - PROCESSAMENTO DA BASE SRM
    # * Principais estilos de cerveja e seus valores SRM Minimo e Máximo

    # Carregamos o arquivo CSV com os estilos de cerveja
    base_file = loadBeersStyles()

    # Entao, para cada estilo, criamos um parametro de comparacao - SRM MIN e SRM MAX
    BeerStyles = [BeerStyle(line["Estilo"], line["MIN_SRM"],
                            line["MAX_SRM"]) for line in base_file]

    # ? PARTE 2 - PRÉ-PROCESSAMENTO DA FOTO DA CERVEJA
    # * Carrega a foto do copo de cerveja
    # * Aplica:
    # * - Filtragem: Filtro Gaussiano (baixo e alto); Filtro por Mediana,
    # * - Limiarização: Otsu
    # * - Detecção de borda: Canny
    # * Com a borda delimitada apenas a parte da cerveja,
    # * aplica-se um AND bit a bit para extrair apenas a area de interesse
    #
    processaComparaFotos(BeerStyles)