'''! Processamento da tabela SRM'''
#* 1 - Carregar a imagem da tabela do computador
#* 2 - processar cada cor do SRM
#*      -> Carregar a imagem da cor
#*      -> Converter RGB para LAB (ou ja carregar em LAB - ver se é possivel)
#*      -> Salvar SRM como chave + Array LAB como valor
#* 3 - Mapear os estilos e valor SRM MIN/MAX da planilha
#* 4 - Criar estrutura de comparação:
#*      -> Estilo da cerveja    
#*      -> ARRAY SRM para valor MIN    
#*      -> ARRAY SRM para valor MAX
#*

import cv2
import os
import csv
import numpy as np

class SRMColor (object):
  def __init__(self, index):
    #Valor SRM da Cor
    self.index = index
    #A cor (uma tupla LAB carregada da imagem)
    self.colorValue = self.processSRMColor(index)

  def processSRMColor(self, index):
    filename = os.path.abspath(os.getcwd()) + "/SRM_BASE/SRM_{0}.PNG".format(index)
    if not os.path.isfile(filename):
      return None

    img = cv2.imread(filename)
    if img is None:
      return None

    #Pegamos o pixel do centro da imagem, e utilizamos a cor dele para definir a cor do SRM
    x,y = img.shape[:2]
    x = round (x/2)
    y = round (y/2)

    #Convertemos a imagem de RGB para LAB
    img = cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)

    #Retornamos a tupla com a cor do pixel central da imagem
    return img[x,y]

class BeerStyle(object):
  def __init__(self, name, min_srm, max_srm):
    self.name = name
    self.minSRM = SRMColor(round (float (min_srm)))
    self.maxSRM = SRMColor(round (float (max_srm)))


def loadBeersStyles():
  file =  open(os.path.abspath(os.getcwd()) + "/SRM_BASE/SRM Beers Values.csv")
  csvFile = csv.DictReader(file, delimiter=";")
  return csvFile
  

if __name__ == "__main__":

  base_file = loadBeersStyles()

  BeerStyles = [BeerStyle(line["Estilo"], line["MIN_SRM"], line["MAX_SRM"]) for line in base_file]

  [print (i.name, i.minSRM.index, i.minSRM.colorValue, i.maxSRM.index, i.maxSRM.colorValue) for i in BeerStyles]


