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

if __name__ == "__main__":
  #SRMChart = "/home/luis/Faculdade/DS878_PDI/Trabalho/SRMChart.png"
  SRMChart = "/home/luis/Faculdade/DS878_PDI/tarefa_4/sample.bmp"
  img = cv2.imread(SRMChart)

  if img is None:
      sys.Exit ('Falha ao carregar a imagem.')

  cv2.imshow("Antes da conversao", img)
  cv2.waitKey(0)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

  cv2.imshow("Depois da conversao", img)
  cv2.waitKey(0)


