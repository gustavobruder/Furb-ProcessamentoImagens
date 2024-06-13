# Alunos:
# - Gustavo Baroni Bruder
# - Ana Carolina da Silva

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


dados = pd.read_csv('./mudas_pinus.csv')

dados_x = []
dados_y = []

for dado in dados.values:
    dados_x.append(dado[:3])
    dados_y.append(dado[3:])

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(dados_x, dados_y, test_size=0.15)

classificador = MLPClassifier()
classificador.fit(x_treinamento, y_treinamento)

previsoes = classificador.predict(x_teste)
acuracia = accuracy_score(y_teste, previsoes)
matriz_confusao = confusion_matrix(y_teste, previsoes)

print(f'Quantidade de dados de treinamento: {len(x_treinamento)}')
print(f'Quantidade de dados de teste: {len(x_teste)}')
print(f'Acurácia: {acuracia:.2%}')
print(f'Matriz de confusão:\n{matriz_confusao}')

plt.title('Dados de treinamento x teste')
plt.bar(['treinamento', 'teste'], [len(y_treinamento), len(y_teste)])
plt.show()

matriz_confusao_display = ConfusionMatrixDisplay(matriz_confusao, display_labels=classificador.classes_)
matriz_confusao_display.plot()
plt.title(f'Matriz de confusão\nAcurácia: {acuracia:.2%}')
plt.show()
