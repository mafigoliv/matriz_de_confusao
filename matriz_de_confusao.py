""" Cálculo de Métricas de Avaliação de Aprendizado """

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# Funções para calcular métricas
def sensibilidade(vp, fn):
    return vp / (vp + fn)

def especificidade(vn, fp):
    return vn / (fp + vn)

def acuracia(vp, vn, total):
    return (vp + vn) / total

def precisao(vp, fp):
    return vp / (vp + fp)

def f_score(precisao, sensibilidade):
    return 2 * (precisao * sensibilidade) / (precisao + sensibilidade)

# Exemplo de matriz de confusão arbitrária
y_true = [2, 0, 2, 2, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 1, 2, 2, 0, 0]
y_pred = [0, 0, 2, 2, 0, 0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 0, 1]

# Calculando a matriz de confusão
classes = [0, 1, 2]
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

# Visualizando a matriz de confusão
figure = plt.figure(figsize=(8, 8))
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Modelo de Rede Neural
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))

model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Treinamento do modelo (com dados fictícios de exemplo)
# x_train e y_train precisam ser definidos com os dados corretos
# model1.fit(x_train, y_train, epochs=10)

# Exemplo de valores para as métricas (valores arbitrários)
vp, vn, fp, fn = 50, 30, 10, 10
total = vp + vn + fp + fn

# Cálculo das métricas
sens = sensibilidade(vp, fn)
esp = especificidade(vn, fp)
acu = acuracia(vp, vn, total)
prec = precisao(vp, fp)
fscore = f_score(prec, sens)

print(f'Sensibilidade: {sens:.2f}')
print(f'Especificidade: {esp:.2f}')
print(f'Acurácia: {acu:.2f}')
print(f'Precisão: {prec:.2f}')
print(f'F-score: {fscore:.2f}')
