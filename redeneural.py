# Import TensorFlow and NumPy
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Exibir a versão do TensorFlow
print(f"Versão do TensorFlow: {tf.__version__}")

# Definir os dados de entrada (x) e os resultados esperados (y)
x_train = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y_train = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Criar o modelo de rede neural
model = Sequential([
    Dense(units=1, input_shape=[1])  # Uma camada com um neurônio
])

# Compilar o modelo
model.compile(optimizer='sgd',  # Otimizador Gradiente Descendente
              loss='mean_squared_error')  # Função de perda: erro quadrático médio

# Treinar o modelo
print("Treinando o modelo...")
history = model.fit(x_train, y_train, epochs=500, verbose=0)  # 500 épocas de treinamento
print("Treinamento concluído!")

# Testar o modelo com novos dados
print("Previsão para x = 10.0:")
x_test = np.array([[10.0]])  # Convertendo entrada para formato 2D
result = model.predict(x_test)
print(f"Resultado: {result}")

# Exibir os pesos aprendidos pelo modelo
weights, biases = model.layers[0].get_weights()
print(f"Pesos: {weights}, Biases: {biases}")




# SAIDA 
Versão do TensorFlow: 2.17.1
Treinando o modelo...
Treinamento concluído!
Previsão para x = 10.0:
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step
Resultado: [[18.987545]]
Pesos: [[1.9981949]], Biases: [-0.9944035]
