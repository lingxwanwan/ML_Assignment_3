from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import random
import numpy as np


def generate_complex_numbers():
    real1 = random.uniform(-10, 10)
    imag1 = random.uniform(-10, 10)
    real2 = random.uniform(-10, 10)
    imag2 = random.uniform(-10, 10)

    c1 = np.complex(real1, imag1)
    c2 = np.complex(real2, imag2)

    op = random.choice(['+', '-', 'x', '/'])

    if op == '+':
        res = c1 + c2
    elif op == '-':
        res = c1 - c2
    elif op == 'x':
        res = c1 * c2
    else:
        res = c1 / c2

    return (c1, c2, op, res)


data = []
for i in range(1000):
    c1, c2, op, res = generate_complex_numbers()
    data.append((c1.real, c1.imag, c2.real, c2.imag, op, res.real, res.imag))


train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

X_train = np.array([[d[0]/10.0, d[1]/10.0, d[2]/10.0, d[3]/10.0]
                   for d in train_data])
y_train = np.array([[d[5]/10.0, d[6]/10.0] for d in train_data])

X_test = np.array([[d[0]/10.0, d[1]/10.0, d[2]/10.0, d[3]/10.0]
                  for d in test_data])
y_test = np.array([[d[5]/10.0, d[6]/10.0] for d in test_data])


model = Sequential()
model.add(Dense(64, input_dim=4, activation='linear'))
model.add(Dense(8, activation='linear'))
model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=32)

y_pred = model.predict(X_test)

mae = np.mean(np.abs(y_pred - y_test))
print('MAE:', mae)

mse = (np.mean((y_pred - y_test)**2))
print('MSE:', mse)

# replace 5th and 6th column with predicted values in test_data

# for i in range(len(test_data)):
#     # test_data[i][5] = y_pred[i, 0]
#     # test_data[i][6] = y_pred[i, 1]