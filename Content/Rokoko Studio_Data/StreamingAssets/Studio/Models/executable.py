from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
import json
import numpy as np
import sys

timesteps = 10
def CreateModel():
    model = Sequential()
    model.add(LSTM(512, input_shape=(timesteps, 230)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
#     model.add(Dense(allOut.shape[1], activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

models = []
print(sys.argv)
for i in range(1, len(sys.argv)):
    print('initializing model', sys.argv[i])
    model = CreateModel()
    model.load_weights(sys.argv[i])
    models.append(model)

print("~ready~")
while True:
    manput = input()
    # manput = ''
    # with open('test_input.txt', 'r') as myfile:
    #     manput=myfile.read().replace('\n', '')
    jonput = json.loads(manput)
    npput = np.array(jonput)
    predictionResult = ''
    for model in models:
        modelPrediction = model.predict(npput)
        predictionResult = predictionResult + str(modelPrediction[0][0]) + ','
    if predictionResult.endswith(','):
        predictionResult = predictionResult[:-1]
    print(predictionResult)
    
