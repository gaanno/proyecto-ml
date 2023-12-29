# %%
from model_data import Model

#Modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import  Adam

# %%
model = Model()

# %%
rf_model = RandomForestClassifier(n_estimators=80, random_state=model.RANDOM_STATE,n_jobs=12)
rf_model.fit(model.X_train, model.y_train)
print(f'{rf_model.score(model.X_test, model.y_test)=}')
# %%
model.save_model(rf_model, 'rf_model')

# %%
dtc_model = DecisionTreeClassifier(random_state=42)
dtc_model.fit(model.X_train, model.y_train)
print(f'{dtc_model.score(model.X_test, model.y_test)=}')

# %%
model.save_model(dtc_model, 'dtc_model')

# %%
#modelo svm
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(model.X_train, model.y_train)
print(f'{svm_model.score(model.X_test, model.y_test)=}')
# %%
model.save_model(svm_model, 'svm_model')

# %%
#svm kernel lineal
svm_model_lineal = SVC(kernel='linear', random_state=42)
svm_model_lineal.fit(model.X_train, model.y_train)
print(f'{svm_model_lineal.score(model.X_test, model.y_test)=}')

# %%
model.save_model(svm_model_lineal, 'svm_model_lineal')

# %%
#svm kernel polinomial
svm_model_polinomial = SVC(kernel='poly', random_state=42)
svm_model_polinomial.fit(model.X_train, model.y_train)
print(f'{svm_model_polinomial.score(model.X_test, model.y_test)=}')

# %%
model.save_model(svm_model_polinomial, 'svm_model_polinomial')

# %%
model_block = Sequential()
model_block.add(InputLayer(input_shape=(model.X_train_t.shape[1],)))
model_block.add(Dense(256, activation='relu'))
model_block.add(Dropout(0.5))
model_block.add(Dense(64, activation='relu'))
model_block.add(Dropout(0.3))
model_block.add(Dense(16, activation='relu'))
model_block.add(Dropout(0.2))
model_block.add(Dense(model.y_train_t.shape[1], activation='softmax'))
model_block.compile(optimizer=Adam(learning_rate=model.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
model_block.summary()
model_fit = model_block.fit(model.X_train_t, model.y_train_t, epochs=model.NUM_EPOCH, validation_data=(model.X_test_t, model.y_test_t), batch_size=model.BATCH_SIZE)

# %%
#guarda el modelo
model.save_model_nn(model_block, 'model_block')
#guarda el historial de entrenamiento
model.save_model(model_fit, 'model_block_history')



