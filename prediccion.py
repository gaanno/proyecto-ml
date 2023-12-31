# %%
from model_data import Model, pd, np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

# %%
model = Model()

# %% [markdown]
# CARGA LOS MODELOS ENTRENADOS

# %%
dtc_model = model.load_model('dtc_model')
rf_model = model.load_model('rf_model')
svm_model = model.load_model('svm_model')
svm_model_lineal = model.load_model('svm_model_lineal')
svm_model_polinomial = model.load_model('svm_model_polinomial')
model_block = model.load_model_nn('model_block')
model_block_history = model.load_model('model_block_history')

# %%
print(f'{dtc_model.score(model.X_test, model.y_test)=}')
print(f'{rf_model.score(model.X_test, model.y_test)=}')
print(f'{svm_model.score(model.X_test, model.y_test)=}')
print(f'{svm_model_lineal.score(model.X_test, model.y_test)=}')
print(f'{svm_model_polinomial.score(model.X_test, model.y_test)=}')
print(f'{model_block.evaluate(model.X_test, model.y_test_t)[1]=}')


# %%
def cm(modelo):
    y_test_labeled = model.y_test.map(model.genres)
    y_pred = modelo.predict(model.X_test)
    y_pred_labeled = pd.Series(y_pred).map(model.genres)
    cm = confusion_matrix(y_test_labeled, y_pred_labeled)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.genres.values())
    disp.plot()
    plt.title(modelo)
    plt.xticks(rotation=90)
    plt.show()

def cm_bloques(modelo):
    predictions = modelo.predict(model.X_test_t)
    predictions = np.argmax(predictions, axis=1)
    real_y = np.argmax(model.y_test_t, axis=1)
    cm = confusion_matrix(real_y, predictions,)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.genres.values())
    disp.plot()
    plt.title(f'Modelo {modelo.name}')
    plt.xticks(rotation=90)
    plt.show()

def graficos(model_fit):
    #grafico perdida vs epoch 
    plt.figure()
    plt.plot(model_fit.epoch, model_fit.history['loss'])
    plt.plot(model_fit.epoch, model_fit.history['val_loss'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Entrenamiento")
    plt.legend(['loss', 'val_loss'])
    plt.show()

    #Grafico accuracy vs val_accuracy
    plt.figure()
    plt.plot(model_fit.epoch, model_fit.history['accuracy'])
    plt.plot(model_fit.epoch, model_fit.history['val_accuracy'])
    plt.xlabel("Accuracy")
    plt.ylabel("Epoch")
    plt.title("Entrenamiento")
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

def mostrar_metricas(y_pred):
    y_test_labeled = model.y_test.map(model.genres)
    y_pred_labeled = pd.Series(y_pred).map(model.genres)

    classification_metrics_labeled = classification_report(y_test_labeled, y_pred_labeled, output_dict=True)
    classification_metrics_labeled_df = pd.DataFrame(classification_metrics_labeled).transpose()
    return classification_metrics_labeled_df


# %%
cm(rf_model)
mostrar_metricas(rf_model.predict(model.X_test))

# %%
cm(dtc_model)
mostrar_metricas(dtc_model.predict(model.X_test))

# %%
cm(svm_model)
mostrar_metricas(svm_model.predict(model.X_test))

# %%
cm(svm_model_lineal)
mostrar_metricas(svm_model_lineal.predict(model.X_test))

# %%
cm(svm_model_polinomial)
mostrar_metricas(svm_model_polinomial.predict(model.X_test))

# %%
cm_bloques(model_block)
graficos(model_block_history)


