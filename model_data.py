import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models

class Model(object):
    _MODEL_FOLDER : str
    LEARNING_RATE : float
    NUM_EPOCH : int
    BATCH_SIZE : int
    RANDOM_STATE : int
    
    genres : dict
    
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : pd.DataFrame
    y_test : pd.DataFrame
    
    X_train_t : np.ndarray
    X_test_t : np.ndarray
    y_train_t : np.ndarray
    y_test_t : np.ndarray
    
    def __init__(self):
        self._MODEL_FOLDER = 'models'
        self.RANDOM_STATE = 1
        self.NUM_EPOCH = 10
        self.LEARNING_RATE = 5e-3
        self.BATCH_SIZE = 32
        
        self._verify_path()
        self._load_data()
        self._load_genres()

    def _verify_path(self):
        if not os.path.exists(self._MODEL_FOLDER):
            os.makedirs(self._MODEL_FOLDER)

    def _load_data(self):
        scaler = StandardScaler()
        
        df = pd.read_csv('generos.csv')
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.RANDOM_STATE, stratify=y)
    
        self.X_train_t = scaler.fit_transform(self.X_train)
        self.X_test_t = scaler.transform(self.X_test)
        self.y_train_t = to_categorical(self.y_train)
        self.y_test_t = to_categorical(self.y_test)
        
    def _load_genres(self):
        self.genres = {}
        sub = pd.read_csv('submission.csv')
        for columna in sub.columns:
            split = columna.rsplit('_', 1)
            self.genres[int(split[1])] = split[0]
            
    def _get_path(self, name):
        return os.path.join(self._MODEL_FOLDER, name)
    
    def save_model(self, model, name):
        pickle.dump(model, open(self._get_path(name), 'wb'))

    def save_model_nn(self, model, name):
        model.save(self._get_path(name))
        
    def load_model_nn(self, name):
        path = self._get_path(name)
        return models.load_model(path)
        
    def load_model(self, name):
        return pickle.load(open(self._get_path(name), 'rb'))