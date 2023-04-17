# general imports
import numpy as np
import pandas as pd
from platform import python_version
if python_version() < '3.8':
    import pickle5 as pickle
else:
    import pickle
from datetime import time
from tqdm import tqdm

# tsa
from statsmodels.tsa.arima.model import ARIMA

# deep learning
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# ensemble learning
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


######################################### CONSTANTS #############################################

N_TRAIN = 2208 # 23 days
N_DEV = 672 # 7 days
N_TEST = 288 # 3 days
M = 96 # periodicity
EARLIEST_START, LATEST_END = time(6, 0, 0), time(18, 30, 0) # earliest observed sunrise and latest observed sundown
BC = 1 # additive constant for box-cox transformation

######################################### GENERAL UTILITY FUNCTIONS #############################

def save_obj(obj, path, name):

    """ pickles and saves an object to a specified location """

    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path, name):

    """ loads a pickled object from a specified location """

    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class PredictionResult:

    """ Class to store the results of a prediction model on development and test sets along with the corresponding target values and error metrics"""
    
    def __init__(self, model_name, y_dev, preds_dev, y_test, preds_test):
        
        self.y_dev = y_dev
        self.preds_dev = preds_dev
        self.y_test = y_test
        self.preds_test = preds_test
        self.model_name = model_name
        
    def compute_errors(self):
        
        self.rmse_dev = np.sqrt( np.mean( (self.y_dev-self.preds_dev)**2 ) )
        self.mape_dev = np.mean( abs( (self.y_dev[self.y_dev>0]-self.preds_dev[self.y_dev>0])/self.y_dev[self.y_dev>0] ) ) * 100
        self.me_dev = np.mean( self.y_dev-self.preds_dev )
        
        self.rmse_test = np.sqrt( np.mean( (self.y_test-self.preds_test)**2 ) )
        self.mape_test = np.mean( abs( (self.y_test[self.y_test>0]-self.preds_test[self.y_test>0])/self.y_test[self.y_test>0] ) ) * 100
        self.me_test = np.mean( self.y_test-self.preds_test )
        
        self.errors = [self.rmse_dev, self.mape_dev, self.me_dev, self.rmse_test, self.mape_test, self.me_test]


def print_plot_results(res: PredictionResult, ax, mode: str='dev'):

    """ Given a PredictionResult instance, plot target values vs. predictions and print errors (either for dev or test set)"""

    if mode == 'dev':

        errors = res.errors[:3]
        actuals = res.y_dev
        preds = res.preds_dev

    else:

        actuals = res.y_test
        preds = res.preds_test
        errors = res.errors[-3:]

    x = range(len(actuals))
    ax.plot(x, actuals, label='actuals')
    ax.plot(x, preds, label='preds')
    ax.legend()
    ax.set_title('Model:  {}. RMSE: {:.0f}. MAPE: {:.2f}%. ME: {:.0f}.'.format(res.model_name, *errors))



######################################### TSA UTILITY FUNCTIONS #############################

def transf(y, lam=.011152871280864698, const=BC):
    """ boxcox transfer function with MLE-optimal lambda as default; constant is used to ensure the data is strictly positive"""
    return ((y+const)**lam-1)/lam

def inv_transf(y, lam=.011152871280864698, const=BC):
    """ inverse boxcox transfer function with MLE-optimal lambda as default"""
    return (lam*y+1)**(1/lam)-const


def eval_sarima(order, seasonal_order, n_train=3*96, n_eval=96, n_cvs=4, verbose=True, mode='search',
                transform=transf, inv_transform=inv_transf, exog=np.array([])):
    
    """ 
    Evaluate a SARIMAX model in a rolling horizon manner.

    order: ARIMA order
    seasonal_order: SARIMA order
    n_train: number of data instances used for training
    n_eval: prediction horizon
    n_cvs: number of cross validation splits
    verbose: if True, print progess
    mode: if mode=='search' (res. 'test'), training and prediction intervalls are chosen such that the last prediction is made for the last n_eval instances of the developmen (res. test) set
    transform: transfomation function to apply to the target
    inv_transf: inverse transformation function to apply to predictions
    exog: explanatory variables
    """
    
    agg_df = pd.read_csv(r'../Data Sets/agg_df.csv')
    
    if mode=='search':
        start_train = N_TRAIN+N_DEV-n_cvs*n_eval-n_train
    else:
        start_train = N_TRAIN+N_DEV+N_TEST-n_cvs*n_eval-n_train
    
    if start_train < 0:
        raise ValueError('Training and development/test set too small. Reduce n_train and/or n_eval.')
    
    if verbose:
        print('Completed CV Splits (of {}): '.format(n_cvs), end='')
        
    rmse = 0
    preds_stacked = np.array([])
    for n_cv in range(1, n_cvs+1):
        y_train = agg_df['dc'].values[start_train:start_train+n_train]
        if transform is not None:
            y_train = transform(y_train)
        if exog.any():
            X_train = exog[start_train:start_train+n_train]
            X_dev = exog[start_train+n_train:start_train+n_train+n_eval]
        else:
            X_train, X_dev = None, None
        y_dev = agg_df['dc'].values[start_train+n_train:start_train+n_train+n_eval]
        model = ARIMA(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
        model = model.fit()
        preds = model.forecast(steps=n_eval, exog=X_dev)
        if transform is not None:
            preds = inv_transform(preds)
        preds_stacked = np.append(preds_stacked, preds)
        rmse += np.sqrt( np.mean( (preds-y_dev)**2 ) ) / n_cvs
        start_train += n_eval 
        
        if verbose:
            print(n_cv, end=',')
            
    return rmse, preds_stacked

######################################### DEEP LEARNING UTILITY FUNCTIONS #############################

def build_net(hidden_nodes, input_shape, seed, model_type):

    """ build a neural network with the specified architecture (model_type must be one of 'MLP', 'LSTM') """
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    model = Sequential()
    
    if model_type=='MLP':
        model.add(Dense(hidden_nodes[0], input_shape=input_shape, activation='relu'))
        for n_nodes in hidden_nodes[1:]:
            model.add(Dense(n_nodes, activation='relu'))
    elif model_type=='LSTM':
        return_sequences = len(hidden_nodes)>1
        model.add(LSTM(hidden_nodes[0], input_shape=input_shape, return_sequences=return_sequences))
        if return_sequences:
            for n_nodes in hidden_nodes[1:-1]:
                model.add(LSTM(n_nodes, return_sequences=True))
            model.add(LSTM(hidden_nodes[-1], return_sequences=False))
        
    model.add(Dense(1))
    
    return model   

def compile_net(model, lr, loss, optimizer, seed):

    """ compile a built neural network, i.e. assign a learning rate and loss function and initialize weights"""

    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr) if optimizer=='adam' else tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer)
    
    return model

def eval_net(lr, hidden_nodes, max_epochs, patience, batch_size, model_type, X, y,
             seed=42, loss='mse', optimizer='adam', timesteps=48,
             n_dev=3*M, n_cvs=7, n_eval=M, mode='search',
             verbose=True,  keras_verbose=0,
             man_adjust=True, scaler=None):
    
    """ evaluate a neural network in a rolling horizon manner
     
    lr: learning rate
    hidden_nodes: list of ints, specifying the number of hidden nodes per layer (and the number of hidden layers)
    max_epochs: maximum number of epochs per training run
    patience: maximum number of epochs without improvement per training run
    bath_size: batch size
    model_type: one of 'MLP', 'LSTM'
    X: feature matrix
    y: target variable
    seed: random seed to enusre reproducability
    loss: network loss function
    optimizer: gradient descent algorithm (one of 'adam', 'sgd')
    timesteps: only relevant if model_tyype=='LSTM'; number of timesteps per input instance
    n_dev: number of instances to use as development set to test for patience termination criterion
    n_cvs: number of cross validation splits
    n_eval: number of timesteps per prediction / cross-validation
    mode: if mode=='search' (res. 'test'), training and prediction intervalls are chosen such that the last prediction is made for the last n_eval instances of the developmen (res. test) set
    verbose: if True, print cross-validation progress
    keras_verbose: if True, print training progress
    man_adjust: if True, manually set nighttime predictions to 0
    scaler: scaler the inverse transform of which to use to recover the original order of magnitude of a model's predictions
    """
    
    
    input_shape = ((X.shape[1],) if model_type=='MLP' else (X.shape[1], X.shape[2]))
    model = build_net(hidden_nodes, input_shape, seed, model_type)
    model = compile_net(model, lr, loss, optimizer, seed)

    n_train = N_TRAIN if model_type=='MLP' else N_TRAIN-timesteps+1
    if mode=='search':
        end_train = n_train+N_DEV-n_cvs*n_eval-n_dev
    elif mode=='test':
        end_train = n_train+N_DEV+N_TEST-n_cvs*n_eval-n_dev
    y_target = y[end_train+n_dev:end_train+n_dev+n_cvs*n_eval]
    

    if verbose:
        print('Completed CV Splits (of {}): '.format(n_cvs), end='')
    preds = np.array([])

    for n_cv in range(1, n_cvs+1):

        X_train, y_train = X[:end_train], y[:end_train]
        X_dev_d, y_dev_d = X[end_train:end_train+n_dev], y[end_train:end_train+n_dev]
        
               
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)                    
        _ = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size,
                      validation_data=(X_dev_d, y_dev_d), verbose=keras_verbose, shuffle=False, callbacks=[callback])
        
        preds_d = model(X[end_train+n_dev:end_train+n_dev+n_eval]).numpy().flatten()
        if model_type=='MLP':
            preds_d = scaler.inverse_transform(np.append(preds_d.reshape(-1,1), X[:n_eval], axis=1))[:, 0]
        elif model_type=='LSTM':
            preds_d = scaler.inverse_transform(np.append(preds_d.reshape(-1,1), X[:n_eval, 0, 1:], axis=1))[:, 0]
        if man_adjust:
            earliest = EARLIEST_START.hour*4 + EARLIEST_START.minute//15
            latest = LATEST_END.hour*4 + LATEST_END.minute//15 + 1
            preds_d[:earliest] = 0
            preds_d[latest:] = 0
        preds = np.append(preds, preds_d)

        if verbose:
            print(n_cv, end=',')

        end_train += n_eval

    if model_type == "MLP":
        y_target = scaler.inverse_transform(np.append(y_target.reshape(-1,1), X[:n_cvs*n_eval], axis=1))[:, 0]
    elif model_type == "LSTM":
        y_target = scaler.inverse_transform(np.append(y_target.reshape(-1,1), X[:n_cvs*n_eval, 0, 1:], axis=1))[:, 0]
    rmse = np.sqrt( np.mean( (y_target-preds)**2 ) )

    return rmse, y_target, preds, model


######################################### ENSEMBLE LEARNING UTILITY FUNCTIONS #############################


def pipeline(model, params, preprocessor, n_jobs = None, cv = 3, verbose=True):

    """ assemble a grid_search_cv instance given a preprocessor, a model and a set of (hyper-)parameters to evaluate in a cross-validated way """

    np.random.seed(42)
    pipeline = Pipeline([('preprocessor', preprocessor),
                         ('regressor', model)], verbose = verbose)
    grid_search_cv = GridSearchCV(pipeline,
                                  params,
                                  n_jobs =n_jobs,
                                  cv=cv)
    return grid_search_cv


def eval_pipeline(pipeline, X_train, X_dev, X_test, y_train, y_dev, y_test, model_name, man_adjust=True):

    """ given a pipeline (better: a GridSearchCV) instance, fit it on the training data and evaluate on development and test sets;
    by default, nighttime predictions are manually adjusted to 0 """

    pipeline_trained = pipeline.fit(X_train,y_train)
    preds_dev = pipeline_trained.predict(X_dev)
    preds_test = pipeline_trained.predict(X_test)

    if man_adjust:
        
        earliest = EARLIEST_START.hour*4 + EARLIEST_START.minute//15
        latest = LATEST_END.hour*4 + LATEST_END.minute//15 + 1
        for d in range(N_DEV//M):    
            preds_dev[d*M:d*M+earliest] = 0
            preds_dev[d*M+latest:(d+1)*M] = 0
        for d in range(N_TEST//M):    
            preds_test[d*M:d*M+earliest] = 0
            preds_test[d*M+latest:(d+1)*M] = 0

    result = PredictionResult(model_name, y_dev, preds_dev, y_test, preds_test)
    result.compute_errors()

    return result

