import os
os.environ["OMP_NUM_THREADS"] = "50" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import pickle as pk
import numpy as np
import pandas as pd
import random
import os
import sys
from datetime import datetime
from multiprocessing import Pool
import psutil
import warnings

#BASE ESTIMATORS
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge

#METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, weightedtau
from scripts.weightedcorr import WeightedCorr
from scipy.stats import rankdata

#OTHER UTILS
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, KFold



def crossVal(dataset_name, encoding, random_state=1234): 
    
    #read data
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        Xl_dcae = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
        y_dcae = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        y_cat = np.where(y_dcae >= np.percentile(y_dcae, 75), 1, 0)
        
    random.seed(random_state)
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    args = []
    
    for i, (train_index, test_index) in enumerate(cv.split(Xl_dcae, y_cat)): 
        args.append((i, train_index, test_index, dataset_name, encoding))
        
    with Pool(None) as pool: 
        results = pool.starmap(job, args, chunksize=1)
        
    predictions = [x[0] for x in results]
    scores = [x[1] for x in results]
    
    return predictions, scores


def job(i, train_index, test_index, dataset_name, encoding): 
    
    with warnings.catch_warnings(): 
        
        warnings.filterwarnings("ignore")
     
        Xl = pk.load(open(f'datasets/{dataset_name}_Xl_{encoding}.pk', 'rb'))
        y = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        
        if encoding != 'dcae': 
            indexes = pk.load(open(f'datasets/{dataset_name}_indexes.pk', 'rb'))
            Xl = Xl.reshape((Xl.shape[0], -1))[indexes]

    #Dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()
    tuned_params = dict()


    #split data 
    Xl_train, Xl_test = Xl[train_index], Xl[test_index]
    y_train, y_test = y[train_index], y[test_index]
 
    #save real y values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    
    #calculate weights for Weighted Spearman's metric
    w = rankdata([-y for y in y_test])
    w = (w-np.min(w))/(np.max(w)-np.min(w))

    #define base estimators 
    regressors = {
                  'rfr': RandomForestRegressor(),
                  'abr': AdaBoostRegressor(), 
                  'dtr': DecisionTreeRegressor(),
                  'r': Ridge(),
                  'svr': SVR(),
                  'knnr': KNeighborsRegressor()
    }
    
    for key, estimator in regressors.items(): 
        
        print(datetime.now(), '-->', key, '(split', i, 'dataset', dataset_name, ', encoding', encoding,')')
        
        #fit
        estimator.fit(Xl_train, y_train)
        prediction = estimator.predict(Xl_test)
        predictions_dict['prediction_'+key] = prediction
        
        #scores
        scores_dict['mae_'+key] = mean_absolute_error(y_test, prediction)
        scores_dict['mse_'+key] = mean_squared_error(y_test, prediction)
        scores_dict['r2_'+key] = r2_score(y_test, prediction)
        scores_dict['spearman_'+key] = spearmanr(y_test, prediction)[0]
        scores_dict['wtau_'+key] = weightedtau(y_test, prediction)[0]
        scores_dict['wspearman_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                     y=pd.Series(prediction), 
                                                     w=pd.Series(w))(method='spearman')
        
    return predictions_dict, scores_dict
if __name__=="__main__": 
    
    datasets = ['bg_strsq', 'avgfp', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1', 'pabp_yeast_2',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    
    encodings = ['dcae', 'pam250']
    encodings = ['unirep', 'eunirep']
    
    for encoding in encodings: 
        for dataset in datasets: 
            print(datetime.now(), 'DATASET:', dataset)
            predictions, scores = crossVal(dataset, encoding, random_state=1234)
            with open(f'results/predictions_supervised_comparison_{dataset}_{encoding}.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open(f'results/scores_supervised_comparison_{dataset}_{encoding}.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)

