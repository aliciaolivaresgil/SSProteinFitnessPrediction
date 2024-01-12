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

#MODELS AND BASE ESTIMATORS
from models.TriTrainingRegressor import TriTrainingRegressor
from models.MERGE_v2 import Merge
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,  AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


#METRICS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, weightedtau
from scripts.weightedcorr import WeightedCorr
from scipy.stats import rankdata

#OTHER UTILS
from sklearn.model_selection import GridSearchCV
from scripts.MultiViewGridSearchCV import MultiViewGridSearchCV
from scripts.SSKFold import SSKFold, SSStratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold

def crossVal(dataset_name, general_model, random_state=1234): 
    
    #read data
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        
        Xl = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
        y = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        y_cat = np.where(y >= np.percentile(y, 75), 1, 0)

        
    random.seed(random_state)
    n_splits = 10
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    args = []
    
    for i, (train_index, test_index) in enumerate(cv.split(Xl, y)): 
        args.append((i, train_index, test_index, dataset_name, general_model))
        
    with Pool(None) as pool: 
        results = pool.starmap(job, args, chunksize=1)
        
    predictions = [x[0] for x in results]
    scores = [x[1] for x in results]
    
    return predictions, scores


def job(i, train_index, test_index, dataset_name, general_model): 
    
    #read data 
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        wild_type = pk.load(open(f'datasets/{dataset_name}_wt_dcae.pk', 'rb'))
        Xl = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
        Xu = pk.load(open(f'datasets/{dataset_name}_Xu_dcae.pk', 'rb'))
        y = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        
    #Dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()
    
    #split data 
    Xl_train, Xl_test = Xl[train_index], Xl[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #save real y values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    
    #calculate weights for Weighted Spearman's metric
    w = rankdata([-y for y in y_test])
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    
    if 'MERGE' == general_model: 
        
        #fit
        models = { 
            'rf': RandomForestRegressor(),  
            'ab': AdaBoostRegressor(), 
            'dt': DecisionTreeRegressor(), 
            'r': Ridge(), 
            'svm': SVR(), 
            'knn': KNeighborsRegressor()
        }
        
        for key in models: 
            print(datetime.now(), '--> Merge + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
            
            #fit 
            merge = Merge(wild_type=wild_type, base_regressor=model)
            merge.fit(Xl_train, y_train)
            prediction_merge = merge.predict(Xl_test)
            
            #scores
            predictions_dict['prediction_merge_'+key] = prediction_merge
            scores_dict['spearman_merge_'+key] = spearmanr(y_test, prediction_merge)[0]
            scores_dict['wtau_merge_'+key] = weightedtau(y_test, prediction_merge)[0]
            scores_dict['wspearman_merge:'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                               y=pd.Series(prediction_merge),
                                                               w=pd.Series(w))(method='spearman')           

    return predictions_dict, scores_dict
    


if __name__=="__main__": 
    
    datasets = ['bg_strsq', 'avgfp', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1', 'pabp_yeast_2',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    
    models = ['MERGE']
    
    for dataset in datasets: 
        for model in models: 
            print(datetime.now(), 'DATASET:', dataset)
            predictions, scores = crossVal(dataset, model)
            with open(f'results/predictions_comparison_{model}_{dataset}.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open(f'results/scores_comparison_{model}_{dataset}.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)
                