import os
os.environ["OMP_NUM_THREADS"] = "50" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import pickle as pk
import pandas as pd
import random
import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold


def crossVal(dataset_name, general_model, n, random_state=1234): 
    
    #read data
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        y = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        y_cat = np.where(y >= np.percentile(y, 75), 1, 0)
        indexes = [i for i in range(len(y))]
        
    train_indexes, test_indexes = [], []
    random.seed(random_state)
    for i in range(10): 
        seed = random.randint(1, 9999)
        test_size = 1-(n/len(y_cat))
        train_index, test_index, _, _ = train_test_split(indexes, 
                                                         y_cat, 
                                                         test_size=test_size, 
                                                         random_state=seed, 
                                                         stratify=y_cat)
        train_indexes.append(train_index)
        test_indexes.append(test_index)
        
    args = []
    
    for i, (train_index, test_index) in enumerate(zip(train_indexes, test_indexes)): 
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
        if general_model == 'MERGE': 
            wild_type = pk.load(open(f'datasets/{dataset_name}_wt_dcae.pk', 'rb'))
            Xl = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
            Xu = pk.load(open(f'datasets/{dataset_name}_Xu_dcae.pk', 'rb'))
        elif general_model == 'Unirep': 
            Xl = pk.load(open(f'datasets/{dataset_name}_Xl_unirep.pk', 'rb'))
        elif general_model == 'eUnirep': 
            Xl = pk.load(open(f'datasets/{dataset_name}_Xl_eunirep.pk', 'rb'))
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
            
    elif 'Unirep' == general_model: 
        
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
            print(datetime.now(), '--> Unirep + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
            
            #fit
            model.fit(Xl_train, y_train)
            prediction_unirep = model.predict(Xl_test)
            
            #scores 
            predictions_dict['prediction_unirep_'+key] = prediction_unirep
            scores_dict['mae_unirep_'+key] = mean_absolute_error(y_test, prediction_unirep)
            scores_dict['mse_unirep_'+key] = mean_squared_error(y_test, prediction_unirep)
            scores_dict['r2_unirep_'+key] = r2_score(y_test, prediction_unirep)
            scores_dict['spearman_unirep_'+key] = spearmanr(y_test, prediction_unirep)[0]
            scores_dict['wtau_unirep_'+key] = weightedtau(y_test, prediction_unirep)[0]
            scores_dict['wspearman_unirep_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                                y=pd.Series(prediction_unirep),
                                                                w=pd.Series(w))(method='spearman')    
    elif 'eUnirep' == general_model: 
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
            print(datetime.now(), '--> eUnirep + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
            
            #fit
            model.fit(Xl_train, y_train)
            prediction_eunirep = model.predict(Xl_test)
            
            #scores 
            predictions_dict['prediction_eunirep_'+key] = prediction_eunirep
            scores_dict['mae_eunirep_'+key] = mean_absolute_error(y_test, prediction_eunirep)
            scores_dict['mse_eunirep_'+key] = mean_squared_error(y_test, prediction_eunirep)
            scores_dict['r2_eunirep_'+key] = r2_score(y_test, prediction_eunirep)
            scores_dict['spearman_eunirep_'+key] = spearmanr(y_test, prediction_eunirep)[0]
            scores_dict['wtau_eunirep_'+key] = weightedtau(y_test, prediction_eunirep)[0]
            scores_dict['wspearman_eunirep_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                                y=pd.Series(prediction_eunirep),
                                                                w=pd.Series(w))(method='spearman')
    
    return predictions_dict, scores_dict


if __name__=="__main__": 
    
    datasets = ['bg_strsq', 'avgfp', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1', 'pabp_yeast_2',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    
    models = [
                #'MERGE', 
                #'Unirep', 
                'eUnirep'
             ]
    
    n_instances = [50, 100, 150, 200, 250]
    
    for dataset in datasets: 
        for model in models: 
            for n in n_instances: 
                print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N:', n)
                predictions, scores = crossVal(dataset, model, n, random_state=1234)
                with open(f'results/predictions_comparison_sota_{model}_{dataset}_{str(n)}_instances.pk', 'wb') as file_predictions: 
                    pk.dump(predictions, file_predictions)
                with open(f'results/scores_comparison_sota_{model}_{dataset}_{str(n)}_instances.pk', 'wb') as file_scores: 
                    pk.dump(scores, file_scores)
    