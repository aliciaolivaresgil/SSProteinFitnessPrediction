import os
os.environ["OMP_NUM_THREADS"] = "5" # export OMP_NUM_THREADS=1
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

#SEMISUPERVISED MODELS
sys.path.insert(1, '/home/aolivares/sslearn')
from sslearn.wrapper import CoTraining, TriTraining, DemocraticCoLearning
from models.MERGE_v2 import Merge
from models.TriTrainingRegressor import TriTrainingRegressor
from models.MultiViewCoRegression import MultiviewCoReg

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
from scripts.MultiViewGridSearchCV import MultiViewGridSearchCV
from scripts.SSKFold import SSKFold, SSStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold



def crossVal(dataset_name, general_model, encoding,  n, random_state=1234, tune=False): 
    

    #read data
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        Xl_dcae = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
        y_dcae = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        y_cat = np.where(y_dcae >= np.percentile(y_dcae, 75), 1, 0)
        indexes = [i for i in range(len(Xl_dcae))]
    
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
        args.append((i, train_index, test_index, dataset_name, general_model, encoding, tune))
        
    with Pool(None) as pool: 
        results = pool.starmap(job, args, chunksize=1)
        
    predictions = [x[0] for x in results]
    scores = [x[1] for x in results] 
    tuned_params = [x[2] for x in results]
    
    return predictions, scores, tuned_params

def job(i, train_index, test_index, dataset_name, general_model, encoding, tune): 
    
    #read data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        wild_type = pk.load(open(f'datasets/{dataset_name}_wt_dcae.pk', 'rb'))
        
        Xl = pk.load(open(f'datasets/{dataset_name}_Xl_{encoding}.pk', 'rb'))
        Xu = pk.load(open(f'datasets/{dataset_name}_Xu_{encoding}.pk', 'rb'))
        y = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        
        if encoding !='dcae': 
            indexes = pk.load(open(f'datasets/{dataset_name}_indexes.pk', 'rb'))
            Xl = Xl.reshape((Xl.shape[0], -1))[indexes]
            Xu = Xu.reshape((Xu.shape[0], -1))
        
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

           
    if 'TriTrainingRegressor' == general_model: 
        #add unlabeled instances to the training split
        Xl_train_tritr = np.concatenate((Xl_train, Xu))
        y_train_tritr = np.concatenate((y_train, np.full(Xu.shape[0], None)))
        
        models = { 
            'rf': RandomForestRegressor(), 
            'ab': AdaBoostRegressor(), 
            'dt': DecisionTreeRegressor(), 
            'r': Ridge(),
            'svm': SVR(), 
            'knn': KNeighborsRegressor()
        }
        
        for key in models: 
            print(datetime.now(), '--> TriTrainingRegressor + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
        
            tritr = TriTrainingRegressor(base_estimator=model)
            
            if tune: 
                grid = {'y_tol_per': [0.000001, 0.0001, 0.001, 0.01, 0.1]}
                cv = SSKFold(n_splits=5, shuffle=True)
                search = GridSearchCV(tritr, grid, cv=cv)
                result = search.fit(Xl_train_tritr, y_train_tritr)
                best_model = result.best_estimator_
                tuned_params['tritr_'+key] = result
                prediction_tritr = best_model.predict(Xl_test)
            else: 
                tritr.fit(Xl_train_tritr, y_train_tritr)
                prediction_tritr = tritr.predict(Xl_test)

            #scores
            predictions_dict['prediction_tritr_'+key] = prediction_tritr
            scores_dict['mae_tritr_'+key] = mean_absolute_error(y_test, prediction_tritr)
            scores_dict['mse_tritr_'+key] = mean_squared_error(y_test, prediction_tritr)
            scores_dict['r2_tritr_'+key] = r2_score(y_test, prediction_tritr)
            scores_dict['spearman_tritr_'+key] = spearmanr(y_test, prediction_tritr)[0]
            scores_dict['wtau_tritr_'+key] = weightedtau(y_test, prediction_tritr)[0]
            scores_dict['wspearman_tritr_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                               y=pd.Series(prediction_tritr), 
                                                               w=pd.Series(w))(method='spearman')

    
    if 'CoRegression' == general_model: 
        #fit
        print(datetime.now(), '--> CoRegression (split ', i,'dataset', dataset_name,')')
        Xl_train_cor = np.concatenate((Xl_train, Xu))
        y_train_cor = np.concatenate((y_train, np.full(Xu.shape[0], None)))
        
        cor = MultiviewCoReg(max_iters=100, pool_size=100)
        
        if tune:
            grid = [{'p1': [2], 'p2': [3, 4, 5]},
                    {'p1': [3], 'p2': [4, 5]}, 
                    {'p1': [4], 'p2': [5]}]
            cv = SSKFold(n_splits=5, shuffle=True)
            search = GridSearchCV(cor, grid, cv=cv, n_jobs=1)
            result = search.fit(Xl_train_cor, y_train_cor)
            best_model = result.best_estimator_
            tuned_params['cor'] = result
            prediction_cor = best_model.predict(Xl_test)
        else: 
            cor.fit(Xl_train_cor, y_train_cor)
            prediction_cor = cor.predict(Xl_test)
            
            
        #scores
        predictions_dict['prediction_cor'] = prediction_cor
        scores_dict['mae_cor'] = mean_absolute_error(y_test, prediction_cor)
        scores_dict['mse_cor'] = mean_squared_error(y_test, prediction_cor)
        scores_dict['r2_cor'] = r2_score(y_test, prediction_cor)
        scores_dict['spearman_cor'] = spearmanr(y_test, prediction_cor)[0]
        scores_dict['wtau_cor'] = weightedtau(y_test, prediction_cor)[0]
        scores_dict['wspearman_cor'] = WeightedCorr(x=pd.Series(y_test), 
                                                     y=pd.Series(prediction_cor), 
                                                     w=pd.Series(w))(method='spearman')
        
    return predictions_dict, scores_dict, tuned_params        


if __name__=="__main__": 
    
    datasets = ['bg_strsq', 'avgfp', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1', 'pabp_yeast_2',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']

    models = [ 'TriTrainingRegressor', 'CoRegression']
    tune = True                          
    encodings = ['dcae', 'pam250']
    n_instances = [50, 100, 150, 200, 250]
    
    if len(sys.argv) <2: #No model, encoding or dataset specified (all). 

        for model in models: 
            for encoding in encodings: 
                for dataset in datasets:                                                                                        
                    for n in n_instances: 
                        print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N:', n)
                        predictions, scores, tuned_params = crossVal(dataset, model, encoding, n, random_state=1234, tune=tune)
                        with open(f'results/predictions_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_predictions: 
                            pk.dump(predictions, file_predictions)
                        with open(f'results/scores_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_scores: 
                            pk.dump(scores, file_scores)
                        with open(f'results/tuned_params_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_tuned_params: 
                            pk.dump(tuned_params, file_tuned_params)
                            
    elif len(sys.argv) <3: #Model specified. No encoding or dataset specified (all). 
        
        model = sys.argv[1]
        
        for encoding in encodings: 
            for dataset in datasets:                                    
                for n in n_instances: 
                    print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N:', n)
                    predictions, scores, tuned_params = crossVal(dataset, model, encoding, n, random_state=1234, tune=tune)
                    with open(f'results/predictions_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_predictions: 
                        pk.dump(predictions, file_predictions)
                    with open(f'results/scores_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_scores: 
                        pk.dump(scores, file_scores)
                    with open(f'results/tuned_params_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_tuned_params: 
                        pk.dump(tuned_params, file_tuned_params)
                        
    elif len(sys.argv) <4: # Model and encoding specified. No dataset specified (all). 
        
        model = sys.argv[1]
        encoding = sys.argv[2]
        
        for dataset in datasets:                                    
            for n in n_instances: 
                print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N:', n)
                predictions, scores, tuned_params = crossVal(dataset, model, encoding, n, random_state=1234, tune=tune)
                with open(f'results/predictions_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_predictions: 
                    pk.dump(predictions, file_predictions)
                with open(f'results/scores_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_scores: 
                    pk.dump(scores, file_scores)
                with open(f'results/tuned_params_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_tuned_params: 
                    pk.dump(tuned_params, file_tuned_params)
                    
    elif len(sys.argv) == 4: #Model, encoding and dataset specified. 
        
        model = sys.argv[1]
        encoding = sys.argv[2]
        dataset = sys.argv[3]
        
        for n in n_instances: 
            print(datetime.now(), 'DATASET:', dataset, 'MODEL:', model, 'N:', n)
            predictions, scores, tuned_params = crossVal(dataset, model, encoding, n, random_state=1234, tune=tune)
            with open(f'results/predictions_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open(f'results/scores_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)
            with open(f'results/tuned_params_comparison_{model}_{dataset}_{encoding}_{str(n)}_instances.pk', 'wb') as file_tuned_params: 
                pk.dump(tuned_params, file_tuned_params)

        
        
        
