import os

os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import pickle as pk
import pandas as pd
import numpy as np
import random
import sys
sys.path.insert(1, '/home/aolivares/sslearn')
from sslearn.wrapper import CoTraining
from sslearn.wrapper import TriTraining
from sslearn.wrapper import DemocraticCoLearning
from sklearn.model_selection import StratifiedKFold, KFold
from models.MultiViewCoRegression import MultiviewCoReg

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, weightedtau
from scripts.weightedcorr import WeightedCorr
from scipy.stats import rankdata

from sklearn.model_selection import GridSearchCV
from scripts.MultiViewGridSearchCV import MultiViewGridSearchCV
from scripts.SSKFold import SSKFold, SSStratifiedKFold
from models.MERGE_v2 import Merge
from models.TriTrainingRegressor import TriTrainingRegressor
from datetime import datetime
from multiprocessing import Pool
import psutil
import warnings

from sklearn.linear_model import Ridge

def crossVal(dataset, general_model, random_state=1234, tune=False): 
    
    dataset_name, n_subs = dataset 
    
    scores_dict = dict()
    predictions_dict = dict()
    
    #prepare data 
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        
        Xl_dcae_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_dcae.pk', 'rb'))
        Xl_unirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_unirep.pk', 'rb'))
        Xl_eunirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_eunirep.pk', 'rb'))
        
        for n in n_subs: 
            
            Xl_dcae = np.empty((0, Xl_dcae_aux.shape[1]))
            Xl_unirep = np.empty((0, Xl_unirep_aux.shape[1]))
            Xl_eunirep = np.empty((0, Xl_eunirep_aux.shape[1]))
            y = np.empty(0) 
            
            #add substitutions to the specific case 
            for _n in range(1, n+1): 
                
                indexes = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_indexes.pk', 'rb'))
                Xl_dcae_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_dcae.pk', 'rb'))
                Xl_unirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_unirep.pk', 'rb'))[indexes]
                Xl_eunirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_eunirep.pk', 'rb'))[indexes]
                y_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_y_dcae.pk', 'rb')) 
                
                Xl_dcae = np.concatenate((Xl_dcae, Xl_dcae_aux), axis=0)
                Xl_unirep = np.concatenate((Xl_unirep, Xl_unirep_aux), axis=0)
                Xl_eunirep = np.concatenate((Xl_eunirep, Xl_eunirep_aux), axis=0)
                y = np.concatenate((y, y_aux))
            

            random.seed(random_state)
            n_splits = 10
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            args = []

            for i, (train_index, test_index) in enumerate(cv.split(Xl_dcae)): 
                args.append((i, train_index, test_index, dataset_name, general_model, n))

            with Pool(None) as pool: 
                results = pool.starmap(job, args, chunksize=1)

            predictions = [x[0] for x in results]
            scores = [x[1] for x in results]

            predictions_dict[str(n)+'_subs'] = predictions
            scores_dict[str(n)+'_subs'] = scores
            
    
    return predictions_dict, scores_dict

def job(i, train_index, test_index, dataset_name, general_model, n_subs):
    
    #read data
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore")
        wild_type = pk.load(open(f'datasets/{dataset_name}_wt_dcae.pk', 'rb'))
        
        Xl_dcae_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_dcae.pk', 'rb'))
        Xl_unirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_unirep.pk', 'rb'))
        Xl_eunirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_1_Xl_eunirep.pk', 'rb'))
        
        Xu_dcae = pk.load(open('datasets/'+dataset_name+'_Xu_dcae.pk', 'rb'))
            
        Xl_dcae = np.empty((0, Xl_dcae_aux.shape[1]))
        Xl_unirep = np.empty((0, Xl_unirep_aux.shape[1]))
        Xl_eunirep = np.empty((0, Xl_eunirep_aux.shape[1]))
        y = np.empty(0) 

        #add substitutions to the specific case 
        for _n in range(1, n_subs+1): 

            indexes = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_indexes.pk', 'rb'))
            Xl_dcae_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_dcae.pk', 'rb'))
            Xl_unirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_unirep.pk', 'rb'))[indexes]
            Xl_eunirep_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_Xl_eunirep.pk', 'rb'))[indexes]
            y_aux = pk.load(open('datasets/'+dataset_name+'_subs_'+str(_n)+'_y_dcae.pk', 'rb')) 

            Xl_dcae = np.concatenate((Xl_dcae, Xl_dcae_aux), axis=0)
            Xl_unirep = np.concatenate((Xl_unirep, Xl_unirep_aux), axis=0)
            Xl_eunirep = np.concatenate((Xl_eunirep, Xl_eunirep_aux), axis=0)
            y = np.concatenate((y, y_aux))
        
    #Dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()

    #split data
    Xl_dcae_train, Xl_dcae_test = Xl_dcae[train_index], Xl_dcae[test_index]
    Xl_unirep_train, Xl_unirep_test = Xl_unirep[train_index], Xl_unirep[test_index]
    Xl_eunirep_train, Xl_eunirep_test = Xl_eunirep[train_index], Xl_eunirep[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #save real y values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    
    w = rankdata([-y for y in y_test])
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    
    if 'TriTrainingRegressor' == general_model: 
        #add unlabeled instances to the training split
        Xl_dcae_train_tritr = np.concatenate((Xl_dcae_train, Xu_dcae))
        y_train_tritr = np.concatenate((y_train, np.full(Xu_dcae.shape[0], None)))

        #fit
        models = { 
            'rf': RandomForestRegressor(), 
            'ab': AdaBoostRegressor(), 
            'dt': DecisionTreeRegressor(), 
            'r': Ridge(), 
            'svm': SVR(), 
            'knn': KNeighborsRegressor(),
        }

        for key in models: 
            print(datetime.now(), '--> TriTrainingRegressor + '+key+' ( subs '+str(n_subs)+' dataset', dataset_name,')')
            model = models[key]

            tritr = TriTrainingRegressor(base_estimator=model)

            tritr.fit(Xl_dcae_train_tritr, y_train_tritr)
            prediction_tritr = tritr.predict(Xl_dcae_test)

            #scores
            predictions_dict['prediction_tritr_'+key+'_subs_'+str(n_subs)] = prediction_tritr
            scores_dict['mae_tritr_'+key+'_subs_'+str(n_subs)] = mean_absolute_error(y_test, prediction_tritr)
            scores_dict['mse_tritr_'+key+'_subs_'+str(n_subs)] = mean_squared_error(y_test, prediction_tritr)
            scores_dict['r2_tritr_'+key+'_subs_'+str(n_subs)] = r2_score(y_test, prediction_tritr)
            scores_dict['spearman_tritr_'+key+'_subs_'+str(n_subs)] = spearmanr(y_test, prediction_tritr)[0]
            scores_dict['wtau_tritr_'+key+'_subs_'+str(n_subs)] = weightedtau(y_test, prediction_tritr)[0]
            scores_dict['wspearman_tritr_'+key+'_subs_'+str(n_subs)]= WeightedCorr(x=pd.Series(y_test), 
                                                                                   y=pd.Series(prediction_tritr), 
                                                                                   w=pd.Series(w))(method='spearman')




    if 'MERGE' == general_model: 

        #fit
        models = { 
            'rf': RandomForestRegressor(),                          
            'ab': AdaBoostRegressor(), 
            'dt': DecisionTreeRegressor(), 
            'r': Ridge(),
            'svm': SVR(),
            'knn': KNeighborsRegressor(),
        }

        for key in models: 
            print(datetime.now(), '--> Merge ( subs '+str(n_subs)+' dataset', dataset_name,')')
            model = models[key]

            merge = Merge(wild_type=wild_type, base_regressor=model)

            merge.fit(Xl_dcae_train, y_train)
            prediction_merge = merge.predict(Xl_dcae_test)

            #scores
            predictions_dict['prediction_merge_'+key+'_subs_'+str(n_subs)] = prediction_merge

            scores_dict['spearman_merge_'+key+'_subs_'+str(n_subs)] = spearmanr(y_test, prediction_merge)[0]
            scores_dict['wtau_merge_'+key+'_subs_'+str(n_subs)] = weightedtau(y_test, prediction_merge)[0]
            scores_dict['wspearman_merge_'+key+'_subs_'+str(n_subs)] = WeightedCorr(x=pd.Series(y_test), 
                                                                            y=pd.Series(prediction_merge), 
                                                                            w=pd.Series(w))(method='spearman')



    if 'Unirep' == general_model: 

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
            print(datetime.now(), '--> Unirep + '+key+' ( subs '+str(n_subs)+' dataset', dataset_name,')')
            model = models[key]

            model.fit(Xl_unirep_train, y_train)
            prediction_unirep = model.predict(Xl_unirep_test)

            #scores
            predictions_dict['prediction_unirep_'+key+'_subs_'+str(n_subs)] = prediction_unirep

            scores_dict['spearman_unirep_'+key+'_subs_'+str(n_subs)] = spearmanr(y_test, prediction_unirep)[0]
            scores_dict['wtau_unirep_'+key+'_subs_'+str(n_subs)] = weightedtau(y_test, prediction_unirep)[0]
            scores_dict['wspearman_unirep_'+key+'_subs_'+str(n_subs)] = WeightedCorr(x=pd.Series(y_test), 
                                                                            y=pd.Series(prediction_unirep), 
                                                                            w=pd.Series(w))(method='spearman')


    if 'eUnirep' == general_model: 

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
            print(datetime.now(), '--> eUnirep + '+key+' ( subs '+str(n_subs)+' dataset', dataset_name,')')
            model = models[key]

            model.fit(Xl_eunirep_train, y_train)
            prediction_eunirep = model.predict(Xl_eunirep_test)

            #scores
            predictions_dict['prediction_eunirep_'+key+'_subs_'+str(n_subs)] = prediction_eunirep

            scores_dict['spearman_eunirep_'+key+'_subs_'+str(n_subs)] = spearmanr(y_test, prediction_eunirep)[0]
            scores_dict['wtau_eunirep_'+key+'_subs_'+str(n_subs)] = weightedtau(y_test, prediction_eunirep)[0]
            scores_dict['wspearman_eunirep_'+key+'_subs_'+str(n_subs)] = WeightedCorr(x=pd.Series(y_test), 
                                                                            y=pd.Series(prediction_eunirep), 
                                                                            w=pd.Series(w))(method='spearman')
    return predictions_dict, scores_dict
    
    
if __name__=="__main__": 
    
    datasets = [('avgfp', [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])]
    
    models = [
              'Unirep', 
              'eUnirep', 
              'TriTrainingRegressor', 
              'MERGE'
            ]

    for dataset in datasets: 
        for model in models: 
            print(datetime.now(), 'DATASET:', dataset)
            predictions, scores = crossVal(dataset, model, random_state=1234)
            with open('results/predictions_adding_substitutions_'+model+'_'+dataset[0]+'.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open('results/scores_adding_substitutions_'+model+'_'+dataset[0]+'.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)
