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
from sklearn.model_selection import StratifiedKFold, KFold





def crossVal(dataset_name, general_model, random_state=1234, tune=False): 
    
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
        args.append((i, train_index, test_index, dataset_name, general_model, tune))
          
    with Pool(None) as pool: 
        results = pool.starmap(job, args, chunksize=1)
        
    predictions = [x[0] for x in results]
    scores = [x[1] for x in results] 
    tuned_params = [x[2] for x in results]
    
    return predictions, scores, tuned_params

def job(i, train_index, test_index, dataset_name, general_model, tune): 
    
    #read data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        wild_type = pk.load(open(f'datasets/{dataset_name}_wt_dcae.pk', 'rb'))
        Xl_dcae = pk.load(open(f'datasets/{dataset_name}_Xl_dcae.pk', 'rb'))
        Xu_dcae = pk.load(open(f'datasets/{dataset_name}_Xu_dcae.pk', 'rb'))
        y_dcae = pk.load(open(f'datasets/{dataset_name}_y_dcae.pk', 'rb'))
        indexes = pk.load(open(f'datasets/{dataset_name}_indexes.pk', 'rb'))
        Xl_ohe = pk.load(open(f'datasets/{dataset_name}_Xl_ohe.pk', 'rb'))
        Xu_ohe = pk.load(open(f'datasets/{dataset_name}_Xu_ohe.pk', 'rb'))
        Xl_reshaped = Xl_ohe.reshape((Xl_ohe.shape[0], -1))[indexes]
        Xu_reshaped = Xu_ohe.reshape((Xu_ohe.shape[0], -1))
        y = pk.load(open(f'datasets/{dataset_name}_y.pk', 'rb'))

    #Dictionaries to save results
    scores_dict = dict()
    predictions_dict = dict()
    tuned_params = dict()

    #categorize y
    y_cat = np.where(y_dcae >= np.percentile(y_dcae, 75), 1, 0)

    #split data 
    Xl_dcae_train, Xl_dcae_test = Xl_dcae[train_index], Xl_dcae[test_index]
    Xl_ohe_train, Xl_ohe_test = Xl_reshaped[train_index], Xl_reshaped[test_index]
    y_train, y_test = y_dcae[train_index], y_dcae[test_index]
    y_cat_train, y_cat_test = y_cat[train_index], y_cat[test_index]

    #save real y values (to potentially calculate new metrics)
    predictions_dict['y_test'] = y_test
    predictions_dict['y_cat_test'] = y_cat_test
    
    #calculate weights for Weighted Spearman's metric
    w = rankdata([-y for y in y_test])
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    
    
    if 'CoTraining' == general_model: 
        #add unlabeled instances to the training split
        X_dcae_train_cot = np.concatenate((Xl_dcae_train, Xu_dcae))
        X_ohe_train_cot = np.concatenate((Xl_ohe_train, Xu_reshaped))
        y_cat_train_cot = np.concatenate((y_cat_train, np.ones(Xu_dcae.shape[0])*(-1)))

        models = { 
            'rf': RandomForestClassifier(),
            'ab': AdaBoostClassifier(),
            'dt': DecisionTreeClassifier(), 
            'svm': SVC(probability=True),
            'gnb': GaussianNB(), 
            'knn': KNeighborsClassifier()
        }

        
        for key in models:
            print(datetime.now(), '--> CoTraining + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
            
            #fit 
            cot = CoTraining(base_estimator=model)
            cot.fit(X_dcae_train_cot, y_cat_train_cot, X2=X_ohe_train_cot)
            prediction_cot = cot.predict_proba(Xl_dcae_test, X2=Xl_ohe_test)

            #scores
            predictions_dict['prediction_cot_'+key] = prediction_cot
            scores_dict['spearman_cot_'+key] = spearmanr(y_test, prediction_cot[:,1])[0]
            scores_dict['wtau_cot_'+key] = weightedtau(y_test, prediction_cot[:,1])[0]
            scores_dict['wspearman_cot_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                             y=pd.Series(prediction_cot[:,1]), 
                                                             w=pd.Series(w))(method='spearman')


    if 'TriTraining' == general_model: 
        #add unlabeled instances to the training split
        X_dcae_train_trit = np.concatenate((Xl_dcae_train, Xu_dcae))
        y_cat_train_trit = np.concatenate((y_cat_train, np.ones(Xu_dcae.shape[0])*(-1)))

        models = { 
            'rf': RandomForestClassifier(), 
            'ab': AdaBoostClassifier(), 
            'dt': DecisionTreeClassifier(),
            'svm': SVC(probability=True), 
            'gnb': GaussianNB(),
            'knn': KNeighborsClassifier()
        }
        
        for key in models: 
            print(datetime.now(), '--> TriTraining + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
            
            #fit
            trit = TriTraining(base_estimator=model)
            trit.fit(X_dcae_train_trit, y_cat_train_trit)
            prediction_trit = trit.predict_proba(Xl_dcae_test)

            #scores 
            predictions_dict['prediction_trit_'+key] = prediction_trit
            scores_dict['spearman_trit_'+key] = spearmanr(y_test, prediction_trit[:,1])[0]
            scores_dict['wtau_trit_'+key] = weightedtau(y_test, prediction_trit[:,1])[0]
            scores_dict['wspearman_trit_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                              y=pd.Series(prediction_trit[:,1]), 
                                                              w=pd.Series(w))(method='spearman')
            
        

            
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
            'knn': KNeighborsRegressor()
        }
        
        for key in models: 
            print(datetime.now(), '--> TriTrainingRegressor + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key]
        
            tritr = TriTrainingRegressor(base_estimator=model)
            
            if tune: 
                grid = {'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}
                cv = SSKFold(n_splits=5, shuffle=True)
                search = GridSearchCV(tritr, grid, cv=cv)
                result = search.fit(Xl_dcae_train_tritr, y_train_tritr)
                best_model = result.best_estimator_
                tuned_params['tritr_'+key] = result
                prediction_tritr = best_model.predict(Xl_dcae_test)
            else: 
                tritr.fit(Xl_dcae_train_tritr, y_train_tritr)
                prediction_tritr = tritr.predict(Xl_dcae_test)

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
        
        
    
    if 'DemocraticCoLearning' == general_model: 
        #add unlabeled instances to the training split  
        X_dcae_train_dcol = np.concatenate((Xl_dcae_train, Xu_dcae))
        y_cat_train_dcol = np.concatenate((y_cat_train, np.ones(Xu_dcae.shape[0])*(-1)))

        #fit
        models = { 
            'rf+gnb+svc': (RandomForestClassifier(), 
                           GaussianNB(), 
                           SVC()), 
            'ab+gnb+svc': (AdaBoostClassifier(), 
                           GaussianNB(), 
                           SVC()),
            'dt+gnb+svc': (DecisionTreeClassifier(), 
                           GaussianNB(), 
                           SVC())
        }
        
        for key in models: 
            print(datetime.now(), '--> DemocraticCoLearning + '+key+' (split ', i,'dataset', dataset_name,')')
            model1 = models[key][0]
            model2 = models[key][1]
            model3 = models[key][2]
        
            dcol = DemocraticCoLearning(base_estimator=[model1, model2, model3])
            dcol.fit(X_dcae_train_dcol, y_cat_train_dcol)
            prediction_dcol = dcol.predict_proba(Xl_dcae_test)

            #scores
            predictions_dict['prediction_dcol_'+key] = prediction_dcol
            scores_dict['spearman_dcol_'+key] = spearmanr(y_test, prediction_dcol[:,1])[0]
            scores_dict['wtau_dcol_'+key] = weightedtau(y_test, prediction_dcol[:,1])[0]
            scores_dict['wspearman_dcol_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                              y=pd.Series(prediction_dcol[:,1]), 
                                                              w=pd.Series(w))(method='spearman')
    
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
            merge.fit(Xl_dcae_train, y_train)
            prediction_merge = merge.predict(Xl_dcae_test)

            #scores
            predictions_dict['prediction_merge_'+key] = prediction_merge
            scores_dict['spearman_merge_'+key] = spearmanr(y_test, prediction_merge)[0]
            scores_dict['wtau_merge_'+key] = weightedtau(y_test, prediction_merge)[0]
            scores_dict['wspearman_merge_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                             y=pd.Series(prediction_merge), 
                                                             w=pd.Series(w))(method='spearman')

    
    if 'CoRegression' == general_model: 
        #fit
        print(datetime.now(), '--> CoRegression (split ', i,'dataset', dataset_name,')')
        X_train_coreg = np.concatenate((Xl_dcae_train, Xu_dcae))
        y_train_coreg = np.concatenate((y_train, np.full(Xu_dcae.shape[0], None)))

        cor = MultiviewCoReg(max_iters=100, pool_size=100)
        grid = [{'p1': [2], 'p2': [3, 4, 5]},
                {'p1': [3], 'p2': [4, 5]}, 
                {'p1': [4], 'p2': [5]}]
        cv = SSKFold(n_splits=5, shuffle=True)
        search = GridSearchCV(cor, grid, cv=cv)
        result = search.fit(X_train_coreg, y_train_coreg)
        best_model = result.best_estimator_
        tuned_params['cor'] = result

        #scores
        prediction_cor = best_model.predict(Xl_dcae_test)
        predictions_dict['prediction_cor'] = prediction_cor

        scores_dict['mae_cor'] = mean_absolute_error(y_test, prediction_cor)
        scores_dict['mse_cor'] = mean_squared_error(y_test, prediction_cor)
        scores_dict['r2_cor'] = r2_score(y_test, prediction_cor)
        scores_dict['spearman_cor'] = spearmanr(y_test, prediction_cor)[0]
        scores_dict['wtau_cor'] = weightedtau(y_test, prediction_cor)[0]
        scores_dict['wspearman_cor_'+key] = WeightedCorr(x=pd.Series(y_test), 
                                                         y=pd.Series(prediction_cor), 
                                                         w=pd.Series(w))(method='spearman')
    
    return predictions_dict, scores_dict, tuned_params
    

if __name__=="__main__": 
    
    datasets = ['bg_strsq', 'avgfp', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1', 'pabp_yeast_2',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    
    models = ['CoTraining', 'TriTraining', 'DemocraticCoLearning', 'MERGE', 'CoRegression', 'TriTrainingRegressor']
    tune = True
    

    for dataset in datasets: 
        for model in models: 
            print(datetime.now(), 'DATASET:', dataset)
            predictions, scores, tuned_params = crossVal(dataset, model, random_state=1234, tune=tune)
            with open(f'results/predictions_comparison_{model}_{dataset}.pk', 'wb') as file_predictions: 
                pk.dump(predictions, file_predictions)
            with open(f'results/scores_comparison_{model}_{dataset}.pk', 'wb') as file_scores: 
                pk.dump(scores, file_scores)
            with open(f'results/tuned_params_comparison_{model}_{dataset}.pk', 'wb') as file_tuned_params: 
                pk.dump(tuned_params, file_tuned_params)
        