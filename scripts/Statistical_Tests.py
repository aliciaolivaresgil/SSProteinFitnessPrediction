import scipy.stats as stats
import scikit_posthocs as sp
import cv2

import csv
import sys
import baycomp as bc

import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

from scripts.baycomp_plotting import tern

def get_overall_data(models, base_estimators, datasets, metric, n=None): 
    """
    Reads scores and constructs the list needed to perform the bayesian test. 
    
    :param models: dictionary of models names
    :param base_estimators: list of base estimators names
    :param datasets: list of dataset names
    :param metric: name of metric
    :param n: number of instances used for training
    
    :return 1: list of model names (in order)
    :return 2: list with the scores needed to perform the bayesian test. 
    :return 3: list with the mean scores
    """
    
    path1 = f'../results/scores_comparison_'
    path2 = f'../results/scores_supervised_comparison_'
    overall_data = []
    overall_data_mean = []
    model_names = []
    
    #Add scores of semi-supervised models
    for key, estimators in models.items():
        for estimator in estimators:
            model_names.append(estimator)
            overall_aux = []
            overall_aux_mean = []
            for dataset in datasets: 
                if n==None: 
                    scores = pk.load(open(path1+f'{key}_{dataset}.pk', 'rb'))
                else: 
                    scores = pk.load(open(path1+f'{key}_{dataset}_{str(n)}_instances.pk', 'rb'))
                overall_aux_mean.append(np.mean([score[metric+'_'+estimator] for score in scores]))
                overall_aux.append([score[metric+'_'+estimator] for score in scores])
            overall_data_mean.append(overall_aux_mean)
            overall_data.append(overall_aux)
    
    #Add scores of supervised models 
    for be in base_estimators: 
        overall_aux = []
        overall_aux_mean = []
        model_names.append(be)
        for dataset in datasets: 
            if n==None: 
                scores = pk.load(open(path2+f'{dataset}.pk', 'rb'))
            else: 
                scores = pk.load(open(path2+f'{dataset}_{str(n)}_instances.pk', 'rb'))
            overall_aux_mean.append(np.mean([score[metric+'_'+be] for score in scores]))
            overall_aux.append([score[metric+'_'+be] for score in scores])
        overall_data_mean.append(overall_aux_mean)
        overall_data.append(overall_aux)
            
    return model_names, overall_data, overall_data_mean

def get_sota_data(models, best_model, base_estimators, datasets, metric, n=None): 
    
    """
    Reads scores and constructs the list needed to perform the bayesian test for the State-of-the-art methods. 
    
    :param models: dictionary of models names
    :param best_model: tuple with the name and key of the overall best_model from the previous experiments. 
    :param base_estimators: list of base estimators names
    :param datasets: list of dataset names
    :param metric: name of metric
    :param n: number of instances used for training
    
    :return 1: list of model names (in order)
    :return 2: list with the scores needed to perform the bayesian test. 
    :return 3: list with the mean scores
    """
    
    path1 = f'../results/scores_comparison_'
    path2 = f'../results/scores_comparison_sota_'
    overall_data = []
    overall_data_mean = []
    model_names = []
    
    #Add score of best model 
    key = best_model[0]
    estimator = best_model[1]
    model_names.append(estimator)
    overall_aux = []
    overall_aux_mean = []
    for dataset in datasets: 
        if n==None: 
            scores = pk.load(open(path1+f'{key}_{dataset}.pk', 'rb'))
        else: 
            scores = pk.load(open(path1+f'{key}_{dataset}_{str(n)}_instances.pk', 'rb'))
        overall_aux_mean.append(np.mean([score[metric+'_'+estimator] for score in scores]))
        overall_aux.append([score[metric+'_'+estimator] for score in scores])
    overall_data_mean.append(overall_aux_mean)
    overall_data.append(overall_aux)
    
    #Add scores of semi-supervised models
    for key, estimators in models.items():
        for estimator in estimators:
            model_names.append(estimator)
            overall_aux = []
            overall_aux_mean = []
            for dataset in datasets: 
                if n==None: 
                    scores = pk.load(open(path2+f'{key}_{dataset}.pk', 'rb'))
                else: 
                    scores = pk.load(open(path2+f'{key}_{dataset}_{str(n)}_instances.pk', 'rb'))
                overall_aux_mean.append(np.mean([score[metric+'_'+estimator] for score in scores]))
                overall_aux.append([score[metric+'_'+estimator] for score in scores])
            overall_data_mean.append(overall_aux_mean)
            overall_data.append(overall_aux)
            
    return model_names, overall_data, overall_data_mean


def bayesian(models, data, datasets, metric_name, rope=0.05, n=None):
    """
    Performs bayesian test and stores results. 
    
    :param models: list of model names
    :param data: list of scores calculated with get_overall_data function. 
    :param datasets: list of dataset names
    :param metric_name: metric name
    :param rope: ROPE param for bayesian test. 
    :param n: number of instances used for training
    """
    path = f'../results/bayesian_posteriors_rope='
    if len(datasets) == 2: 
        datasets = 'dm'
    else: 
        datasets = 's'
    
    for idx, model in enumerate(models): 
        for idx2, model2 in enumerate(models): 

            data1 = np.array(data[idx])
            data2 = np.array(data[idx2])
            names = [model, model2]
            
            if idx!=idx2: 
                warnings.filterwarnings("ignore")
                posterior = bc.HierarchicalTest(data1, data2, rope=rope)
                
                if n==None: 
                    filename = path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}.pk'
                    with open(filename, 'wb') as f: 
                        pk.dump(posterior, f)
                else: 
                    filename = path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}_{str(n)}_instances.pk'
                    with open(filename, 'wb') as f:
                        pk.dump(posterior, f)

                        
def generatePlots(models, datasets, metric_name, rope=0.05, n=None): 
    """
    Reads probabilities calculated by bayesian function, generates figs and saves them. 
    
    :param models: list of model names
    :param datasets: list of dataset names
    :param metric_name: metric name
    :param rope: ROPE param for bayesian test
    :param n: number of instances used for training
    """
    path = f'../results/bayesian_posteriors_rope='
    fig_path = f'../bayesian_figs/bayesian_rope='
    
    if len(datasets) == 2: 
        datasets = 'dm'
    else: 
        datasets = 's'
    
    for idx, model in enumerate(models): 
        for idx2, model2 in enumerate(models): 
            
            if idx != idx2: 
                if n==None: 
                    posterior = pk.load(open(path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}.pk', 'rb'))

                    fig = tern(posterior, l_tag=model, r_tag=model2)
                    plt.savefig(fig_path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}.png')
                else: 
                    posterior = pk.load(open(path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}_{str(n)}_instances.pk', 'rb'))
                    fig = tern(posterior, l_tag=model, r_tag=model2)
                    plt.savefig(fig_path+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}_{str(n)}_instances.png')
                    
                matplotlib.pyplot.close()

    
def generateBigPlot(models, datasets, metric_name, rope=0.05,  n=None):
    """
    Loads figs generated by generatePlots function, generates a big matrix figure and saves it. 
    
    :param models: list of model names
    :param datasets: list of dataset names
    :param metric_name: metric name
    :param rope: ROPE param for bayesian test
    :param n: number of instances used for training
    """
    fig_path1 = f'../bayesian_figs/bayesian_rope='
    fig_path2 = f'../bayesian_figs/bayesian_comparative_rope='
    
    if len(datasets) == 2: 
        datasets = 'dm'
    else: 
        datasets = 's'
    
    aux_img = cv2.imread(fig_path1+f'{str(rope)}_{models[0]}_{models[1]}_{metric_name}_{datasets}.png')
    aux_img[:] = (242, 242, 242)
    
    final = None 
    for idx, model in enumerate(models): 
        row = None 
        for idx2, model2 in enumerate(models): 
            
            if idx==idx2: 
                img = aux_img
            else: 
                if n==None: 
                    img = cv2.imread(fig_path1+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}.png')
                else: 
                    img = cv2.imread(fig_path1+f'{str(rope)}_{model}_{model2}_{metric_name}_{datasets}_{str(n)}_instances.png')
                
            if idx2==0: 
                row = img
            else: 
                row = cv2.hconcat([row, img])
                
        if idx == 0: 
            final = row
        else: 
            final = cv2.vconcat([final, row])
    if n==None:         
        cv2.imwrite(fig_path2+f'{str(rope)}_{metric_name}_{datasets}.png' ,final)
    else: 
        cv2.imwrite(fig_path2+f'{str(rope)}_{metric_name}_{datasets}_{str(n)}_instances.png', final)
            
    
    
if __name__=="__main__": 
    
    """
    This script performs the bayesian test for all pairs of methods defined in models dictionary and base estimators list. 
    Method's scores must be already stored in results directory. 
    """
    
    #***********************
    #***********************
    #1. EXPERIMENTAL RESULTS
    #***********************
    #***********************
    
    models = {'MERGE': ('merge_rf', 'merge_ab', 'merge_dt', 'merge_r', 'merge_svm', 'merge_knn'), 
              'CoTraining': ('cot_rf', 'cot_ab', 'cot_dt', 'cot_svm', 'cot_gnb', 'cot_knn'), 
              #'DemocraticCoLearning': ('dcol_rf+gnb+svc', 'dcol_ab+gnb+svc', 'dcol_dt+gnb+svc'), 
              'TriTraining': ('trit_rf', 'trit_ab', 'trit_dt', 'trit_svm', 'trit_gnb', 'trit_knn'),
              'TriTrainingRegressor': ('tritr_rf', 'tritr_ab', 'tritr_dt', 'tritr_r', 'tritr_svm', 'tritr_knn'),
              'CoRegression': ('cor',)}

    base_estimators = ['rfc', 'abc', 'dtc', 'svc', 'gnb', 'knnc', 'rfr', 'abr', 'dtr', 'r', 'svr', 'knnr']
    
    
    model_names = ['merge_rf', 'merge_ab', 'merge_dt', 'merge_r', 'merge_svm', 'merge_knn', 
               'cot_rf', 'cot_ab', 'cot_dt', 'cot_svm', 'cot_gnb', 'cot_knn', 
               'trit_rf', 'trit_ab', 'trit_dt', 'trit_svm', 'trit_gnb', 'trit_knn', 
               'tritr_rf', 'tritr_ab', 'tritr_dt', 'tritr_r', 'tritr_svm', 'tritr_knn', 
               'cor', 
               'rfc', 'abc', 'dtc', 'svc', 'gnb', 'knnc', 'rfr', 'abr', 'dtr', 'r', 'svr', 'knnr']
    
    #Single substitution datasets
    datasets1 = ['bg_strsq', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    #Multiple and double substitution datasets 
    datasets2 = ['avgfp', 'pabp_yeast_2']
    
    

    #**************************************
    #1.a. SINGLE SUBSTITUTIONS FULL DATASET
    #**************************************
    metrics = ['spearman', 'wspearman']
    for metric in metrics: 
        model_names, overall_data, overall_data_mean = get_overall_data(models, base_estimators, datasets1, metric)
        bayesian(model_names, overall_data, datasets1, metric, rope=0.01)
        generatePlots(model_names, datasets1, metric, rope=0.01)
        #generateBigPlot(model_names, datasets1, metric, rope=0.01)
    

    #***************************************************
    #1.b. DOUBLE AND MULTIPLE SUBSTITUTIONS FULL DATASET
    #***************************************************
    metrics = ['spearman', 'wspearman']
    for metric in metrics: 
        model_names, overall_data, overall_data_mean = get_overall_data(models, base_estimators, datasets2, metric)
        bayesian(model_names, overall_data, datasets2, metric, rope=0.01)
        generatePlots(model_names, datasets2, metric, rope=0.01)
        #generateBigPlot(model_names, datasets2, metric, rope=0.01)
    
    
    #***************************************
    #1.c. SINGLE SUBSTITUTIONS FEW INSTANCES
    #***************************************    
    for n in [250, 200, 150, 100, 50]: 
        for metric in ['spearman', 'wspearman']: 
            model_names, overall_data, overall_data_mean = get_overall_data(models, base_estimators, datasets1, metric, n=n)
            bayesian(model_names, overall_data, datasets1, metric, n=n, rope=0.01)
            generatePlots(model_names, datasets1, metric, n=n, rope=0.01)
            #generateBigPlot(model_names, datasets1, metric, n=n, rope=0.01)
            
    #****************************************************
    #1.d. DOUBLE AND MULTIPLE SUBSTITUTIONS FEW INSTANCES
    #****************************************************   
    for n in [250, 200, 150, 100, 50]: 
        for metric in ['spearman', 'wspearman']: 
            model_names, overall_data, overall_data_mean = get_overall_data(models, base_estimators, datasets2, metric, n=n)
            bayesian(model_names, overall_data, datasets2, metric, n=n, rope=0.01)
            generatePlots(model_names, datasets2, metric, n=n, rope=0.01)
            #generateBigPlot(model_names, datasets2, metric, n=n, rope=0.01)

    #***************
    #***************
    #2. SOTA RESULTS
    #***************
    #***************
    
    models = {
              'MERGE': ('merge_rf', 'merge_ab', 'merge_dt', 'merge_r', 'merge_svm', 'merge_knn'), 
              'Unirep': ('unirep_rf', 'unirep_ab', 'unirep_dt', 'unirep_r', 'unirep_svm', 'unirep_knn'), 
              'eUnirep': ('eunirep_rf', 'eunirep_ab', 'eunirep_dt', 'eunirep_r', 'eunirep_svm', 'eunirep_knn')
             }
    
    best_model = ('TriTrainingRegressor', 'tritr_svm')
    
    #**************************************
    #2.a. SINGLE SUBSTITUTIONS FULL DATASET
    #**************************************
    metrics = ['spearman', 'wspearman']
    for metric in metrics: 
        model_names, overall_data, overall_data_mean = get_sota_data(models, best_model, base_estimators, datasets1, metric)
        bayesian(model_names, overall_data, datasets1, metric, rope=0.01)
        generatePlots(model_names, datasets1, metric, rope=0.01)
        #generateBigPlot(model_names, datasets1, metric, rope=0.01)
    
    
    #***************************************************
    #2.b. DOUBLE AND MULTIPLE SUBSTITUTIONS FULL DATASET
    #***************************************************
    metrics = ['spearman', 'wspearman']
    for metric in metrics: 
        model_names, overall_data, overall_data_mean = get_sota_data(models, best_model, base_estimators, datasets2, metric)
        bayesian(model_names, overall_data, datasets2, metric, rope=0.01)
        generatePlots(model_names, datasets2, metric, rope=0.01)
        #generateBigPlot(model_names, datasets2, metric, rope=0.01)
    
    
    #***************************************
    #2.c. SINGLE SUBSTITUTIONS FEW INSTANCES
    #***************************************    
    for n in [250, 200, 150, 100, 50]: 
        for metric in ['spearman', 'wspearman']: 
            model_names, overall_data, overall_data_mean = get_sota_data(models, best_model, base_estimators, datasets1, metric, n=n)
            bayesian(model_names, overall_data, datasets1, metric, n=n, rope=0.01)
            generatePlots(model_names, datasets1, metric, n=n, rope=0.01)
            #generateBigPlot(model_names, datasets1, metric, n=n, rope=0.01)
            
    #****************************************************
    #2.d. DOUBLE AND MULTIPLE SUBSTITUTIONS FEW INSTANCES
    #****************************************************   
    for n in [250, 200, 150, 100, 50]: 
        for metric in ['spearman', 'wspearman']: 
            model_names, overall_data, overall_data_mean = get_sota_data(models, best_model, base_estimators, datasets2, metric, n=n)
            bayesian(model_names, overall_data, datasets2, metric, n=n, rope=0.01)
            generatePlots(model_names, datasets2, metric, n=n, rope=0.01)
            #generateBigPlot(model_names, datasets2, metric, n=n, rope=0.01)
            
            
    #Pystan saves all models in caché and fills the disc. Delete jsonlines.gz in /home/user/.cache/httpstan/...
        
      
