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
import math

from baycomp_plotting import tern

def get_overall_data(models, datasets, metric, n=None): 
    """
    Reads scores and constructs the list needed to perform the bayesian test. 
    
    :param models: list of tuples (GENERAL_METHOD, ENCODING, MODEL)
    :param datasets: list of dataset names
    :param metric: name of metric
    :param n: number of instances used for training
    
    :return 1: list of model names (in order)
    :return 2: list with the scores needed to perform the bayesian test. 
    :return 3: list with the mean scores
    """
    
    abbreviations = {'rfr': 'RF', 
                     'abr': 'AB',
                     'dtr': 'DT', 
                     'r': 'Ridge', 
                     'svr': 'SVM', 
                     'knnr': 'K-NN', 
                     'tritr_rf': 'TriTR[RF]', 
                     'tritr_ab': 'TriTR[AB]', 
                     'tritr_dt': 'TriTR[DT]', 
                     'tritr_svm': 'TriTR[SVM]', 
                     'tritr_r': 'TriTR[Ridge]', 
                     'tritr_knn': 'TriTR[K-NN]', 
                     'cor': 'COREG', 
                     'merge_rf': 'MERGE[RF]', 
                     'merge_ab': 'MERGE[AB]', 
                     'merge_dt': 'MERGE[DT]', 
                     'merge_r': 'MERGE[Ridge]', 
                     'merge_svm': 'MERGE[SVM]', 
                     'merge_knn': 'MERGE[K-NN]'
                    }
    
    path1 = f'../results/scores_supervised_comparison_'
    path2 = f'../results/scores_comparison_'
    overall_data = []
    overall_data_mean = []
    model_names = []
    
    for general_method, encoding, model in models: 
        overall_aux = []
        overall_aux_mean = []
        for dataset in datasets: 
            #Strategies 0 and 1
            if general_method == None: 
                if n==None: 
                    scores = pk.load(open(path1+f'{dataset}_{encoding}.pk', 'rb'))
                else: 
                    scores = pk.load(open(path1+f'{dataset}_{n}_instances_{encoding}.pk', 'rb'))
            #Strategy 2
            elif general_method == 'MERGE': 
                if n==None: 
                    scores = pk.load(open(path2+f'MERGE_{dataset}.pk', 'rb'))
                else: 
                    scores = pk.load(open(path2+f'MERGE_{dataset}_{n}_instances.pk', 'rb'))
            #Strategies 3 and 4
            else: 
                if n==None: 
                    scores = pk.load(open(path2+f'{general_method}_{dataset}_{encoding}.pk', 'rb'))
                else: 
                    scores = pk.load(open(path2+f'{general_method}_{dataset}_{encoding}_{n}_instances.pk', 'rb'))
              
            
            aux = [0 if math.isnan(score[f'{metric}_{model}']) else abs(score[f'{metric}_{model}']) for score in scores]
                
            overall_aux_mean.append(np.mean(aux))
            overall_aux.append(aux)
        overall_data_mean.append(overall_aux_mean)
        overall_data.append(overall_aux)
        
        if encoding == None: 
            model_names.append(abbreviations[model])
        else: 
            model_names.append(f'{encoding}+{abbreviations[model]}')
            
    return model_names, overall_data, overall_data_mean


def bayesian(models, data, best_model_idx, datasets, metric_name, rope=0.05, n=None):
    """
    Performs bayesian test and stores results. 
    
    :param models: list of model names
    :param data: list of scores calculated with get_overall_data function. 
    :param best_model_idx: index of the best_model in 'models' and 'data'. 
    :param datasets: list of dataset names
    :param metric_name: metric name
    :param rope: ROPE param for bayesian test. 
    :param n: number of instances used for training
    """
    path = f'../results_baycomp/bayesian_posteriors_rope='
    
    best_model = models[best_model_idx]
    best_model_data = np.array(data[best_model_idx])
    
    for idx, model in enumerate(models): 

        model_data = np.array(data[idx])
        names = [best_model, model]

        if best_model_idx!=idx: 
            posterior = bc.HierarchicalTest(best_model_data, model_data, rope=rope)
            if n==None: 
                filename = path+f'{str(rope)}_{best_model}_{model}_{metric_name}.pk'
                with open(filename, 'wb') as f: 
                    pk.dump(posterior, f)
            else: 
                filename = path+f'{str(rope)}_{best_model}_{model}_{metric_name}_{str(n)}_instances.pk'
                with open(filename, 'wb') as f:
                    pk.dump(posterior, f)




                        
def generatePlots(models, best_model_idx, metric_name,rope=0.05, n=None): 
    """
    Reads probabilities calculated by bayesian function, generates figs and saves them. 
    
    :param models: list of model names
    :param best_model_idx: index of the best_model in 'models'.
    :param datasets: list of dataset names
    :param metric_name: metric name
    :param rope: ROPE param for bayesian test
    :param n: number of instances used for training
    """
    path = f'../results_baycomp/bayesian_posteriors_rope='
    fig_path = f'../bayesian_figs/bayesian_rope='
    
    best_model = models[best_model_idx]
    
    for idx, model in enumerate(models): 

        if best_model_idx != idx: 
            if n==None: 
                posterior = pk.load(open(path+f'{str(rope)}_{best_model}_{model}_{metric_name}.pk', 'rb'))
                fig = tern(posterior, l_tag=best_model, r_tag=model)
                plt.savefig(fig_path+f'{str(rope)}_{best_model}_{model}_{metric_name}.png')
            else: 
                posterior = pk.load(open(path+f'{str(rope)}_{best_model}_{model}_{metric_name}_{str(n)}_instances.pk', 'rb'))
                fig = tern(posterior, l_tag=best_model, r_tag=model)
                plt.savefig(fig_path+f'{str(rope)}_{best_model}_{model}_{metric_name}_{str(n)}_instances.png')

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
    
    
    aux_img = cv2.imread(fig_path1+f'{str(rope)}_{models[0]}_{models[1]}_{metric_name}.png')
    aux_img[:] = (242, 242, 242)
    
    final = None 
    for idx, model in enumerate(models): 
        row = None 
        for idx2, model2 in enumerate(models): 
            
            if idx==idx2: 
                img = aux_img
            else: 
                if n==None: 
                    img = cv2.imread(fig_path1+f'{str(rope)}_{model}_{model2}_{metric_name}.png')
                else: 
                    img = cv2.imread(fig_path1+f'{str(rope)}_{model}_{model2}_{metric_name}_{str(n)}_instances.png')
                
            if idx2==0: 
                row = img
            else: 
                row = cv2.hconcat([row, img])
                
        if idx == 0: 
            final = row
        else: 
            final = cv2.vconcat([final, row])
    if n==None:         
        cv2.imwrite(fig_path2+f'{str(rope)}_{metric_name}.png' ,final)
    else: 
        cv2.imwrite(fig_path2+f'{str(rope)}_{metric_name}_{str(n)}_instances.png', final)
            
    
    
if __name__=="__main__": 
    
    """
    This script performs the bayesian test for all pairs of methods defined in models list. 
    Method's scores must be already stored in results directory. 
    """

    models = [ 
        #GENERAL_METHOD, ENCODING, MODEL
        #Strategy 0
        (None, 'pam250', 'rfr'), 
        (None, 'pam250', 'abr'), 
        (None, 'pam250', 'dtr'), 
        (None, 'pam250', 'r'), 
        (None, 'pam250', 'svr'), 
        (None, 'pam250', 'knnr'), 
        (None, 'unirep', 'rfr'), 
        (None, 'unirep', 'abr'), 
        (None, 'unirep', 'dtr'), 
        (None, 'unirep', 'r'), 
        (None, 'unirep', 'svr'), 
        (None, 'unirep', 'knnr'), 
        #Strategy 1
        (None, 'dcae', 'rfr'), 
        (None, 'dcae', 'abr'), 
        (None, 'dcae', 'dtr'), 
        (None, 'dcae', 'r'), 
        (None, 'dcae', 'svr'), 
        (None, 'dcae', 'knnr'), 
        (None, 'eunirep', 'rfr'), 
        (None, 'eunirep', 'abr'), 
        (None, 'eunirep', 'dtr'), 
        (None, 'eunirep', 'r'), 
        (None, 'eunirep', 'svr'), 
        (None, 'eunirep', 'knnr'), 
        #Strategy 2
        ('MERGE', None, 'merge_rf'), 
        ('MERGE', None, 'merge_ab'), 
        ('MERGE', None, 'merge_dt'), 
        ('MERGE', None, 'merge_r'), 
        ('MERGE', None, 'merge_svm'), 
        ('MERGE', None, 'merge_knn'), 
        #Strategy 3
        ('TriTrainingRegressor', 'pam250', 'tritr_rf'), 
        ('TriTrainingRegressor', 'pam250', 'tritr_ab'), 
        ('TriTrainingRegressor', 'pam250', 'tritr_dt'), 
        ('TriTrainingRegressor', 'pam250', 'tritr_r'),
        ('TriTrainingRegressor', 'pam250', 'tritr_svm'), 
        ('TriTrainingRegressor', 'pam250', 'tritr_knn'), 
        ('CoRegression', 'pam250', 'cor'), 
        #Strategy 4
        ('TriTrainingRegressor', 'dcae', 'tritr_rf'), 
        ('TriTrainingRegressor', 'dcae', 'tritr_ab'), 
        ('TriTrainingRegressor', 'dcae', 'tritr_dt'), 
        ('TriTrainingRegressor', 'dcae', 'tritr_r'),
        ('TriTrainingRegressor', 'dcae', 'tritr_svm'), 
        ('TriTrainingRegressor', 'dcae', 'tritr_knn'), 
        ('CoRegression', 'dcae', 'cor'), 
    ]
    
    #Single substitution datasets
    datasets1 = ['bg_strsq', 'blat_ecolx_1', 'blat_ecolx_2', 'blat_ecolx_3', 'blat_ecolx_4', 'brca1_human_1', 
                'brca1_human_2', 'gal4_yeast', 'hg_flu', 'hsp82_yeast', 'mth3_haeaestabilized', 'pabp_yeast_1',
                'polg_hcvjf', 'rl401_yeast_1', 'rl401_yeast_2', 'ube4b_mouse', 'yap1_human']
    #Multiple and double substitution datasets 
    datasets2 = ['avgfp', 'pabp_yeast_2']
    
    #Both
    datasets_1_2 = datasets1+datasets2
    
    metrics = ['spearman', 'wspearman']
   

    for n in [None, 250, 200, 150, 100, 50]: 
        for metric in metrics: 
            
            best_model_idx = 28 #('MERGE', None, 'svr')
            
            model_names, overall_data, overall_data_mean = get_overall_data(models, datasets_1_2, metric, n=n)
            bayesian(model_names, overall_data, best_model_idx, datasets_1_2, metric, n=n, rope=0.01)
            generatePlots(model_names, best_model_idx, metric, n=n, rope=0.01)
            #generateBigPlot(model_names, datasets_1_2, metric, n=n, rope=0.01)


    #Pystan saves all models in caché and fills the disc. Delete jsonlines.gz in /home/user/.cache/httpstan/.../fits

      
      
