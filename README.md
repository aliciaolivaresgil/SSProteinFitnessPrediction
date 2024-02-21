# SSProteinFitnessPrediction

This repository contains the code needed to perform the experiments described in "Semi-supervised prediction of protein fitness for data-driven protein engineering" by Alicia Olivares-Gil, José A. Barbero-Aparicio, Juan J. Rodríguez, José F. Díez-Pastor, César García-Osorio and Mehdi D. Davari. 

It is a systematic comparison, on 19 different datasets, of several semi-supervised regression methods for predicting the fitness of protein variants. Mainly, in situations where the number of protein sequences with a known fitness (labeled instances) is scarce. 

## Requirements
In order to reproduce the results of this work, some prerequisites must be met: 

### 1. Download UniRef100 database
Some methods (MERGE) and encodings (DCA encoding, eUnirep encoding) need to perform a homologous search on the UniRef100 database. 

Download: 
```
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz
```
Unzip file: 
```
gzip -d uniref100.fasta.gz
```

### 2. Install HMMER
This software is needed to perform the homologous search (jackhmmer search) on the UniRef100 database. 

Instructions: http://hmmer.org/documentation.html

### 3. Install PLMC
This software is needed to infer the Coupling Analysis statistical model used by MERGE and the DCA encoding. 

Instructions: https://github.com/debbiemarkslab/plmc

### 4. Conda environments
All the Python code was executed using the conda environments available in this repository:  
- `experiments_env.yml`: Used to execute all the experiments except the Hierarchical Bayesian tests.
- `bayesian_tests_env.yml`: Used to execute the Bayesian tests.

To install these conda environments: 
```
conda env create -f experiments_env.yml
```
To activate the environment: 
```
conda activate experiments_env
```

### 5. Datasets

The 19 datasets used for this comparison will be soon available in a public Zenodo repository. 

|                   **Dataset** | **Sequence length** | **#Sequences** | **Substitutions** |
|------------------------------:|:-------------------:|:--------------:|:-----------------:|
|         YAP1 HUMAN Fields2021 |          34         |       313      |       single      |
|        UBE4B_MOUSE_Klevit2013 |         102         |       518      |       single      |
|       GAL4_YEAST_Shendure2015 |          64         |       803      |       single      |
|      BLAT_ECOLX_Tenaillon2013 |         286         |       975      |       single      |
|         PABP_YEAST_Fields2013 |          75         |      1142      |       single      |
|         RL401_YEAST_Bolon2013 |          75         |      1154      |       single      |
|    BRCA1_HUMAN_Fields2015_y2h |         303         |      1278      |       single      |
|         RL401_YEAST_Bolon2014 |          75         |      1282      |       single      |
| MTH3_HAESTABILIZED_Tawfik2015 |         329         |      1611      |       single      |
|            POLG_HCVJF_Sun2014 |          86         |      1613      |       single      |
|            BG_STRSQ_Abate2015 |         478         |      2598      |       single      |
|     BRCA1_HUMAN_Fields2015_e3 |         303         |      2846      |       single      |
|         HSP82_YEAST_Bolon2016 |         230         |      4065      |       single      |
|     BLAT_ECOLX_Ostermeier2014 |         286         |      4799      |       single      |
|    BLAT_ECOLX_Ranganathan2015 |         286         |      4921      |       single      |
|       BLAT_ECOLX_Palzkill2012 |         286         |      4922      |       single      |
|              HG_FLU_Bloom2016 |         564         |      10337     |       single      |
|         PABP_YEAST_Fields2013 |          75         |      33771     |       double      |
|          avGPF_Kondrashov2016 |         235         |      32610     |      multiple     |

## Usage 
In order to reproduce the results shown in the paper, follow these steps: 

### 1. jackhmmer search and Coupling Analysis 
For each dataset, a jackhmmer search on the UniRep100 database must be performed. The homologous protein sequences found are used to construct a MSA that will be consequently pre-processed before bein used as input for interring a Coupling Analysis statistical model using PLMC. 

An example of this is available in [Homologous sequence search example.ipynb](https://github.com/aliciaolivaresgil/SSProteinFitnessPrediction/blob/main/Homologous%20sequence%20search%20example.ipynb).

To perform these steps for all the datasets: 
```
python scripts/jackhmmer.py [Uniref100.fasta path in your computer]
```
> [!IMPORTANT]
> Make sure that the paths defined in each script match the paths of the files on your computer.

### 2. Encoding of the sequences
All the sequences (sometimes including the homologous sequences found in the previous step) are encoded using the following encoding methods: 
- PAM250 encoding
- Unirep encoding
- eUnirep encoding
- DCA encoding

An aexample of this is available in [Encoding example.ipynb](https://github.com/aliciaolivaresgil/SSProteinFitnessPrediction/blob/main/Encoding%20example.ipynb). 

To perform this step for all the datasets: 
```
python scripts/encoding.py
```

### 3. Comparison with few labeled sequences
Strategies 0 and 1 (see publication): 
```
python Comparison_SUPERVISED_Few_Instances.py
```
Strategy 2 (see publication): 
```
python Comparison_MERGE_Few_Instances.py
```
Strategies 3 and 4 (see publication): 
```
python Comparison_WRAPPER_Few_Instances.py
```
> [!IMPORTANT]
> Make sure that the paths defined in each script match the paths of the files on your computer.

### 4. Hyperparameters selection 
In the previous step, a grid-search to find the best hyperparameters for the Wrapper methods is performed. To be able to use these values in the next step it is necessary tu execute the following script: 
```
python hyperparams.py
```

### 5. Comparison with all the available labeled sequences
Strategies 0 and 1 (see publication): 
```
python Comparison_SUPERVISED.py
```
Strategy 2 (see publication): 
```
python Comparison_MERGE.py
```
Strategies 3 and 4 (see publication): 
```
python Comparison_WRAPPER.py
```
> [!IMPORTANT]
> Make sure that the paths defined in each script match the paths of the files on your computer.

### 6. Create resulting plots
To visually check the results of the comparison, use the functions defined in [Results.ipynb](https://github.com/aliciaolivaresgil/SSProteinFitnessPrediction/blob/main/Results.ipynb). 

### 7. Hierarchical Bayesian statistical tests
To check whether the differences between the best method found and the rest are significant, a Hierarchical Bayesian test is run. 
```
python scripts/Statistical_Tests.py
```

To visually check the results of these tests, use the functions defined in [Statistical Tests.ipynb](https://github.com/aliciaolivaresgil/SSProteinFitnessPrediction/blob/main/Statistical%20Tests.ipynb).

## References
