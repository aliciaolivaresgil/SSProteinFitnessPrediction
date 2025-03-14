# Semi-supervised Protein Fitness Prediction

This repository contains the code needed to perform the experiments described in "Semi-supervised prediction of protein fitness for data-driven protein engineering" by Alicia Olivares-Gil, José A. Barbero-Aparicio, Juan J. Rodríguez, José F. Díez-Pastor, César García-Osorio and Mehdi D. Davari. 

It is a systematic comparison, on 19 different datasets, of several semi-supervised regression methods for predicting the fitness of protein variants. Mainly, in situations where the number of protein sequences with a known fitness (labeled instances) is scarce. 

![Semi-supervised strategies](https://github.com/aliciaolivaresgil/SSProteinFitnessPrediction/blob/main/ilustrations/methods.png)

## Requirements
In order to reproduce the results of this work, some prerequisites must be met: 

### 1. Download UniRef100 database
Some methods (MERGE) and encodings (DCA encoding, eUnirep encoding) need to perform a homologous search on the UniRef100 database. 

Download: 
```
wget https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref100/uniref100.fasta.gz
```
Or selecting the corresponding fasta file in here: https://www.uniprot.org/help/downloads

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

|                               **Dataset** | **Sequence length** | **#Sequences** | **Substitutions** |
|------------------------------------------:|:-------------------:|:--------------:|:-----------------:|
|           YAP1 HUMAN Fields2021 [[3]](#3) |          34         |       313      |       single      |
|        UBE4B_MOUSE_Klevit2013 [[12]](#12) |         102         |       518      |       single      |
|         GAL4_YEAST_Shendure2015 [[8]](#8) |          64         |       803      |       single      |
|        BLAT_ECOLX_Tenaillon2013 [[7]](#7) |         286         |       975      |       single      |
|           PABP_YEAST_Fields2013 [[4]](#4) |          75         |      1142      |       single      |
|           RL401_YEAST_Bolon2013 [[2]](#2) |          75         |      1154      |       single      |
|    BRCA1_HUMAN_Fields2015_y2h [[10]](#10) |         303         |      1278      |       single      |
|           RL401_YEAST_Bolon2014 [[1]](#1) |          75         |      1282      |       single      |
| MTH3_HAESTABILIZED_Tawfik2015 [[11]](#11) |         329         |      1611      |       single      |
|              POLG_HCVJF_Sun2014 [[6]](#6) |          86         |      1613      |       single      |
|            BG_STRSQ_Abate2015 [[16]](#16) |         478         |      2598      |       single      |
|     BRCA1_HUMAN_Fields2015_e3 [[10]](#10) |         303         |      2846      |       single      |
|         HSP82_YEAST_Bolon2016 [[15]](#15) |         230         |      4065      |       single      |
|       BLAT_ECOLX_Ostermeier2014 [[5]](#5) |         286         |      4799      |       single      |
|    BLAT_ECOLX_Ranganathan2015 [[13]](#13) |         286         |      4921      |       single      |
|       BLAT_ECOLX_Palzkill2012 [[17]](#17) |         286         |      4922      |       single      |
|              HG_FLU_Bloom2016 [[14]](#14) |         564         |      10337     |       single      |
|           PABP_YEAST_Fields2013 [[4]](#4) |          75         |      33771     |       double      |
|            avGPF_Kondrashov2016 [[9]](#9) |         235         |      32610     |      multiple     |

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

<a id="1">[1]</a> Benjamin P Roscoe and Daniel NA Bolon. Systematic exploration of ubiquitin sequence, e1 activation efficiency, and experimental fitness in yeast. Journal of molecular biology, 426(15):2854–2870, 2014.

<a id="2">[2]</a> Benjamin P Roscoe, Kelly M Thayer, Konstantin B Zeldovich, David Fushman, and Daniel NA Bolon. Analyses of the effects of all ubiquitin point mutants on yeast growth rate. Journal of molecular biology, 425(8):1363–1377, 2013.

<a id="3">[3]</a> Carlos L Araya, Douglas M Fowler, Wentao Chen, Ike Muniez, Jeffery W Kelly, and Stanley Fields. A fundamental protein property, thermodynamic stability, revealed solely from large-scale measurements of protein function. Proceedings of the National Academy of Sciences, 109(42):16858–16863, 2012.

<a id="4">[4]</a> Daniel Melamed, David L Young, Caitlin E Gamble, Christina R Miller, and Stanley Fields. Deep mutational scanning of an rrm domain of the saccharomyces cerevisiae poly (a)-binding protein. Rna, 19(11):1537–1551, 2013.

<a id="5">[5]</a> Elad Firnberg, Jason W Labonte, Jeffrey J Gray, and Marc Ostermeier. A comprehensive, high-resolution map of a gene’s fitness landscape. Molecular biology and evolution, 31(6):1581–1592, 2014.

<a id="6">[6]</a> Hangfei Qi, C Anders Olson, Nicholas C Wu, Ruian Ke, Claude Loverdo, Virginia Chu, Shawna Truong, Roland Remenyi, Zugen Chen, Yushen Du, et al. A quantitative high-resolution genetic profile rapidly identifies sequence determinants of hepatitis c viral fitness and drug sensitivity. PLoS pathogens, 10(4):e1004064, 2014.

<a id="7">[7]</a> Hervé Jacquier, André Birgy, Hervé Le Nagard, Yves Mechulam, Emmanuelle Schmitt, Jérémy Glodt, Beatrice Bercot, Emmanuelle Petit, Julie Poulain, Guil`ene Barnaud, et al. Capturing the mutational landscape of the beta-lactamase tem-1. Proceedings of the National Academy of Sciences, 110(32):13067–13072, 2013.

<a id="8">[8]</a> Jacob O Kitzman, Lea M Starita, Russell S Lo, Stanley Fields, and Jay Shendure. Massively parallel single-amino-acid mutagenesis. Nature methods, 12(3):203–206, 2015.

<a id="9">[9]</a> Karen S Sarkisyan, Dmitry A Bolotin, Margarita V Meer, Dinara R Usmanova, Alexander S Mishin, George V Sharonov, Dmitry N Ivankov, Nina G Bozhanova, Mikhail S Baranov, Onuralp Soylemez, et al. Local fitness landscape of the green fluorescent protein. Nature, 533(7603):397–401, 2016.

<a id="10">[10]</a> Lea M Starita, David L Young, Muhtadi Islam, Jacob O Kitzman, Justin Gullingsrud, Ronald J Hause, Douglas M Fowler, Jeffrey D Parvin, Jay Shendure, and Stanley Fields. Massively parallel functional analysis of brca1 ring domain variants. Genetics, 200(2):413–422, 2015.

<a id="11">[11]</a> Liat Rockah-Shmuel, Ágnes Tóth-Petróczy, and Dan S Tawfik. Systematic mapping of protein mutational space by prolonged drift reveals the deleterious effects of seemingly neutral mutations. PLoS computational biology, 11(8):e1004421, 2015.

<a id="12">[12]</a> Lea M Starita, Jonathan N Pruneda, Russell S Lo, Douglas M Fowler, Helen J Kim, Joseph B Hiatt, Jay Shendure, Peter S Brzovic, Stanley Fields, and Rachel E Klevit. Activity-enhancing mutations in an e3 ubiquitin ligase identified by high-throughput mutagenesis. Proceedings of the National Academy of Sciences, 110(14):E1263–E1272, 2013.

<a id="13">[13]</a> Michael A Stiffler, Doeke R Hekstra, and Rama Ranganathan. Evolvability as a function of purifying selection in tem-1 β-lactamase. Cell, 160(5):882–892, 2015.

<a id="14">[14]</a> Michael B Doud and Jesse D Bloom. Accurate measurement of the effects of all amino-acid mutations on influenza hemagglutinin. Viruses, 8(6):155, 2016.

<a id="15">[15]</a> Parul Mishra, Julia M Flynn, Tyler N Starr, and Daniel NA Bolon. Systematic mutant analyses elucidate general and client-specific aspects of hsp90 function. Cell reports, 15(3):588–598, 2016.

<a id="16">[16]</a> Philip A Romero, Tuan M Tran, and Adam R Abate. Dissecting enzyme function with microfluidic-based deep mutational scanning. Proceedings of the National Academy of Sciences, 112(23):7159–7164, 2015.

<a id="17">[17]</a> Zhifeng Deng, Wanzhi Huang, Erol Bakkalbasi, Nicholas G Brown, Carolyn J Adamski, Kacie Rice, Donna Muzny, Richard A Gibbs, and Timothy Palzkill. Deep sequencing of systematic combinatorial libraries reveals β-lactamase sequence constraints at high resolution. Journal of molecular biology, 424(3-4):150–167, 2012.




