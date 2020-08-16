# Pattern Attention Model with Value Embedding (PAVE)


This repository contains the official PyTorch implementation of the following paper:

> ** An Interpretable Risk Prediction Model forHealthcare with Pattern Attention (ICIBM2020)**<br>
> Sundreen Asad Kamal, Changchang Yin, Buyue Qian and Ping Zhang <br>
>
> **Abstract:** 
*Background: The availability of massive amount of data enables the possibility of clinical predictive tasks.Deep learning methods have achieved promising performance on the tasks. However, most existing methodssuffer from three limitations: (i) There are lots of missing value for real value events, many methods imputethe missing value and then train their models based on the imputed values, which may introduce imputationbias. The models’ performance is highly dependent on the imputation accuracy. (ii) Lots of existing studies justtake Boolean value medical events (e.g. diagnosis code) as inputs, but ignore real value medical events (e.g.,lab tests and vital signs), which are more important for acute disease (e.g., sepsis) and mortality prediction.(iii) Existing interpretable models can illustrate which medical events are conducive to the output results, butare not able to give contributions of patterns among medical events.*
>
> *Methods:In this study, we propose a novel interpretablePatternAttention model withValueEmbedding(PAVE) to predict the risks of certain diseases. PAVE takes the embedding of various medical events, theirvalues and the corresponding occurring time as inputs, leverage self-attention mechanism to attend tomeaningful patterns among medical events for risk prediction tasks. Because only the observed values areembedded into vectors, we don’t need to impute the missing values and thus avoids the imputations bias.Moreover, the self-attention mechanism is helpful for the model interpretability, which means the proposedmodel can output which patterns cause high risks.*
>
> *Results:We conduct sepsis onset prediction and mortality prediction experiments on a publicly availabledataset MIMIC-III and our proprietary EHR dataset. The experimental results show that PAVE outperformsexisting models. Moreover, by analyzing the self-attention weights, our model outputs meaningful medicalevent patterns related to mortality.*
>
> *Conclusions:PAVE learns effective medical event representation by incorporating the values and occurringtime, which can improve the risk prediction performance. Moreover, the presented self-attention mechanismcan not only capture patients’ health state information, but also output the contributions of various medicalevent patterns, which pave the way for interpretable clinical risk predictions.*


## Build the env
	
		pip install -r requirement.txt

## Data preparing
-	Put your data in the folder ./data/
-	There are three csv files:
	-	demo.csv: patients' demographics
	-	label.csv: ground truth
	-	data.csv: temporal records

## Data preprocessing

-	Creat result folder for data preprocessing results

		mkdir result
		mkdir data
		mkdir data/models

-	Generate json files 

		cd preprocessing/
		python gen_master_feature.py 
		python gen_feature_time.py 
		python gen_vital_feature.py 
		python gen_label.py 


##	Train and validate the model, the best model will saved in ../data/models/
		
		cd ../code/
		python main.py 

##	Test

		python main.py --phase test --resume ../data/models/best.ckpt

