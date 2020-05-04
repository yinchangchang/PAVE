# Pattern Attention Model with Value Embedding (PAVE)

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

