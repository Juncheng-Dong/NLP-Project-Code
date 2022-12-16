# Install all required packages 
pip install -r requirements.txt 

# Command for running experiments

## Flags for control
(--flag [value_to_pass] : meaning)
--no_train : zero-shot
--Bayesian : using Bayesian method
--tau 0.25: hyperparameter of the Bayesian method
--model_name_or_path [roberta-large/roberta-base]: which langauage model to use
-- few_shot_type [prompt-demo/prompt]: whether to use demonstration or not

# SST-2 Dataset 
python run.py     --task_name SST-2     --data_dir data/k-shot/SST-2/16-42     --overwrite_output_dir     --do_train     --do_eval     --do_predict     --evaluate_during_training     --model_name_or_path roberta-large    --few_shot_type prompt-demo   --num_k 16     --max_steps 1000     --eval_steps 100     --per_device_train_batch_size 2     --learning_rate 1e-5     --num_train_epochs 0     --output_dir result/tmp     --seed 42     --template "*cls**sent_0*_It_was*mask*.*sep+*"     --mapping "{'0':'terrible','1':'great'}"     --num_sample 16 --mapping_score_path /hpc/home/jd420/Projects/LM-BFF/auto_label_mapping/SST-2/16-42.score.txt 

# MNLI Dataset 

python run.py     --task_name MNLI     --data_dir data/k-shot/MNLI/16-100     --overwrite_output_dir     --do_train     --do_eval     --do_predict     --evaluate_during_training     --model_name_or_path roberta-large     --few_shot_type prompt     --num_k 16     --max_steps 1000     --eval_steps 100     --per_device_train_batch_size 2     --learning_rate 1e-5     --num_train_epochs 0     --output_dir result/tmp     --seed 100     --template "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"   --mapping "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"  --mapping_score_path /hpc/home/jd420/Projects/LM-BFF/auto_label_mapping/MNLI/16-100.score.txt --max_seq_len 256 --num_sample 4 

