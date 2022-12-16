import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from tools.my_generate_template import generate_template


def best_template(mapping):
	generate_template(mapping = mapping)
	print("\n\nTemplates generated.\n\n")
	with open("my_auto_template/SST-2/16-13.txt", "r") as f:
		lines = f.readlines()
	templates = [line.strip() for line in lines]
	eval_loss, eval_accu, test_loss, test_accu = [], [], [], []
	for template in templates:
		command = 'python run.py --task_name SST-2 --data_dir data/k-shot/SST-2/16-42 --overwrite_output_dir --do_train --do_eval --do_predict --evaluate_during_training' +\
			' --model_name_or_path roberta-base --few_shot_type prompt --num_k 16 --max_steps 100 --eval_steps 10 --per_device_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 0' +\
			f' --output_dir result/tmp --seed 42 --template "{template}" --mapping "{mapping}" --num_sample 16'
		os.system(command)
		with open("result/tmp/eval_results_sst-2.txt", "r") as f:
			lines = f.readlines()
			eval_loss.append(eval(lines[0][lines[0].find(" = ") + 3:-1]))
			eval_accu.append(eval(lines[1][lines[1].find(" = ") + 3:-1]))
		with open("result/tmp/test_results_sst-2.txt", "r") as f:
			lines = f.readlines()
			test_loss.append(eval(lines[0][lines[0].find(" = ") + 3:-1]))
			test_accu.append(eval(lines[1][lines[1].find(" = ") + 3:-1]))
		print(f"\n\ntemplate {template} evaluated - eval_loss: {eval_loss[-1]}, eval_accu: {eval_accu[-1]}, test_loss: {test_loss[-1]}, test_accu: {test_accu[-1]}\n\n")
	df = pd.DataFrame({"templates": templates, "eval_loss": eval_loss, "eval_accu": eval_accu, "test_loss": test_loss, "test_accu": test_accu})
	return df, df.templates[df.test_accu.argmax()], df.test_accu.max()


def best_mapping(template):
	temp_mapping = {'0':'zero','1':'one'}
	command = 'python tools/generate_labels.py --overwrite_output_dir --output_dir /tmp/output --model_name_or_path roberta-base' +\
		f' --output_file my_auto_label_mapping/manual_template/SST-2/16-13.txt --template {template} --mapping "{temp_mapping}"' +\
		' --task_name SST-2 --data_dir data/k-shot/SST-2/16-13 --k_likely 10 --k_neighbors 5 --n_pairs 10 --max_seq_len 256 --per_device_eval_batch_size 16'
	print("\n\nMappings generated.\n\n")
	os.system(command)
	with open("my_auto_label_mapping/manual_template/SST-2/16-13.txt", "r") as f:
		lines = f.readlines()
	mappings = [eval(line[:-1]) for line in lines]
	eval_loss, eval_accu, test_loss, test_accu = [], [], [], []
	for mapping in mappings:
		command = 'python run.py --task_name SST-2 --data_dir data/k-shot/SST-2/16-42 --overwrite_output_dir --do_train --do_eval --do_predict --evaluate_during_training' +\
			' --model_name_or_path roberta-base --few_shot_type prompt --num_k 16 --max_steps 100 --eval_steps 10 --per_device_train_batch_size 1 --learning_rate 1e-5 --num_train_epochs 0' +\
			f' --output_dir result/tmp --seed 42 --template "{template}" --mapping "{mapping}" --num_sample 16'
		os.system(command)
		with open("result/tmp/eval_results_sst-2.txt", "r") as f:
			lines = f.readlines()
			eval_loss.append(eval(lines[0][lines[0].find(" = ") + 3:-1]))
			eval_accu.append(eval(lines[1][lines[1].find(" = ") + 3:-1]))
		with open("result/tmp/test_results_sst-2.txt", "r") as f:
			lines = f.readlines()
			test_loss.append(eval(lines[0][lines[0].find(" = ") + 3:-1]))
			test_accu.append(eval(lines[1][lines[1].find(" = ") + 3:-1]))
		print(f"\n\nmapping {mapping} evaluated - eval_loss: {eval_loss[-1]}, eval_accu: {eval_accu[-1]}, test_loss: {test_loss[-1]}, test_accu: {test_accu[-1]}\n\n")
	df = pd.DataFrame({"mappings": mappings, "eval_loss": eval_loss, "eval_accu": eval_accu, "test_loss": test_loss, "test_accu": test_accu})
	return df, df.mappings[df.test_accu.argmax()], df.test_accu.max()


def iterate(n_iter, init_mapping, init_template):
	directory = os.path.join("iterative_results", str(datetime.now(timezone('EST'))).split(".")[0].replace(":", "-").replace(" ", "-"))
	os.mkdir(directory)
	with open(os.path.join(directory, "init.txt"), "w") as f:
		f.write(f"n_iter: {n_iter}\n")
		f.write(f"init_mapping: {init_mapping}\n")
		f.write(f"init_template: {init_template}\n")
	results, accuracies = [], [-1]
	if init_mapping is None:
		template = init_template
		results.append(template)
		for i in range(n_iter):
			df, mapping, accu = best_mapping(template)
			df.to_csv(os.path.join(directory, f"iter_{i + 1}_mapping.csv"), index = False)
			results.append(mapping)
			accuracies.append(accu)
			df, template, accu = best_template(mapping)
			df.to_csv(os.path.join(directory, f"iter_{i + 1}_template.csv"), index = False)
			results.append(template)
			accuracies.append(accu)
	else:
		mapping = init_mapping
		results.append(mapping)
		for i in range(n_iter):
			df, template, accu = best_template(mapping)
			df.to_csv(os.path.join(directory, f"iter_{i + 1}_template.csv"), index = False)
			results.append(template)
			accuracies.append(accu)
			df, mapping, accu = best_mapping(template)
			df.to_csv(os.path.join(directory, f"iter_{i + 1}_mapping.csv"), index = False)
			results.append(mapping)
			accuracies.append(accu)
	with open(os.path.join(directory, "results.txt"), "w") as f:
		for result, accu in zip(results, accuracies):
			f.write(f"{result}  ---  {accu}\n")


# print(best_template({'0':'terrible','1':'great'}))
# print(best_mapping("*cls**sent_0*_It_was*mask*.*sep+*"))
# iterate(n_iter = 1, init_mapping = None, init_template = "*cls**sent_0*_It_was*mask*.*sep+*")

iterate(n_iter = 4, init_mapping = None, init_template = "*cls**sent_0*_It_was*mask*.*sep+*")
iterate(n_iter = 4, init_mapping = {"0": "negative", "1": "positive"}, init_template = None)
iterate(n_iter = 4, init_mapping = None, init_template = "*cls**sent_0*_*mask*.*sep+*")
iterate(n_iter = 4, init_mapping = {"0": "zero", "1": "one"}, init_template = None)

