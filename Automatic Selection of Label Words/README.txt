For running the automatic selection of label words experiments which are described in our report, one would need to choose the following:
1. Experiment type on /tools/run_generate_labels.sh line 169:
	adding --no_train flag runs a zero-shot scenario while removing it runs prompt-based fine-tuning.

2. Samples type on /tools/run_generate_labels.sh line 163:
	TYPE=prompt-demo runs the experiment with demonstrations, TYPE=prompt and runs the experiment with no demonstrations.

3. Metric parameters on /tools/generate_labels.py lines 252-265: 
	k_or_p parameter allows choosing between Nucleus or top-k sampling.
	p_likely parameter controls the probability threshold for Nucleus sampling.
	only_true_label=True runs the original metric and only_true_label=False runs our metric as described in the report.
	false_labels_weight parameter sets lambda as described in the report.
	
After choosing these, running /tools/run_generate_labels.sh script will yield the label mappings and their accuracy over the development and test sets.