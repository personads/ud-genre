#!/usr/bin/python3

import argparse, json, logging, os, sys

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import sentence_transformers as st

from collections import defaultdict, OrderedDict

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from data.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Supervised Classification')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
	arg_parser.add_argument('-e', '--epochs', type=int, default=100, help='maximum number of epochs (default: 100)')
	arg_parser.add_argument('-es', '--early_stop', type=int, default=5, help='maximum number of epochs without improvement (default: 5)')
	arg_parser.add_argument('-mp', '--model_path', help='path to pretrained model (default: None - default pretrained weights)')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=64, help='maximum number of sentences per batch (default: 64)')
	arg_parser.add_argument('-lw', '--label_weights', action='store_true', default=False, help='enable weighted loss (default: False)')
	arg_parser.add_argument('-lr', '--learning_rate', type=float, default=1e-7, help='learning rate (default: 1e-7)')
	arg_parser.add_argument('-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
	return arg_parser.parse_args()


def get_batches(ud, udidcs, domain_labels, size):
	udidcs = set(udidcs)
	# gather domain sets of sentences
	sen_domains = {idx:set(ud.get_domains_of_index(idx)) for idx in udidcs}

	while len(udidcs) > 0:
		batch_idcs = list(np.random.choice(list(udidcs), min(size, len(udidcs)), replace=False))
		batch = [s.to_text() for s in ud[batch_idcs]]

		# get label (1 from sentence's set of domains) and weights (1/num_treebank_domains)
		labels, weights = zip(*[
			(domain_labels[sen_domains[idx].pop()], 1/len(ud.get_domains_of_index(idx)))
			for idx in batch_idcs
		])
		# convert labels and weights
		labels, weights = torch.LongTensor(labels), torch.FloatTensor(weights)

		# remove indices for which all domains have been exhausted
		del_idcs = {idx for idx in batch_idcs if len(sen_domains[idx]) < 1}
		udidcs = udidcs - set(del_idcs)
		num_remaining = len(udidcs)

		yield batch, labels, weights, num_remaining


def forward(model, model_type, tokenizer, loss, batch, labels=None, weights=None):
	# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]]}
	tkn_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
	# move batch to GPU (if available)
	if torch.cuda.is_available():
		tkn_batch = {k: v.to(torch.device('cuda')) for k, v in tkn_batch.items()}
		labels = labels.to(torch.device('cuda'))
		if weights is not None: weights = weights.to(torch.device('cuda'))


	# run through appropriate model
	if model_type == 'sentence':
		model_out = model.forward({k:tkn_batch[k] for k in ['input_ids', 'attention_mask']})
		logits = model_out['sentence_embedding']
	# standard type
	else:
		# perform standard transformer embedding forward pass
		model_out = model(**tkn_batch)
		logits = model_out.logits # (batch_size, num_labels)

	# calculate loss
	loss_out = loss(logits, labels) # (batch_size)
	if weights is not None: loss_out = torch.mul(loss_out, weights)
	loss_out = torch.mean(loss_out)

	return loss_out, logits


def run(model, model_type, tokenizer, loss, optimizer, batch_generator, num_total, weighted=False, mode='train'):
	stats = defaultdict(list)

	# set model to training mode
	if mode == 'train':
		model.train()
	# set model to eval mode
	elif mode == 'eval':
		model.eval()

	# iterate over batches
	for bidx, batch_data in enumerate(batch_generator):
		batch, labels, weights, num_remaining = batch_data
		if not weighted: weights = None

		# when training, perform both forward and backward pass
		if mode == 'train':
			# zero out previous gradients
			optimizer.zero_grad()

			loss_out, logits = forward(model, model_type, tokenizer, loss, batch, labels, weights)

			# propagate loss
			loss_out.backward()
			optimizer.step()

		# when evaluating, perform forward pass without gradients
		elif mode == 'eval':
			with torch.no_grad():
				loss_out, logits = forward(model, model_type, tokenizer, loss, batch, labels, weights)

		# store statistics
		stats[f'{mode}/loss'].append(float(loss_out))
		if weighted: stats[f'{mode}/weights'].append(float(torch.mean(weights)))

		# print batch statistics
		pct_complete = (1 - (num_remaining/num_total))*100
		sys.stdout.write(f"\r[{mode.capitalize()} | Batch {bidx+1} | {pct_complete:.2f}%] \
Size: {tuple(logits.shape)}, Weights: {'%.4f' % np.mean(stats[f'{mode}/weights']) if weighted else 'None'}, \
Loss: {loss_out:.4f} ({np.mean(stats[f'{mode}/loss']):.4f} mean)")
		sys.stdout.flush()

	# clear line
	print("\r", end='')

	return stats


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_success = setup_output_directory(args.out_path)
	if not setup_success: return

	# setup logging
	setup_logging(os.path.join(args.out_path, 'classify.log'))

	# set random seed
	transformers.set_seed(args.seed)
	torch.random.manual_seed(args.seed)
	np.random.seed(args.seed)

	# load data split definition (if supplied)
	ud_filter = None
	split_idcs = None
	if args.split:
		with open(args.split, 'rb') as fp:
			split_idcs = pickle.load(fp)
		# create filter to load only relevant indices (train, dev)
		relevant_idcs = set(split_idcs['train']) | set(split_idcs['dev'])
		ud_filter = UniversalDependenciesIndexFilter(relevant_idcs)
		logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in split_idcs.items()])} with filter {ud_filter}.")

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	domains = ud.get_domains()
	domain_labels = {domain:didx for didx, domain in enumerate(domains)}
	logging.info(f"Loaded {ud} with {len(domains)} domains ({', '.join(domains)}).")

	# use all of UD for each split if none are provided
	if split_idcs is None:
		split_idcs = {split: list(range(len(ud))) for split in ['train', 'dev', 'test']}

	# load transformer model
	model_type = 'standard'
	if args.model.startswith('sentence/'):
		if args.model_path:
			model = st.SentenceTransformer(args.model_path)
		else:
			# load base sentence transformer and extract modules
			base_model = st.SentenceTransformer(args.model.replace('sentence/', ''))
			modules = base_model._modules # OrderedDict
			# add linear layer on top (emb_dim, num_domains)
			modules['classifier'] = st.models.Dense(
				in_features=base_model._last_module().out_features,
				out_features=len(domains),
				bias=True,
				activation_function=torch.nn.Identity() # just use as linear layer
			)
			# recombine into model
			model = st.SentenceTransformer(modules=modules)
		tokenizer = model._first_module().tokenizer
		model_type = 'sentence'
	else:
		tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
		model = transformers.AutoModelForSequenceClassification.from_pretrained(
			args.model_path if args.model_path else args.model,
			num_labels=len(domains),
			return_dict=True
		)
		objective = None # use built in loss function
	logging.info(f"Loaded {model_type}-type '{args.model}' ({model.__class__.__name__} with {tokenizer.__class__.__name__}).")

	# move to CUDA device if available
	if torch.cuda.is_available(): model.to(torch.device('cuda'))

	# initialize loss
	loss = torch.nn.CrossEntropyLoss(reduction='none')
	logging.info(f"Using {loss.__class__.__name__} {'with' if args.label_weights else 'without'} weighted labels.")
	# initialize optimizer
	optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)
	logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")

	# main training loop
	stats = defaultdict(list)
	for ep_idx in range(args.epochs):
		# iterate over batches with known domain labels
		train_batches = get_batches(ud, split_idcs['train'], domain_labels, args.batch_size)
		cur_stats = run(
			model, model_type, tokenizer, loss, optimizer,
			train_batches, len(split_idcs['train']),
			weighted=args.label_weights
		)
		# store statistics
		stats['train/loss'].append(np.mean(cur_stats['train/loss']))
		# print training statistics
		logging.info(f"[Epoch {ep_idx+1}/{args.epochs}] Training completed with \
Weights: {'%.4f' % np.mean(cur_stats['train/weights']) if args.label_weights else 'None'}, \
Loss: {np.mean(cur_stats['train/loss']):.4f} ({max(cur_stats['train/loss']):.4f} max, {min(cur_stats['train/loss']):.4f} min)")

		# iterate over batches with known domain labels
		eval_batches = get_batches(ud, split_idcs['dev'], domain_labels, args.batch_size)
		cur_stats = run(
			model, model_type, tokenizer, loss, optimizer,
			eval_batches, len(split_idcs['dev']),
			weighted=args.label_weights, mode='eval'
		)
		# store statistics
		stats['eval/loss'].append(np.mean(cur_stats['eval/loss']))
		# print training statistics
		logging.info(f"[Epoch {ep_idx+1}/{args.epochs}] Evaluation completed with \
Weights: {'%.4f' % np.mean(cur_stats['eval/weights']) if args.label_weights else 'None'}, \
Loss: {np.mean(cur_stats['eval/loss']):.4f} ({max(cur_stats['eval/loss']):.4f} max, {min(cur_stats['eval/loss']):.4f} min)")

		# save most recent model
		model_path = os.path.join(args.out_path, 'newest')
		if model_type == 'sentence':
			model.save(model_path)
		else:
			model.save_pretrained(model_path)
		logging.info(f"Saved model from epoch {ep_idx + 1} to '{model_path}'.")

		# save best model
		if np.mean(cur_stats['eval/loss']) <= min(stats['eval/loss']):
			model_path = os.path.join(args.out_path, 'best')
			if model_type == 'sentence':
				model.save(model_path)
			else:
				model.save_pretrained(model_path)
			logging.info(f"Saved model with best loss {np.mean(cur_stats['eval/loss']):.4f} to '{model_path}'.")

		# check for early stopping
		if (ep_idx - stats['eval/loss'].index(min(stats['eval/loss']))) >= args.early_stop:
			logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['eval/loss']):.4f} loss). Early stop.")
			break


if __name__ == '__main__':
	main()
