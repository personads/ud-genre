#!/usr/bin/python3

import argparse, json, logging, os, sys

import numpy as np
import torch
import torch.nn.functional as F

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.ud import *
from data.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Zero-Shot Classification')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library')
	arg_parser.add_argument('embeddings', help='path to pre-computed embedding directory for UD')
	arg_parser.add_argument('out_path', help='path to experiment output directory')
	arg_parser.add_argument('-d', '--domains', help='comma-separated domains to classify (default: UD domains)')
	arg_parser.add_argument('-t', '--template', help='template to insert domains into, e.g. "This text is about @domain@." (default: None)')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size (default: 32)')
	arg_parser.add_argument('-cd', '--closed_domain', action='store_true', default=False, help='limit prediction to set of known domains (default: False)')
	arg_parser.add_argument('-rs', '--seed', type=int, default=42, help='seed for probabilistic components (default: 42)')
	return arg_parser.parse_args()


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_output_directory(args.out_path)

	# setup logging
	setup_logging(os.path.join(args.out_path, 'zero.log'))

	# set random seed
	torch.random.manual_seed(args.seed)
	np.random.seed(args.seed)

	# load Universal Dependencies
	ud = UniversalDependencies.from_directory(args.ud_path, verbose=True)
	logging.info(f"Loaded {ud}.")

	# load default domains from UD
	domains = ud.get_domains()
	# load custom domains
	if args.domains:
		assert not args.closed_domain, "[Error] Custom prediction domains cannot be combined with closed-domain prediction."
		domains = sorted([d.strip() for d in args.domains.split(',')])
	domain_labels = {domain: didx for didx, domain in enumerate(domains)}
	logging.info(f"Classifying {len(domains)} domains ({', '.join(domains)}).")

	# load pre-computed embeddings if provided
	embeddings = torch.Tensor(load_embeddings(args.embeddings, 'All'))
	logging.info(f"Loaded {embeddings.shape[0]} pre-computed embeddings from '{args.embeddings}'.")
	assert embeddings.shape[0] == len(ud), f"[Error] Number of embeddings (n={embeddings.shape[0]}) and sentences in UD (n={len(ud)}) do not match."

	# load transformer model
	model_type = 'standard'
	# sentence-level model
	if args.model.startswith('sentence/'):
		from sentence_transformers import SentenceTransformer
		model = SentenceTransformer(args.model.replace('sentence/', ''))
		tokenizer = model.tokenizer
		model.eval()
		model_type = 'sentence'
	# LASER
	elif args.model == 'laser':
		from laserembeddings import Laser
		model = Laser()
		tokenizer = None
		model_type = 'laser'
	# standard transformer
	else:
		import transformers
		transformers.set_seed(args.seed)
		tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
		model = transformers.AutoModel.from_pretrained(args.model, return_dict=True)
		# check CUDA availability
		if torch.cuda.is_available(): model.to(torch.device('cuda'))
		# set model to inference mode
		model.eval()
	logging.info(f"Loaded '{args.model}' ({model.__class__.__name__} with {tokenizer.__class__.__name__}).")

	# embed domain terms
	with torch.no_grad():
		domain_texts = list(domains)
		# insert domains into template if provided
		if args.template:
			assert '@domain@' in args.template, f"[Error] No '@domain@' placeholder in template '{args.template}'."
			domain_texts = [args.template.replace('@domain@', d) for d in domains]
			logging.info(f"Inserting domain labels into template '{args.template}'.")
		# embed batch (sentence-level)
		if model_type == 'sentence':
			# SentenceTransformer takes list[str] as input
			domain_embeddings = torch.Tensor(model.encode(domain_texts, convert_to_tensor=True)) # (batch_size, hidden_dim)
		elif model_type == 'laser':
			# run through LASER
			domain_embeddings = torch.Tensor(model.embed_sentences(domain_texts, lang='en')) # TODO: more language codes
		# embed batch (token-level)
		else:
			tkn_domains = tokenizer(domain_texts, return_tensors='pt', padding=True, truncation=True)
			# move input to GPU (if available)
			if torch.cuda.is_available():
				tkn_domains = {k: v.to(torch.device('cuda')) for k, v in tkn_domains.items()}
			# perform embedding forward pass
			model_outputs = model(**tkn_domains)
			emb_domains = model_outputs.last_hidden_state # (batch_size, max_len, hidden_dim)
			domain_embeddings = torch.zeros((emb_domains.shape[0], emb_domains.shape[2])) # (batch_size, hidden_dim)
			# iterate over sentences in batch
			for didx in range(emb_domains.shape[0]):
				# mean pooling over tokens in each domain descriptor
				att_domain = tkn_domains['attention_mask'][didx] > 0
				domain_embeddings[didx, :] = torch.mean(emb_domains[didx, att_domain, :], dim=0)
	logging.info(f"Created domain embeddings of shape {tuple(domain_embeddings.shape)}")

	# initialize results
	domain_dist = np.zeros((len(ud), len(domains)))
	corpus_map = OrderedDict([('All', OrderedDict())])

	# iterate over UD
	cursor = 0
	while cursor < len(ud):
		# set up batch
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, len(ud))
		cursor = end_idx
		emb_sentences = embeddings[start_idx:end_idx]

		# (batch_size, num_domains)
		distribution = torch.zeros((emb_sentences.shape[0], domain_embeddings.shape[0]))
		for bidx, udidx in enumerate(range(start_idx, end_idx)):
			# calculate cosine similarity (num_domains,)
			cos_similarity = F.cosine_similarity(
				emb_sentences[bidx].view(1, -1),
				domain_embeddings
			)
			# if closed domain prediction, apply softmax over known domains
			if args.closed_domain:
				# retrieve sentence domains from UD metadata
				sen_labels = [domain_labels[d] for d in ud.get_domains_of_index(udidx)]
				distribution[bidx, sen_labels] = F.softmax(cos_similarity[sen_labels], dim=-1)
			# if open domain prediction, apply softmax over all domains
			else:
				distribution[bidx, :] = F.softmax(cos_similarity, dim=-1)

		# append distribution to results
		domain_dist[start_idx:end_idx, :] = np.array(distribution.cpu())

		# update corpus map
		for sidx in range(start_idx, end_idx):
			language = ud.get_language_of_index(sidx).replace(' ', '_')
			treebank = ud.get_treebank_name_of_index(sidx)
			tb_key = f'{language}/{treebank}'
			if tb_key not in corpus_map['All']:
				corpus_map['All'][tb_key] = OrderedDict()

			tb_file = ud.get_treebank_file_of_index(sidx)
			if tb_file not in corpus_map['All'][tb_key]:
				corpus_map['All'][tb_key][tb_file] = OrderedDict()

			sentence_key = f'sentence-{ud[sidx].idx}'
			corpus_map['All'][tb_key][tb_file][sentence_key] = [sidx]

		sys.stdout.write(f"\r[{(cursor*100)/len(ud):.2f}%] Predicting...")
		sys.stdout.flush()

	# export results
	results_path = os.path.join(args.out_path, 'All.pkl')
	with open(results_path, 'wb') as fp:
		pickle.dump(
			{
				'corpus_map': corpus_map,
				'domain_dist': domain_dist,
				'domains': domains
			},
			fp
		)
	logging.info(f"\rPredicted {'closed-domain' if args.closed_domain else 'open-domain'} labels for {len(ud)} sentences. \
Saved results to '{results_path}'.")


if __name__ == '__main__':
	main()
