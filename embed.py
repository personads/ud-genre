#!/usr/bin/python3

import argparse, hashlib, json, logging, os, sys

import numpy as np
import torch

# local imports
from data.ud import *
from data.utils import *


def parse_arguments():
	arg_parser = argparse.ArgumentParser(description='Universal Dependencies - Contextual Embedding')
	arg_parser.add_argument('ud_path', help='path to Universal Dependencies directory')
	arg_parser.add_argument('model', help='model name in the transformers library or path to model')
	arg_parser.add_argument('out_path', help='path to output directory')
	arg_parser.add_argument('-s', '--split', help='path to data split definition pickle (default: None - full UD)')
	arg_parser.add_argument('-ml', '--model_layer', type=int, default=-1, help='embedding layer to export (default: last)')
	arg_parser.add_argument('-bs', '--batch_size', type=int, default=64, help='batch size while embedding (default: 64)')
	arg_parser.add_argument('-pl', '--pooling', default='mean', choices=['mean', 'cls', 'sw-mean', 'none'], help='pooling strategy (default: mean)')
	arg_parser.add_argument('-of', '--out_format', default='numpy', choices=['numpy', 'map', 'txt'], help='output format (default: numpy)')
	return arg_parser.parse_args()

#
# pooling strategies
#


def reduce_pieces_to_words(tokenizer, sentences, tok_batch, emb_pieces):
	emb_words = torch.zeros_like(emb_pieces)
	att_words = torch.zeros(emb_pieces.shape[:-1], dtype=torch.bool, device=emb_pieces.device)
	max_len = 0
	# iterate over sentences
	for sidx in range(emb_pieces.shape[0]):
		# get string tokens of current sentence
		tokens = tokenizer.convert_ids_to_tokens(tok_batch['input_ids'][sidx])
		offsets = tok_batch['offset_mapping'][sidx]

		tidx = -1
		for widx, orig_word in enumerate(sentences[sidx]):
			# init aggregate word embedding
			emb_word = torch.zeros(emb_pieces.shape[-1], device=emb_pieces.device)  # (emb_dim,)
			num_tokens = 0
			coverage = 0
			while coverage < len(orig_word):
				tidx += 1
				if tidx >= len(emb_pieces[sidx, :]):
					print("Sent:", sentences[sidx])
					print("Tokens:", tokens)
					print("Offsets:", offsets)
					print(f"{tidx} >= {len(emb_pieces[sidx, :])}")
				# skip if special tokens ([CLS], [SEQ], [PAD])
				if tok_batch['special_tokens_mask'][sidx, tidx] == 1: continue

				token_span = offsets[tidx] # (start_idx, end_idx + 1) within orig_word
				# add WordPiece embedding to current word embedding sum
				emb_word += emb_pieces[sidx, tidx]
				num_tokens += 1
				coverage = token_span[1]

			# add mean of aggregate WordPiece embeddings and set attention to True
			emb_words[sidx, widx] = emb_word / num_tokens
			att_words[sidx, widx] = True

		# store new maximum sequence length
		max_len = len(sentences[sidx]) if len(sentences[sidx]) > max_len else max_len

	# reduce embedding and attention matrices to new maximum length
	emb_words = emb_words[:, :max_len, :] # (batch_size, max_len, emb_dim)
	att_words = att_words[:, :max_len] # (batch_size, max_len)

	return emb_words, att_words


def gather_subwords(embeddings, vocab_embs, sentences, emb_words):
	for sidx, sentence in enumerate(sentences):
		for widx, word in enumerate(sentence):
			# aggreate embeddings and counts
			embeddings.append(emb_words[sidx, widx])
			vocab_embs[word].append(len(embeddings) - 1)
	return embeddings, vocab_embs


def mean_pool_sentences(embeddings, emb_batch, att_batch):
	for sidx in range(emb_batch.shape[0]):
		embeddings.append(torch.mean(emb_batch[sidx, att_batch[sidx], :], dim=0))
	return embeddings


def gather_sentence_cls(embeddings, emb_batch):
	for sidx in range(emb_batch.shape[0]):
		embeddings.append(emb_batch[sidx, 0, :])
	return embeddings


def gather_nonpad_embeddings(embeddings, emb_batch, att_batch):
	for sidx in range(emb_batch.shape[0]):
		embeddings.append(emb_batch[sidx, att_batch[sidx], :])
	return embeddings

#
# export formats
#


def export_numpy(ud, ud_idcs, embeddings, out_path):
	# split embedded corpus by language
	start_idx = 0
	for sidx, uidx in enumerate(ud_idcs):
		cur_language = ud.get_language_of_index(uidx)
		# check if next index is new language
		if (sidx == len(ud_idcs) - 1) or (cur_language != ud.get_language_of_index(ud_idcs[sidx+1])):
			# convert relevant embedding tensors to numpy
			num_embeddings = sidx - start_idx + 1
			cur_embeddings = np.zeros((num_embeddings, embeddings[start_idx].shape[-1])) # (num_embs, hidden_dim)
			for eidx, emb in enumerate(embeddings[start_idx:sidx+1]):
				cur_embeddings[eidx] = emb.cpu().numpy()
			# save tensors to disk as numpy arrays
			tensor_path = os.path.join(out_path, f'{cur_language.replace(" ", "_")}.npy')
			np.save(tensor_path, cur_embeddings)
			start_idx = sidx+1
			logging.info(f"Saved embeddings to '{tensor_path}' as numpy array.")


def export_map(sen_hashes, embeddings, out_path):
	hash_emb_map = {sen_hashes[sidx]: embeddings[sidx] for sidx in range(len(embeddings))}
	# save to disk
	out_file = os.path.join(out_path, 'map.pkl')
	with open(out_file, 'wb') as fp:
		pickle.dump(hash_emb_map, fp)
	logging.info(f"Saved 'sentence hash' -> 'embedding tensor' map to '{out_file}'.")


def export_txt(embeddings, vocab_embs, out_path):
	# convert list(Tensor) -> Tensor
	emb_tensor = torch.zeros((len(embeddings), embeddings[0].shape[0]))  # (num_words, emb_dim)
	for eidx, embedding in enumerate(embeddings):
		emb_tensor[eidx] = embedding

	# mean-pool embeddings of each word type (sorted by frequency)
	emb_strs = []
	for word, idcs in sorted(vocab_embs.items(), key=lambda el: len(el[1]), reverse=True):
		emb_anchor = torch.mean(emb_tensor[idcs, :], axis=0)
		emb_strs.append(word + ' ' + ' '.join([str(float(val)) for val in emb_anchor]) + '\n')
	logging.info(f"Mean pooled {len(embeddings)} token embeddings into {len(vocab_embs)} vocab anchor embeddings.")

	out_file = os.path.join(out_path, 'anchors.txt')
	with open(out_file, 'w', encoding='utf8') as fp:
		fp.write(f'{len(vocab_embs)} {emb_tensor.shape[1]}\n')  # first line has to be "num_vocab emb_dim"
		fp.writelines(emb_strs)

	logging.info(f"Saved anchor embeddings to '{out_file}'.")

#
# main embedding process
#


def main():
	args = parse_arguments()

	# check if output dir exists
	setup_output_directory(args.out_path)

	# setup logging
	setup_logging(os.path.join(args.out_path, 'embed.log'))

	# load data split definition (if supplied)
	ud_idcs, ud_filter = None, None
	if args.split:
		with open(args.split, 'rb') as fp:
			splits = pickle.load(fp)
		# create filter to load only relevant indices (train, dev)
		ud_idcs = set(splits['train']) | set(splits['dev']) | set(splits['test'])
		ud_filter = UniversalDependenciesIndexFilter(ud_idcs)
		ud_idcs = sorted(ud_idcs)
		logging.info(f"Loaded data splits {', '.join([f'{s}: {len(idcs)}' for s, idcs in splits.items()])}.")

	# load Universal Dependencies (relevant sentences only)
	ud = UniversalDependencies.from_directory(args.ud_path, ud_filter=ud_filter, verbose=True)
	ud_idcs = list(range(len(ud))) if ud_idcs is None else sorted(ud_idcs)
	logging.info(f"Loaded {ud} with {len(ud_idcs)} relevant sentences.")

	# load transformer model
	model_type = 'standard'
	if args.model.startswith('sentence/'):
		from sentence_transformers import SentenceTransformer
		model = SentenceTransformer(args.model.replace('sentence/', ''))
		tokenizer = model.tokenizer
		model_type = 'sentence'
		# check if pooling method is appropriate
		assert args.pooling != 'sw-mean', "[Error] Sentence encoders cannot be combined with the 'sw-mean' pooling method."
	else:
		from transformers import AutoTokenizer, AutoModel
		tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
		model = AutoModel.from_pretrained(args.model, return_dict=True)
		# check CUDA availability
		if torch.cuda.is_available(): model.to(torch.device('cuda'))
	model.eval()

	logging.info(f"Loaded {model_type}-type '{args.model}' {model.__class__.__name__} with {tokenizer.__class__.__name__}.")

	# main embedding loop
	embeddings = []  # (num_sents, emb_dim) or (vocab, emb_dim) for sw-mean
	sen_hashes = []  # (num_sents,) of MD5 hashed sentences
	vocab_embs = defaultdict(list)  # {word: sent_idcs} for sw-mean
	tkn_count, unk_count, max_len = 0, 0, 1
	cursor = 0
	while cursor < len(ud_idcs):
		# set up batch
		start_idx = cursor
		end_idx = min(start_idx + args.batch_size, len(ud_idcs))
		cursor = end_idx

		sentences = ud[ud_idcs[start_idx:end_idx]]
		batch = [s.to_words() for s in sentences]
		sen_hashes += [hashlib.md5(' '.join(s).encode('utf-8')).hexdigest() for s in batch]

		if tokenizer:
			# tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]]}
			tok_batch = tokenizer(
				batch, return_tensors='pt',
				is_split_into_words=True, return_special_tokens_mask=True, return_offsets_mapping=True,
				padding=True, truncation=True)
			# count tokens and UNK
			tkn_count += int(torch.sum(tok_batch['input_ids'] != tokenizer.pad_token_id))
			unk_count += int(torch.sum(tok_batch['input_ids'] == tokenizer.unk_token_id))
			# set maximum length
			cur_max_len = int(torch.max(torch.sum(tok_batch['attention_mask'], dim=-1)))
			max_len = cur_max_len if cur_max_len > max_len else max_len

		# no gradients required during inference
		with torch.no_grad():
			# embed batch (sentence-level)
			if model_type == 'sentence':
				# SentenceTransformer takes list[str] as input
				emb_sentences = model.encode(batch, convert_to_tensor=True) # (batch_size, hidden_dim)
				embeddings += [emb_sentences[sidx] for sidx in range(emb_sentences.shape[0])]
			# embed batch (token-level)
			else:
				# move input to GPU (if available)
				model_inputs = {
					k: tok_batch[k].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
					for k in ['input_ids', 'token_type_ids', 'attention_mask']
				}
				# perform embedding forward pass
				model_outputs = model(**model_inputs, output_hidden_states=True)
				emb_batch = model_outputs.hidden_states[args.model_layer] # (batch_size, max_len, hidden_dim)
				att_batch = tok_batch['attention_mask'] > 0 # create boolean mask (batch_size, max_len)

				# perform pooling over sentence tokens with specified strategy
				# mean pooling over tokens in each sentence
				if args.pooling == 'mean':
					# reduce (1, max_length, hidden_dim) -> (1, num_tokens, hidden_dim) -> (hidden_dim)
					embeddings = mean_pool_sentences(embeddings, emb_batch, att_batch)
				# get cls token from each sentence
				elif args.pooling == 'cls':
					# reduce (1, max_length, hidden_dim) -> (hidden_dim)
					embeddings = gather_sentence_cls(embeddings, emb_batch)
				# TODO get mean-pooled subword embeddings
				elif args.pooling == 'sw-mean':
					emb_words, att_words = reduce_pieces_to_words(tokenizer, batch, tok_batch, emb_batch)
					embeddings, vocab_embs = gather_subwords(embeddings, vocab_embs, batch, emb_words)
				# no reduction
				elif args.pooling == 'none':
					# (sen_len, hidden_dim)
					embeddings = gather_nonpad_embeddings(embeddings, emb_batch, att_batch)

		sys.stdout.write(f"\r[{(cursor*100)/len(ud_idcs):.2f}%] Embedding...")
		sys.stdout.flush()

	print("\r")
	if tkn_count: logging.info(f"{tokenizer.__class__.__name__} encoded corpus to {tkn_count} tokens with {unk_count} UNK tokens ({unk_count/tkn_count:.4f}).")
	logging.info(f"{model.__class__.__name__} embedded corpus with {len(embeddings)} items.")

	# export embeddings to numpy arrays
	if args.out_format == 'numpy':
		# TODO export with padding
		if args.pooling == 'none': raise NotImplementedError
		export_numpy(ud, ud_idcs, embeddings, args.out_path)

	# export embeddings to sent_hash->emb_map
	elif args.out_format == 'map':
		export_map(sen_hashes, embeddings, args.out_path)

	# export embeddings as 'token dim0 dim1 ... dimN\n'
	elif args.out_format == 'txt':
		export_txt(embeddings, vocab_embs, args.out_path)

	logging.info(
		f"Completed embedding of {len(ud_idcs)} sentences using '{args.model}' model (Layer {args.model_layer}) "
		f"and '{args.pooling}' pooling strategy."
	)


if __name__ == '__main__':
	main()