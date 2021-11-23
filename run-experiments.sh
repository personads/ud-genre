#!/bin/bash

#
# Script for running all instance genre prediction experiments
#

seeds=( 41 42 43 )

# Compute embeddings for all sentences in UD
python embed.py ud28/treebanks/ bert-base-multilingual-cased ud28/emb/mbert/

# Run experiments with all random seeds
for seed in "${seeds[@]}"; do

# Run LDA clustering (global)
python cluster/lda.py ud28/treebanks/ exp/lda/all-rs$seed -s ud28/splits/102-915-204.pkl -sn test -cl all -vu char -vn 3-6 -rs $seed
python eval/genre.py ud28/treebanks/ exp/lda/all-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test

# Run LDA clustering (treebank-level)
python cluster/lda.py ud28/treebanks/ exp/lda/tb-rs$seed -s ud28/splits/102-915-204.pkl -sn test -cl treebank -vu char -vn 3-6 -rs $seed
python cluster/label.py ud28/treebanks/ ud28/emb/mbert/ exp/lda/tb-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test
mkdir exp/lda/tb-rs$seed/labels; mv exp/lda/tb-rs$seed/All-labeled.pkl exp/lda/tb-rs$seed/labels/All.pkl
python eval/genre.py ud28/treebanks/ exp/lda/tb-rs$seed/labels/ -s ud28/splits/102-915-204.pkl -sn test
python eval/sentence-genre.py ud28/treebanks/ ud28/meta.json ud28/tb-genres.json exp/lda/tb-rs$seed/labels/ -s ud28/splits/102-915-204.pkl -sn test

# Run GMM clustering (global)
python cluster/gmm.py ud28/treebanks/ ud28/emb/mbert/ exp/gmm/all-mbert-rs$seed -s ud28/splits/102-915-204.pkl -sn test -cl all -rs $seed
python eval/genre.py ud28/treebanks/ exp/gmm/all-mbert-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test

# Run GMM clustering (treebank-level)
python cluster/gmm.py ud28/treebanks/ ud28/emb/mbert/ exp/gmm/tb-mbert-rs$seed -s ud28/splits/102-915-204.pkl -sn test -cl treebank -rs $seed
python cluster/label.py ud28/treebanks/ ud28/emb/mbert/ exp/gmm/tb-mbert-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test
mkdir exp/lda/tb-mbert-rs$seed/labels; mv exp/lda/tb-mbert-rs$seed/All-labeled.pkl exp/lda/tb-mbert-rs$seed/labels/All.pkl
python eval/genre.py ud28/treebanks/ exp/gmm/tb-mbert-rs$seed/labels/ -s ud28/splits/102-915-204.pkl -sn test
python eval/sentence-genre.py ud28/treebanks/ ud28/meta.json ud28/tb-genres.json exp/gmm/tb-mbert-rs$seed/labels/ -s ud28/splits/102-915-204.pkl -sn test

# Train Approximate Classifier
python classify/classify.py ud28/treebanks/ bert-base-multilingual-cased exp/class/mbert-rs$seed -s ud28/splits/71-30-0.pkl -e 30 -rs $seed
python classify/predict.py ud28/treebanks bert-base-multilingual-cased exp/class/mbert-rs$seed/
python eval/genre.py ud28/treebanks/ exp/class/mbert-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test
python eval/sentence-genre.py ud28/treebanks/ ud28/meta.json ud28/tb-genres.json exp/class/mbert-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test

# Train Boot
python classify/boot.py ud28/treebanks/ bert-base-multilingual-cased exp/boot/mbert-rs$seed -s ud28/splits/71-30-0.pkl -e 30 -rs $seed
python classify/predict.py ud28/treebanks bert-base-multilingual-cased exp/boot/mbert-rs$seed/
python eval/genre.py ud28/treebanks/ exp/boot/mbert-rs$seed/ -s ud28/splits/102-915-204.pkl -sn test
python eval/sentence-genre.py ud28/treebanks/ ud28/meta.json ud28/tb-genres.json exp/boot/mbert-rs$seed/closed-domain/ -s ud28/splits/102-915-204.pkl -sn test

done

# Run Zero
python classify/zero.py ud28/treebanks/ bert-base-multilingual-cased ud28/emb/mbert/ exp/zero/mbert
python eval/genre.py ud28/treebanks/ exp/zero/mbert/ -s ud28/splits/102-915-204.pkl -sn test
python eval/sentence-genre.py ud28/treebanks/ ud28/meta.json ud28/tb-genres.json exp/zero/mbert/ -s ud28/splits/102-915-204.pkl -sn test
