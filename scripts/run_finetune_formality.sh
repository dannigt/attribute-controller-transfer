#!/bin/bash

GPU=0,1,2,3
attr="formalonly"
path_2_data=# attribute-specific data, e.g., $HOME/projects/classifier_guidance/data/CoCoA-MT/prepro/bin/$attr
PRETRAINED_MODEL=#path to pretrained checkpoint e.g., $HOME/projects/nllb/model/nllb200densedst600mcheckpoint

save_dir=# e.g., $HOME/projects/classifier_guidance/model/formality.de-es-fr-hi-it.fullfinetune.$attr
mkdir -p $save_dir

lr=0.0001

MODEL_FOLDER=$HOME/projects/nllb/model
lang_list=$MODEL_FOLDER/"langs.txt"

lang_pairs="eng_Latn-deu_Latn,eng_Latn-spa_Latn,eng_Latn-fra_Latn,eng_Latn-hin_Deva,eng_Latn-ita_Latn"
valid_lang_pairs="eng_Latn-deu_Latn,eng_Latn-spa_Latn,eng_Latn-fra_Latn,eng_Latn-hin_Deva,eng_Latn-ita_Latn"

batch_size=4096

CUDA_VISIBLE_DEVICES=$GPU fairseq-train $path_2_data \
    --save-dir $save_dir \
    --arch transformer \
    --encoder-normalize-before --decoder-normalize-before \
    --encoder-layers 12 --decoder-layers 12 \
    --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --encoder-embed-dim 1024 --decoder-embed-dim 1024 \
    --encoder-ffn-embed-dim 4096 --decoder-ffn-embed-dim 4096 \
    --task translation_multi_simple_epoch \
    --lang-pairs $lang_pairs \
    --valid-lang-pairs $valid_lang_pairs \
    --langs $lang_list \
    --share-all-embeddings \
    --add-data-source-prefix-tags \
    --finetune-from-model $PRETRAINED_MODEL \
    --max-source-positions 512 --max-target-positions 512 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt \
    --lr $lr \
    --warmup-init-lr 1e-07 \
    --stop-min-lr 1e-09 \
    --warmup-updates 20 \
    --max-update 60 \
    --max-tokens $batch_size \
    --update-freq 1 \
    --seed 1 \
    --no-epoch-checkpoints \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --encoder-langtok "src" \
    --decoder-langtok \
    --validate-interval-updates 1000 \
    --log-interval 100 \
    --save-interval-updates 9999 \
    --left-pad-source False \
    --keep-last-epochs 1 \
    --keep-interval-updates 1 \
    --train-with-epoch-remainder-batch \
    --fp16 2>&1 | tee -a $save_dir/train.log