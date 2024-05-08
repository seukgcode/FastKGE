#! /bin/bash
export LANG=zh_CN.UTF-8


python main.py -model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 2 -dataset FB_CKGE -learning_rate 1e-1 -using_various_ranks True
python main.py -model_name LoraKGE_Layers -ent_r 150 -rel_r 20 -num_ent_layers 2 -num_rel_layers 1 -gpu 2 -dataset WN_CKGE -learning_rate 1e-1 -using_various_ranks True