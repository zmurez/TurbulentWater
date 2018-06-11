#!/bin/bash

echo "Hello World!"
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name both_L1VGGAdv
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name color_L1VGGAdv --no-warp-net
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name warp_L1VGG --no-color-net
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name warp_L1 --no-color-net --weight-Y-VGG 0
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name color_L1VGG --no-warp-net --weight-Z-Adv 0
CUDA_VISIBLE_DEVICES=1 python main.py --exp-name warp_L1VGG_synth --no-color-net --synth-data
python concat_results.py
