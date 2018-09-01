#!/bin/bash
python label_image.py \
    --graph=tf_files/retrained_graph.pb \
    --labels=tf_files/retrained_labels.txt \
    --input_layer=Placeholder \
    --input_height=224 \
    --input_width=224 \
    --output_layer=final_result \
    --image_dir=$1
