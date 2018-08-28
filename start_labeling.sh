#!/bin/bash
python label_image.py \
    --graph=tf_files/retrained_graph.pb \
    --labels=tf_files/retrained_labels.txt \
    --input_layer=Placeholder \
    --output_layer=final_result \
    --image=$1
