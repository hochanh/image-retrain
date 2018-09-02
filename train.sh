#!/bin/bash
python retrain.py \
  --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/feature_vector/1 \
  --how_many_training_steps=10000 \
  --bottleneck_dir=/tmp/bottlenecks \
  --summaries_dir=/tmp/training_summaries/mobilenet_v1_100_224 \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --model_dir=tf_files/models/ \
  --image_dir=${1-/data/}
