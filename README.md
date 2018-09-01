Adapted from https://www.tensorflow.org/hub/tutorials/image_retraining

## Build

```
make
```

## Train

```
docker run -v /your/train/images/:/data -v $(pwd):/model/tf_files landmark /bin/bash /model/train.sh
```

You will get back trained models in your current folder.

## Predict

```
docker run -v /your/test/images/:/data -v $(pwd):/result landmark /bin/bash /model/predict.sh
```

You will get your `submission.csv` in your current folder.
