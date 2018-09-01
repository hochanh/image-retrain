Adapted from https://www.tensorflow.org/hub/tutorials/image_retraining

## Build

```
make
```

## Predict

```
docker run -v /your/images/:/data -v $(pwd):/result landmark /bin/bash /model/predict.sh
```

You will get your `submission.csv` in your current folder.
