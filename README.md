Adapted from https://www.tensorflow.org/hub/tutorials/image_retraining

## Setup

You can build a docker image to use by using **make**:

```
make
```

Or just don't use docker at all:

```
pip install tensorflow tensorflow-hub
```

## Train

### Docker

```
docker run -v /your/train/images/:/data -v $(pwd)tf_files:/model/tf_files landmark /bin/bash /model/train.sh
```

You will get back trained models in your current folder (in tf_files/).


### No docker

```
./train.sh /your/train/images/
```


## Predict

### Docker

```
docker run -v /your/test/images/:/data -v $(pwd):/result landmark /bin/bash /model/predict.sh
```

You will get your `submission.csv` in your current folder.


### No docker

```
./predict.sh /your/test/images/ submission.csv
```
