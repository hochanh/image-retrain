FROM tensorflow/tensorflow:1.10.1

RUN mkdir /model
WORKDIR /model
ADD . /model/

RUN pip install tensorflow-hub

VOLUME ["/data", "/result"]

CMD ["/model/predict.sh"]
