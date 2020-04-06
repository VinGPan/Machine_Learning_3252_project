# Network Based Intrusion Detection System in IOT Devices

In this project we will be analysing a Kaggle Data set provided at 
https://www.kaggle.com/speedwall10/iot-device-network-logs

We will develop Machine Learning models which not only can **detect** 
attacks but also can **predict** the future attacks.

### Data:
Download Preprocessed_data.csv from https://www.kaggle.com/speedwall10/iot-device-network-logs
and copy it under data folder

### Third Party Dependencies
Download PyEEG library from the gitlink https://github.com/forrestbao/pyeeg
into third_party/pyeeg

Install PyEEG as per the PyEEG readme. 
```sh
$ cd pyeeg
$ python setup.py install
```

### Run the project note book
Run project.ipynb

### Make a new experiment
make a yml description of your experiment under experiments folder. 
Suggested to start with existing experiment and make required modification

```sh
$ python -m src.run_experiment experiments/<your_experiment_name>
```  