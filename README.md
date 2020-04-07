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


### Experiments
An experiment is one invocation of model building process. Description of the 
experiment is provided via a yml file. In this project we have 5 experiments.
The yml containing experiment description and can be found under experiments folder.
This yml file provides 
1) details of how to preproces the data
2) details of PyEEG features to be computed (optional)
3) details of ML models to be build
 
To run an experiment
```sh
$ python -m src.run_experiment experiments/<experiment_yml_filename>
```  