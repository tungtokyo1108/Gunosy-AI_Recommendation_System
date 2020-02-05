[![Build Status](https://travis-ci.com/tungtokyo1108/Gunosy-Data_Analysis.svg?token=yuqS1QiMpPCTMctfsjVi&branch=master)](https://travis-ci.com/tungtokyo1108/Gunosy-Data_Analysis)

# Gunosy data analysis project: News classification 

## Pre-requisites:

* `docker` and `docker-compose`: The [Docker](https://docs.docker.com/install/) and [Docker-compose](https://docs.docker.com/compose/install/) command-line interface. Follow these installation instructions for your system.
* The minimum recommended resources for this model is 16GB RAM + 8 CPUs (Ubuntu). 

## Research 

- Including the data collection, data pre-processing, feature engineering, model training, model evaluation. In order to read the specific methods, please go to [here](https://github.com/tungtokyo1108/Gunosy-Data_Analysis/blob/master/research/README.md) 
- Solution for Gunosy's second requirement for the improvement of document classification performance. 

## Run Locally

1. [Build the Model](#1-build-the-model)
2. [Deploy the Model](#2-deploy-the-model)
3. [Use the Model](#3-use-the-model)
4. [Cleanup](#4-cleanup)

### 1. Build the Model

Clone this repository locally. In a terminal, run the following command:

```
$ git clone https://github.com/tungtokyo1108/Gunosy-Data_Analysis.git
```

Change directory into the repository base folder:

```
$ cd Gunosy-Data_Analysis
```

To build locally, run:

```
$ sudo docker-compose build
```

All required model assets will be downloaded during the build process. 

### 2. Deploy the Model

To run:

```
$ sudo docker-compose up
```

For the first time, the training data is not available, you have to wait to generate the database. For next time, when the training data is available, the training model is started like images below. 

![Swagger Doc Screenshot](docs/First_time_training.png)

![Swagger Doc Screenshot](docs/Next_time_train.png)

If you want to run separately data generating and model training, to run below: 

```
$ sudo docker-compose run -p 8000 --rm web python Gunosy_data_generate.py
```

```
$ sudo docker-compose run -p 8000 --rm web python Gunosy_model_training.py
```

If you want to make the new data base, you can remove the `News_dataset.pickle` and run 

```
$ sudo docker-compose up
```
or 
```
$ sudo docker-compose run -p 8000 --rm web python Gunosy_data_generate.py
```

### 3. Use the Model

The API server automatically generates an page. Go to `http://0.0.0.0:8000/api/v1/` to load it. From there you can explore the API and also create test requests.

![Swagger Doc Screenshot](docs/API_Root.png)

![Swagger Doc Screenshot](docs/List_Algorithm.png)

Get predictions from the API, please go to `http://0.0.0.0:8000/api/v1/Gunosy_classifier/prediction`. Providing the links of `https://gunosy.com/` into the "Content" box (please put link inside "" like image below), click the "POST" button and wait a few seconds to take the result of prediction. You can see views like images beloew. 

![Swagger Doc Screenshot](docs/Sport_news.png)

![Swagger Doc Screenshot](docs/Sport_predict.png)

The results of predictions provide the two groups with the highest probability

### 4. Cleanup

To stop the Docker container, type `CTRL` + `C` in your terminal.