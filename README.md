# I'm Something of a Painter Myself - Kaggle [Competition link](https://www.kaggle.com/c/gan-getting-started/)

This project is a part of EE541 - A Computational Introduction to Deep Learning. The goal of this project is to generate Monet styled paintings given a photograph as an input. 

## Setup (Google Cloud Platform)
We trained the model on GCP with a Tesla V100 GPU with 16GB RAM.

### Installation instructions
* Setup Nvidia GPU drivers is not setup already
* Install conda environment using the ```ee541.yml``` file
    
    ```conda env create -f ee514.yml```
* Download the dataset from [here](https://www.kaggle.com/suyashdamle/cyclegan?select=monet2photo)
* To start training the model run the following command from the root directory

    ```python cyclegan/train.py```

## Code Structure
* The complete code for the project is in ```cyclegan``` directory
* ```testNotebook.ipynb``` contains the code to test the model and also represents the significance of different loss functions that we use in the process of training

## Architecture
The architecture for this model is inspired from the original paper by Zhu et al. You can find the paper [here](https://arxiv.org/abs/1703.10593)
