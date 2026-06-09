## PSV project to predict likes on Instagram Post

This is a project where I have two different models each one of them has a main function on the project. 
- EfficientNet Model --> The prediction that this model does, is a multilabel image prediction. There are 5 different type of categories in which the model will find different characteristics while training and categorize based on that. 
    Categories : Player, Action, Celebration, Graphic, Crowd
- Random Forest Model --> The predicition this model does is a regression problem. The target variable is the likes each instagram post could have based on different features. Such as: the named before categories, caption and timestamp. 

## Getting Started

First, clone the repository and install the dependencies: 

git clone https://github.com/nahiaortiz/PSV-project.git


## Run the Project

Run the notebook, to: 
+ Load the dataset
+ Train the both model ( it will save them everytime you run the notebook)
+ Evaluate performance
+ Check confusion matrix

## Run Prototype

First make sure you have your virtual machine created and you have all the dependencies installed. Afterwards, activate your virtual machine. Then you need to write the following lines in your terminal, this will get you in the right folder and make the html start on localhost. 

cd app 
uvicorn main:app

To finish just open your google chrome and type https://127.0.0.1:8000/predict

And everything will be set up for you to try the models!!

## Requirements

Make sure you have the following installed:

os, pandas, numpy, sklearn, joblib, re, matplotlib, tensorflow, fastapi, aiofiles


## Learn more 

To understand the technologies used in this project, you can explore:

+ PyTorch documentation
+ Transfer learning techniques
+ Image classification with CNNs

## Project Structure
data/
 ├── multilabel_dataset.csv/
 ├── (the rest of them are trials)/

nahiavenv/ (virtual machine)
 ├── Include/
 ├── Lib/
 ├── Scripts/

raw_images/ (images)
 ├── DMfBK5xMG_i_0.jpg
 ├── DMfBK5xMG_i_1.jpg
 ├── DMfBK5xMG_i_2.jpg

 notebooks/ (the notebooks of the models)
 ├── Efficient.ipynb/ (trial)
 ├── MobileNet2.ipynb/ 
 ├── newmodel_efficient.ipynb/
 ├── random-forest.ipynb/

 models/  (the save models for the prototype)
 ├── efficient-model.keras/
 ├── random_forest_likes.kpl/

  app/ (the pipeline for the prototype)
 ├── _pycache_/
 ├── static/
 ├── main.py/
 ├── predict.py/

 README.md


## Future Improvements
+ Improve dataset quality and balance
+ Add more categories such as : different players, timelines of inteviews ...
+ Improve performance on similar-looking pictures 
+ Extend the system to recognition of players 
+ Distinguish between matches and training