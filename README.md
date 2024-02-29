#  Zomato Rating Prediction

### End-to-End Machine Learning Project

This project demonstrates a complete machine learning pipeline that predicts customer ratings for restaurants on Zomato, a popular restaurant discovery platform. 

##  Table of Contents

* [General info](#general-info)
* [Data Source](#data-source)
* [Problem Statement](#problem-statement)
* [Demo Photos](#demo-photos)
* [Library Used](#Library-Used)
* [Project Structure](#project-structure)
* [Run Locally](#run-locally)
* [Project Creators](#project-creators)

##  General Info
This project aims to empower Zomato to enhance customer experience by predicting their likelihood of rating a restaurant. By leveraging this information, Zomato can personalize recommendations and recommendations. 

##  Data Source
Dataset: Simulated Zomato rating data
Note: Due to potential privacy concerns, we avoid using real Zomato data publicly. You can find similar datasets on Kaggle for educational purposes.

##  Problem Statement
This project seeks to answer the following questions:
Predict customer ratings: Can we build a model to predict how a customer might rate a restaurant based on their past behavior and other restaurant characteristics?
Identify key factors: Which factors most significantly influence customer ratings?
Targeted recommendations: Which customer segments are more likely to appreciate specific restaurants?

##  Demo Photos
<img width="1006" alt="Rating" src="https://github.com/zahidshaikh10101/Zomato-Rating-Prediction/blob/main/templates/Screenshot/init.PNG">

<img width="1006" alt="Rating" src="https://github.com/zahidshaikh10101/Zomato-Rating-Prediction/blob/main/templates/Screenshot/App.PNG">
 Our Website will be look like and when we hit the submit button prediction will happen and output will give rating on the parameters given by customers.

## ️ Libraries Used
```
pandas - For data analysis and manipulation
numpy - For numerical computations and array operations
seaborn - For data visualization
matplotlib - For creating various types of plots
scikit-learn - For machine learning algorithms and tools
Flask - For creating web applications (optional for prediction interface)
dill - For serialization (optional for deployment)
```

## ⚙️ Project Structure

### There are structure used for different-different work:
 * ```setup.py```This contains all details about the Project.
 * ```requirements.txt``` Contains the all the libraries used in the project.
 * ```logger.py``` is responsible for the log all the information whatever is happening in the project at which perticular time or file.
 * ```exception.py``` is responsible for the give the Customexception when an error in any file, So it give the file_name,Lineno and error also.
 * ```.gitignore``` will add all the files which we don't want to push on the github.
 * ```readme.md``` contain general informtion about the project steps and requiremnts for further explaination.
 * ```data```contain the dataset.
 * ```src``` contain many subfolder. we need to give a ```__init__.py``` file in each directory i.e. we can use each file as a module.
 * ```src/data_ingestion.py``` responsible for the data ingestion from many different-different source like  ***kaggle*** ,***mongodb*** or ***MySQL*** etc. it split the data into train and test and store them in a perticular ```Artifacts``` folder.
 * ```src/data_transformation.py```responsible for the transform the categorical values into vectors. Also used in Scaling and Handle the Missing values and return a preprocessor which transform the data for the ***Machine Learning Models***.
 * ```src/Model_trainer.py``` is responsible for the model training and Hyperparameter tuning it return a Model Pickle file which is train on the data and used for the further Prediction.
 * ```utils.py``` is used for creating and storing the common function which are used whole through out the Project.
 * ```app.py``` is web app file which interact with user and take the input for new datapoints from the user and show the output by using the pre-trained model.
    
## ‍ Run Locally

Prerequisites:

* Git
* Python 3.x (with NumPy, pandas, etc.)
* Anaconda or miniconda installed on your system

Steps:

* Clone the repository: git clone https://github.com/zahidshaikh10101/Zomato-Rating-Prediction/
* Create a virtual environment: python -m venv venv
* Activate the environment: source venv/bin/activate (Windows: venv\Scripts\activate.bat)
* Install dependencies: pip install -r requirements.txt
* Run the project: python app.py (if using a web interface) or run scripts individually
  
##  Project Creators

Made with ❤️ <br>
[@Zahid Salim Shaikh](https://www.linkedin.com/in/zahid-shaikh-7z8s6s/)
