import os
import pandas as pd
from tabulate import tabulate
import numpy as np

from utils import *
from parse_args import make_args
from preprocessing_utils.analysis_utils import get_numerical_catagorical_columns, get_drop_cols, get_isna_cols
from preprocessing_utils.analysis_utils import check_na, scaleDataset
from preprocessing_utils.dataset_specific_preprocessing import fill_na_age
from sklearn.model_selection import train_test_split
from Model.NN_model import *
from Model.NN_layer import *
from Model.NN_loss_functions import *

class NNPipeline():
    def __init__(self, dataset, args):
        # general
        self.data = dataset
        self.args = args

        # preprocessing
        self.catagorical_data = None
        self.numerical_data = None 
        if args.drop_cols == None:
            self.drop_data = None
        else:
            self.drop_data = args.drop_cols
        self.isna_data = None
        self.X = None
        self.Y = None

        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

        # training 
        self.train_accuracy_per_epoch, self.train_error_per_epoch = None, None
        self.model = None
    
    def analysis(self):
        if os.path.exists(self.args.analysis_file_path):
             os.remove(self.args.analysis_file_path)
        file = open(self.args.analysis_file_path, 'a')

        # data.shape
        file.write(f"Number of data samples in our dataset \t {self.data.shape[0]}\n")
        file.write(f"Number of features in our dataset     \t {self.data.shape[1]}\n")

        # data.head
        file.write(f"\nThe following data shows five samples of data : \n")
        table = tabulate(self.data.head(), headers='keys', tablefmt='simple_grid')
        file.write(table)
        file.write("\n")

        # data.info
        file.write("\nThe following information is about the column types of the dataset.\n")
        headers = ["Column Name", "Null values", "Column Type"]
        info_data = get_col_type(self.data)
        info_table = tabulate(info_data, headers=headers, tablefmt="simple_grid")
        file.write(info_table)
        file.write("\n")

        # data.nunique
        file.write("\nThe following information is about the unique values in the dataset.\n")
        headers = ["Column Name", "Unique Values"]
        nunique_data = get_col_nunique(self.data)
        nunique_table = tabulate(nunique_data, headers=headers, tablefmt="simple_grid")
        file.write(nunique_table)
        file.write("\n")

        # getting types of columns to make preprocessing easier
        catagorical_cols, numerical_cols, id_cols, misc_cols = get_numerical_catagorical_columns(self.data, self.args.catagorical_thresh)
        file.write("\nInformation regarding the type of data in each col - numeric /catagorical\n")
        file.write(f"Catagorical   \t {catagorical_cols}\n")
        file.write(f"Numerical     \t {numerical_cols}\n")
        file.write(f"Id            \t {id_cols}\n")
        file.write(f"Miscelleneous \t {misc_cols}\n")

        assert(len(catagorical_cols + numerical_cols + id_cols + misc_cols) == len(self.data.columns))

        drop_cols = get_drop_cols(self.data, catagorical_cols, id_cols, misc_cols)
        file.write(f"\nColumn suggestions to be dropped : {drop_cols}\n")

        isna_cols = get_isna_cols(self.data, drop_cols)
        file.write(f"\nColumn suggestions to be updated by filling NULL values : {isna_cols}\n")

        self.catagorical_data = catagorical_cols
        self.numerical_data = numerical_cols
        self.drop_data = drop_cols
        self.isna_data = isna_cols

        return 
    
    def data_preprocessing(self):
        if os.path.exists(self.args.preprocessing_file_path):
            os.remove(self.args.preprocessing_file_path)
        file = open(self.args.preprocessing_file_path, 'a')

        # drop the columns 
        self.data.drop(self.drop_data, inplace = True, axis=1)
        log_message(file, f"Columns that have been deleted : {self.drop_data}")
        log_message(file, f"After this step, the number of columns in the data are : {self.data.shape[1]}")
        log_message(file, f"After this step, the columns in the data are : {list(self.data.columns)}")

        # fill the NULL values for the null columns post drop
        # this process is specific to the dataset and should be changed in case there is a change in the dataset.
        mean_ages = fill_na_age(self.data)
        headers = ["Pclass", "Sex", "Mean Age"]
        age_table = tabulate(mean_ages, headers=headers, tablefmt="simple_grid")
        log_message(file, "The division to calculate the age")
        log_message(file, f"\n{age_table}")

        self.data = pd.merge(self.data, mean_ages, on=['Pclass', 'Sex'], how='left')
        self.data['Age'] = self.data['Age'].fillna(self.data['Mean_Age'])
        self.data.drop('Mean_Age', axis=1, inplace=True)
        assert(self.data['Age'].isnull().sum() == 0)
        log_message(file, "Age null values filled.")
        
        self.data['Embarked'] = self.data['Embarked'].fillna(self.data['Embarked'].mode()[0])
        assert(self.data['Embarked'].isnull().sum() == 0)
        log_message(file, "Embarked null values filled.")

        check_na(self.data)
        log_message(file, "All columns have been checked and cleared for NULL values.")

        log_message(file,f"\nThe following data shows five samples of data : \n")
        table = tabulate(self.data.head(), headers='keys', tablefmt='simple_grid')
        log_message(file,f"\n{table}")

        # Convert the non-numerical data to numerical data
        sex_one_hot_encoded = pd.get_dummies(self.data['Sex'],drop_first=True)
        embarked_one_hot_encoded = pd.get_dummies(self.data['Embarked'],drop_first=True, prefix="embarked")
        self.data.drop(['Sex' , 'Embarked'], axis=1, inplace=True)
        self.data = pd.concat([self.data,sex_one_hot_encoded,embarked_one_hot_encoded],axis=1)

        log_message(file, "One hot encoding has been completed.")
        log_message(file,f"The following data shows five samples of data after the step: ")
        table = tabulate(self.data.head(), headers='keys', tablefmt='simple_grid')
        log_message(file,f"\n{table}")

        # Split the data into X and Y and then scale the X features
        self.X = self.data.drop(['Survived'], axis=1)
        log_message(file,f"The following data shows five samples of X: ")
        table = tabulate(self.X.head(), headers='keys', tablefmt='simple_grid')
        log_message(file,f"\n{table}")

        self.Y = self.data['Survived']
        log_message(file,f"The following data shows five samples of Y: ")
        table = tabulate(self.Y[:5].to_frame(), headers=["Survived"], tablefmt='simple_grid')
        log_message(file,f"\n{table}")

        # scale the X features using Min - Max scalar
        self.X = scaleDataset(self.X)
        log_message(file,f"The following data shows five samples of X after scaling: ")
        table = tabulate(self.X.head(), headers='keys', tablefmt='simple_grid')
        log_message(file,f"\n{table}")
        log_message(file, "Preprocessing has been completed. Your data is ready to be split into training and evaluation data.")

        return
    
    def data_visualization():
        pass

    def train_test_split(self):
        if os.path.exists(self.args.train_test_file_path):
            os.remove(self.args.train_test_file_path)
        file = open(self.args.train_test_file_path, 'a')

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.args.train_test_split_ratio, random_state=self.args.random_seed)
        log_message(file, "Train Test split gives the following Dimensions for the data:")
        log_message(file, f"X Train Dimensions \t{self.X_train.shape}")
        log_message(file, f"Y Train Dimensions \t{self.y_train.shape}")
        log_message(file, f"X Test Dimensions  \t{self.X_test.shape}")
        log_message(file, f"Y Test Dimensions  \t{self.y_test.shape}")

    def train(self):
        path = self.args.train_logs_folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        file = open(os.path.join(path, 'train_model_logs_file_'+ str(len(files)) + '.txt'), 'a')


        self.model = NNModel(self.args)
        self.model.add(NNLayer(self.args, self.X_train.shape[1], 8))
        self.model.add(ActivationLayer(self.args))
        self.model.add(NNLayer(self.args, 8, 4))
        self.model.add(ActivationLayer(self.args))
        self.model.add(NNLayer(self.args,4,1))

        self.train_error_per_epoch, self.train_accuracy_per_epoch = self.model.fit(self.X_train.to_numpy(), self.y_train.to_numpy(), args.verbose)
        # logging args
        write_model_logs(self.args, file)
        # logging accuracy and error
        file.write("\n")
        log_message(file,"The accuracy and error with respect to the epochs")
        for i in range(len(self.train_error_per_epoch)):
            file.write(f"Epoch : {i+1} \t Error : {np.round(self.train_error_per_epoch[i], 6)} \t Accuracy : {np.round(self.train_accuracy_per_epoch[i], 6)}\n")

    def evaluate(self):
        path = self.args.train_logs_folder
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        file = open(os.path.join(path, 'train_model_logs_file_'+ str(len(files)-1) + '.txt'), 'a')
        y_pred = self.model.predict(self.X_test.to_numpy())
        test_accuracy = accuracy(y_pred, self.y_test.to_numpy())
        file.write("\n")
        print(f"Test accuracy is {test_accuracy}\n")
        log_message(file, "The predicted values vs actual values are :")
        for i in range(len(y_pred)):
            file.write(f"Predicted : {y_pred[i]} \t Actual : {self.y_test.iloc[i]}\n")
        file.write("\n")
        log_message(file, "The test accuracy is {}".format(test_accuracy))

    def predict():
        pass

if __name__ == "__main__":
    args = make_args()
    if os.path.exists("logs/main_logs.txt"):
         os.remove("logs/main_logs.txt")
    with open("logs/main_logs.txt", 'a') as f:
        log_message(f, "Let us start the training of Neural Networks.")
        try:
            dataset = pd.read_csv(args.dataset_path)
            log_message(f, "Dataset successfully loaded.")
        except:
                log_message(f, "File not found. Check the path.")
        pipeline = NNPipeline(dataset, args)
        log_message(f, "Starting the analysis of the dataset. Based on this analysis select the preprocessing terms and features.")
        pipeline.analysis()
        log_message(f, "Analysis of the dataset has been successfully completed.")
        log_message(f,"Starting the data preprocessing.")
        pipeline.data_preprocessing()
        log_message(f,"Data Preprocessing has successfully been completed.")
        log_message(f, "Starting train test split.")
        pipeline.train_test_split()
        log_message(f, "Train test split has successfully been completed.")
        log_message(f, "The Training of the model is begining.")
        pipeline.train()
        log_message(f, "The training has been completed successfully.")
        log_message(f, "The Evaluation of the model is begining.")
        pipeline.evaluate()
        log_message(f, "The evaluation has been completed successfully.")