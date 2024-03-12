import argparse

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default= "data/Titanic-Dataset.csv")
    parser.add_argument('--analysis_file_path', default="logs/analysis_dataset.txt")
    parser.add_argument('--preprocessing_file_path', default="logs/preprocessing_dataset.txt")
    parser.add_argument('--train_test_file_path', default="logs/train_test_split_dataset.txt")
    parser.add_argument('--train_logs_folder', default='logs/train_logs')
    parser.add_argument('--catagorical_thresh', default=10)
    parser.add_argument('--drop_cols', default=None, nargs="*")
    parser.add_argument('--scalar', default='MinMaxScalar')
    parser.add_argument('--train_test_split_ratio', default=0.25)
    parser.add_argument('--random_seed', default=15)
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--activation_function', default='sigmoid')
    parser.add_argument('--loss_function', default='mean_squared_error')
    parser.add_argument('--verbose', default=1)
    
    args = parser.parse_args()

    return args