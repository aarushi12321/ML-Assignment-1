import datetime

def log_message(file, message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file.write(f'{timestamp} : {message}\n')
    return

def get_col_type(dataset):
    info_data = []
    for col in dataset.columns:
        null_val = dataset.shape[0] - dataset[col].notnull().sum()
        info_data.append([col, null_val, dataset[col].dtype])
    
    return info_data

def get_col_nunique(dataset):
    info_data = []
    for col in dataset.columns:
        info_data.append([col, dataset[col].nunique()])
    return info_data

def write_model_logs(args, file):
    log_message(file, f"scalar : {args.scalar}")
    log_message(file, f"train test split : {args.train_test_split_ratio}")
    log_message(file, f"random state : {args.random_seed}")
    log_message(file, f"epochs : {args.epochs}")
    log_message(file, f"learning rate : {args.learning_rate}")
    log_message(file, f"activation function : {args.activation_function}")
    log_message(file, f"loss function : {args.loss_function}")

    return