import datetime
import matplotlib.pyplot as plt

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
    log_message(file, f"Optimizer : {args.optimizer}")

    return

def plot(train_error_per_epoch, train_accuracy_per_epoch):
    # Example data
    epochs = range(1, len(train_error_per_epoch) + 1)

    # Plot training error per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_error_per_epoch, label='Training Error')
    plt.plot(epochs, train_accuracy_per_epoch, label='Training Accuracy')
    plt.title('Training Error per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('logs/train_logs/Plot-train.png')
    plt.show()


