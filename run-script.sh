#!/bin/bash

learning_rates=(0.01 0.001 0.0001)
epochs=(10 100 1000)
activation_functions=('sigmoid' 'tanh' 'relu')
optimizers=(None 'SGDMomentum' 'AdaGrad' 'Adam')

for lr in "${learning_rates[@]}"; do
  for epoch in "${epochs[@]}"; do
      for activation_function in "${activation_functions[@]}"; do
        for optimizer in "${optimizers[@]}"; do
          echo "Running with learning rate $lr   epochs $epoch   activation function $activation_function   optimizer $optimizer"
          python3 run.py \
            --dataset_path "data/Titanic-Dataset.csv" \
            --analysis_file_path "logs/analysis_dataset.txt" \
            --preprocessing_file_path "logs/preprocessing_dataset.txt" \
            --train_test_file_path "logs/train_test_split_dataset.txt" \
            --train_logs_folder "logs/train_logs" \
            --catagorical_thresh 10 \
            --scalar "MinMaxScalar" \
            --train_test_split_ratio 0.25 \
            --random_seed 15 \
            --epochs $epoch \
            --learning_rate $lr \
            --activation_function $activation_function \
            --loss_function "mean_squared_error" \
            --verbose 0 \
            --optimizer $optimizer \
            --momentum_lr 0.01 \
            --momentum 0.9 \
            --beta1 0.9 \
            --beta2 0.999 \
            --make_plot 0 
        done
    done
  done
done
