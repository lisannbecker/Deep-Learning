################################################################################
# MIT License
#
# Copyright (c) 2023 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2023
# Date Created: 2023-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm

from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

import torch
import matplotlib.pyplot as plt



def confusion_matrix(predictions, targets):
    """
    Computes the confusion matrix, i.e. the number of true positives, false positives, true negatives and false negatives.

    Args:
      predictions: 2D float array of size [batch_size, n_classes], predictions of the model (logits)
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      confusion_matrix: confusion matrix per class, 2D float array of size [n_classes, n_classes]
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    n_classes = 10

    # Initialize confusion matrix
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)

    # Update the confusion matrix based on predictions and targets
    for pred, target in zip(predictions, targets):
        conf_mat[target, pred] += 1

    #######################
    # END OF YOUR CODE    #
    #######################
    return conf_mat


def confusion_matrix_to_metrics(confusion_matrix, beta=1.):
    """
    Converts a confusion matrix to accuracy, precision, recall and f1 scores.
    Args:
        confusion_matrix: 2D float array of size [n_classes, n_classes], the confusion matrix to convert
    Returns: a dictionary with the following keys:
        accuracy: scalar float, the accuracy of the confusion matrix
        precision: 1D float array of size [n_classes], the precision for each class
        recall: 1D float array of size [n_classes], the recall for each clas
        f1_beta: 1D float array of size [n_classes], the f1_beta scores for each class
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    precision_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall_per_class = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    f1_beta = (1+(beta**2))*precision_per_class*recall_per_class/(((beta**2)*precision_per_class) + recall_per_class)
    
    metrics = accuracy, precision_per_class, recall_per_class, f1_beta
    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics


def evaluate_model(model, data_loader, num_classes=10):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
        metrics: A dictionary calculated using the conversion of the confusion matrix to metrics.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset,
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    predictions, targets = [], []

    for data_inputs, data_labels in data_loader: 
        data_inputs = data_inputs.reshape(data_inputs.shape[0], -1)

        preds = model.forward(data_inputs)

        predicted = np.argmax(preds, axis=1) 

        for p in predicted.tolist():
            predictions.append(p) 
        for l in data_labels.tolist():
            targets.append(l)
            
    conf_mat = confusion_matrix(predictions, targets)
    metrics_confusion = confusion_matrix_to_metrics(conf_mat, beta=1.)
    accuracy, precision_per_class, recall_per_class, f1_beta = metrics_confusion
    
    metrics = {"Model": model, "Confusion Matrix": conf_mat, "Accuracy": accuracy, "Precision per Class": precision_per_class, 
               "Recall per Class": recall_per_class, "F1 Beta per Class": f1_beta}

    #######################
    # END OF YOUR CODE    #
    #######################
    return metrics



def plot_loss_and_validation_acc(loss_over_time, validation_acc_over_time):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    #loss 
    ax1.plot(loss_over_time)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Over Time (numpy)')

    #validation accuracy
    ax2.plot(validation_acc_over_time)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Over Time (numpy)')

    plt.tight_layout()
    plt.savefig('combined_plot_numpy.png')



def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    train_loader = cifar10_loader['train']
    validation_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']

    # TODO: Initialize model and loss module
    n_inputs = 3*32*32
    n_classes = 10
    model = MLP(n_inputs, hidden_dims, n_classes)

    
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation


    best_model = None
    loss_over_time, val_accuracies = [], []
    best_val_accuracy = 0.0

    #model.train()

    """TRAINING AND VALIDATION"""
    for epoch in range(epochs): 
        
        # Training
        t_loss = 0.0
        for i, data in enumerate(train_loader): 
            
            data_inputs, data_labels = data
            data_inputs = data_inputs.reshape(data_inputs.shape[0], -1) #flatten input data

            #run model on input data
            preds = model.forward(data_inputs)
            
            #calculate loss
            loss = loss_module.forward(preds, data_labels)

            #backpropagation
            d_loss = loss_module.backward(preds, data_labels)
            model.backward(d_loss)

            # TODO: Do optimization with the simple SGD optimizer
            #SGD update for each layer
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                    layer.params['weight'] -= lr * layer.grads['weight']
                    layer.params['bias'] -= lr * layer.grads['bias']
              
            t_loss += loss.item()

        print(f'Epoch [{epoch + 1}] Loss: {t_loss / len(cifar10_loader):.3f}')
        loss_over_time.append(t_loss / len(cifar10_loader))
        t_loss = 0.0

        
        # Validation
        metrics_validation = evaluate_model(model, validation_loader)
        val_accuracies.append(metrics_validation['Accuracy'])

        print(f"Epoch [{epoch + 1}] Validation Accuracy: {metrics_validation['Accuracy']:.3f}")
        if metrics_validation['Accuracy'] > best_val_accuracy:
          best_val_accuracy = metrics_validation['Accuracy']
          best_model = deepcopy(model)


    # TODO: Test best model
    metrics_test = evaluate_model(best_model, test_loader)

    print(f"\n\nMETRICS FOR BEST MODEL EVALUATED ON TEST SET:\nModel:\n{metrics_test['Model']}\nConfusion Matrix:\n{metrics_test['Confusion Matrix']}\n")

    print(f"Accuracy: {metrics_test['Accuracy']:.4f}")
    for i, precision in enumerate(metrics_test['Precision per Class']):
      print(f"Precision for class {i}: {precision:.4f}")
    for i, recall in enumerate(metrics_test['Recall per Class']):
      print(f"Recall for class {i}: {recall:.4f}")
    for i, f1 in enumerate(metrics_test['F1 Beta per Class']):
        print(f"F1 Beta for class {i}: {f1:.4f}") 
    print()
    test_accuracy = metrics_test['Accuracy']


    # TODO: Add any information you might want to save for plotting
    logging_info = {
    'Validation Accuracies': val_accuracies,
    'Test Accuracies': test_accuracy,
    'Loss Over Time': loss_over_time
    }
    
    print(f"Validation Accuracies: {logging_info['Validation Accuracies']} \nTest Accuracy: {logging_info['Test Accuracies']} \nLoss Over Time: {logging_info['Loss Over Time']}")

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_info = train(**kwargs)

    plot_loss_and_validation_acc(logging_info['Loss Over Time'], logging_info['Validation Accuracies'])

    # Feel free to add any additional functions, such as plotting of the loss curve here
    