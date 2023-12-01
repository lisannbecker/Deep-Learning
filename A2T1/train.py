################################################################################
# MIT License
#
# Copyright (c) 2022 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2022
# Date Created: 2022-11-14
################################################################################

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
from copy import deepcopy
from torch.utils.data import DataLoader

from cifar100_utils import get_train_validation_set, get_test_set



def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(num_classes=100):
    """
    Returns a pretrained ResNet18 on ImageNet with the last layer
    replaced by a linear layer with num_classes outputs.
    Args:
        num_classes: Number of classes for the final layer (for CIFAR100 by default 100)
    Returns:
        model: nn.Module object representing the model architecture.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Get the pretrained ResNet18 model on ImageNet from torchvision.models
    model = models.resnet18(pretrained=True)
    
    all_layers = list(model.children()) #Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    #for a in all_layer:
    #    print(a)

    model.fc = nn.Linear(in_features = model.fc.in_features, out_features = num_classes) #from 1000 to 100

    # Randomly initialize and modify the model's last layer for CIFAR100.
    std = 0.01
    #print(len(list(model.named_parameters()))) 

    for name, param in model.named_parameters():
        if name == 'fc.bias':
            param.data.fill_(0)
        elif name == 'fc.weight':
            param.data.normal_(std=std)    


    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation.
        device: Device to use.
        augmentation_name: Augmentation to use for training.
    Returns:
        model: Model that has performed best on the validation set.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir, validation_size=5000, augmentation_name=None)
    #train_dataset object is a Subset object, and when you iterate over it, you get individual samples instead of batches
    train_loader, val_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True) #XXX

    print("Number of batches:", len(train_loader))

    # Initialize the optimizer (Adam) to train the last layer of the model.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr)

    # Training loop with validation after each epoch. Save the best model.
    model.to(device)

    loss_module = nn.CrossEntropyLoss()
    best_model = None
    val_accuracies = []
    best_val_accuracy = 0.0

    model.train()

    for ep in range(epochs):
        print(f"Epoch [{ep+1}]")

        for i, batch in enumerate(train_loader): 

            data, true_labels = batch
            #(data.shape) 
            data, true_labels = data.to(device), true_labels.to(device)

            #run model on input data / forward pass
            predictions = model(data)

            #calculate loss
            loss = loss_module(predictions, true_labels)

            #backpropagation
            optimizer.zero_grad() #reset gradients per batch
            loss.backward() #compute gradient of the loss by passing it through the network in reverse order
            optimizer.step() #update parameters (weights and bias)
        
        
        print('VALIDATION ACCURACY')
        validation_accuracy = evaluate_model(model, val_loader, device)
        val_accuracies.append(validation_accuracy)

        if validation_accuracy > best_val_accuracy:
            best_model = deepcopy(model)


    # Load the best model on val accuracy and return it.
    model = best_model

    #Save model
    state_dict = model.state_dict()
    torch.save(state_dict, f"{checkpoint_name}-imagenet_cifar100.tar")

    #######################
    # END OF YOUR CODE    #
    #######################

    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set model to evaluation mode (Remember to set it back to training mode in the training loop)
    model.eval()

    # Loop over the dataset and compute the accuracy. Return the accuracy
    # Remember to use torch.no_grad().
    correct_preds, total_preds = 0.,0.
    print('EVALUATION')
    with torch.no_grad():
        for i, (data, true_labels) in enumerate(data_loader):
            data, true_labels = data.to(device), true_labels.to(device)

            #if data.size(0) < data_loader.batch_size:
            #    continue

            preds_batch = model(data) #foward pass, validation set
            _, predicted_labels = torch.max(preds_batch.data, 1) #get class with highest probability

            total_preds += true_labels.shape[0]
            correct_preds += (predicted_labels == true_labels).sum().item()

    
    accuracy = correct_preds / total_preds
    print(f"Accuracy of the model: {accuracy:.3f}")

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def main(lr, batch_size, epochs, data_dir, seed, augmentation_name, test_noise):
    """
    Main function for training and testing the model.

    Args:
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        seed: Seed for reproducibility.
        augmentation_name: Name of the augmentation to use.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Set the seed for reproducibility
    set_seed(seed)

    # Set the device to use for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load the model
    model = get_model(num_classes=100)
    #model = model.to(device)

    # Get the augmentation to use
    train_augmentation = 'test_noise'

    # Train the model
    checkpoint_name = 'best_model'
    model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device, augmentation_name=None) #augmentation_name = 'GaussianNoise'

    # Evaluate the model on the test set
    
    test_dataset = get_test_set(data_dir, test_noise)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print('TEST SET')
    test_accuracy = evaluate_model(model, test_loader, device) #run best model (highest validation accuracy) on test set
    print(f'Test accuracy: {test_accuracy}')

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Feel free to add more arguments or change the setup

    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=123, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR100 dataset.')
    parser.add_argument('--augmentation_name', default=None, type=str,
                        help='Augmentation to use.')
    parser.add_argument('--test_noise', default=False, action="store_true",
                        help='Whether to test the model on noisy images or not.')


    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
