import numpy as np
import tensorflow as tf
import sklearn.metrics

import data_pre_processing

__author__ = "C. I. Tang"
__copyright__ = """Copyright (C) 2020 C. I. Tang"""

"""
This file includes software licensed under the Apache License 2.0, modified by C. I. Tang.

Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def generate_composite_transform_function_simple(transform_funcs):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions
            the function is composed by applying 
            transform_funcs[0] -> transform_funcs[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """
    for i, func in enumerate(transform_funcs):
        print(i, func)
    def combined_transform_func(sample):
        for func in transform_funcs:
            sample = func(sample)
        return sample
    return combined_transform_func

def generate_combined_transform_function(transform_funcs, indices=[0]):
    """
    Create a composite transformation function by composing transformation functions

    Parameters:
        transform_funcs
            list of transformation functions

        indices
            list of indices corresponding to the transform_funcs
            the function is composed by applying 
            function indices[0] -> function indices[1] -> ...
            i.e. f(x) = f3(f2(f1(x)))

    Returns:
        combined_transform_func
            a composite transformation function
    """

    for index in indices:
        print(transform_funcs[index])
    def combined_transform_func(sample):
        for index in indices:
            sample = transform_funcs[index](sample)
        return sample
    return combined_transform_func

def generate_slicing_transform_function(transform_func_structs, slicing_axis=2, concatenate_axis=2):
    """
    Create a transformation function with slicing by applying different transformation functions to different slices.
    The output arrays are then concatenated at the specified axis.

    Parameters:
        transform_func_structs
            list of transformation function structs
            each transformation functions struct is a 2-tuple of (indices, transform_func)

            each transformation function is applied by
                transform_func(np.take(data, indices, slicing_axis))
            
            all outputs are concatenated in the output axis (concatenate_axis)

            Example:
                transform_func_structs = [
                    ([0,1,2], transformations.rotation_transform_vectorized),
                    ([3,4,5], transformations.time_flip_transform_vectorized)
                ]

        slicing_axis = 2
            the axis from which the slicing is applied
            (see numpy.take)

        concatenate_axis = 2
            the axis which the transformed array (tensors) are concatenated
            if it is None, a list will be returned

    Returns:
        slicing_transform_func
            a slicing transformation function
    """
    def slicing_transform_func(sample):
        all_slices = []
        for indices, transform_func in transform_func_structs:
            trasnformed_slice = transform_func(np.take(sample, indices, slicing_axis))
            all_slices.append(trasnformed_slice)
        if concatenate_axis is None:
            return all_slices
        else:
            return np.concatenate(all_slices, axis=concatenate_axis)
    return slicing_transform_func


def get_NT_Xent_loss_gradients(model, samples_transform_1, samples_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    A wrapper function for the NT_Xent_loss function which facilitates back propagation

    Parameters:
        model
            the deep learning model for feature learning 

        samples_transform_1
            inputs samples subject to transformation 1
        
        samples_transform_2
            inputs samples subject to transformation 2

        normalize = True
            normalise the activations if true

        temperature = 1.0
            hyperparameter, the scaling factor of the logits
            (see NT_Xent_loss)
        
        weights = 1.0
            weights of different samples
            (see NT_Xent_loss)

    Return:
        loss
            the value of the NT_Xent_loss

        gradients
            the gradients for backpropagation
    """
    with tf.GradientTape() as tape:
        hidden_features_transform_1 = model(samples_transform_1)
        hidden_features_transform_2 = model(samples_transform_2)
        loss = NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=normalize, temperature=temperature, weights=weights)

    gradients = tape.gradient(loss, model.trainable_variables)
    return loss, gradients



def simclr_train_model(model, dataset, optimizer, batch_size, transformation_function, temperature=1.0, epochs=100, is_trasnform_function_vectorized=False, verbose=0):
    """
    Train a deep learning model using the SimCLR algorithm

    Parameters:
        model
            the deep learning model for feature learning 

        dataset
            the numpy array for training (no labels)
            the first dimension should be the number of samples
        
        optimizer
            the optimizer for training
            e.g. tf.keras.optimizers.SGD()

        batch_size
            the batch size for mini-batch training

        transformation_function
            the stochastic (probabilistic) function for transforming data samples
            two different views of the sample is generated by applying transformation_function twice

        temperature = 1.0
            hyperparameter of the NT_Xent_loss, the scaling factor of the logits
            (see NT_Xent_loss)
        
        epochs = 100
            number of epochs of training
            
        is_trasnform_function_vectorized = False
            whether the transformation_function is vectorized
            i.e. whether the function accepts data in the batched form, or single-sample only
            vectorized functions reduce the need for an internal for loop on each sample

        verbose = 0
            debug messages are printed if > 0

    Return:
        (model, epoch_wise_loss)
            model
                the trained model
            epoch_wise_loss
                list of epoch losses during training
    """

    epoch_wise_loss = []

    for epoch in range(epochs):
        step_wise_loss = []

        # Randomly shuffle the dataset
        shuffle_indices = data_pre_processing.np_random_shuffle_index(len(dataset))
        shuffled_dataset = dataset[shuffle_indices]

        # Make a batched dataset
        batched_dataset = data_pre_processing.get_batched_dataset_generator(shuffled_dataset, batch_size)

        for data_batch in batched_dataset:

            # Apply transformation
            if is_trasnform_function_vectorized:
                transform_1 = transformation_function(data_batch)
                transform_2 = transformation_function(data_batch)
            else:
                transform_1 = np.array([transformation_function(data) for data in data_batch])
                transform_2 = np.array([transformation_function(data) for data in data_batch])

            # Forward propagation
            loss, gradients = get_NT_Xent_loss_gradients(model, transform_1, transform_2, normalize=True, temperature=temperature, weights=1.0)

            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            step_wise_loss.append(loss)

        epoch_wise_loss.append(np.mean(step_wise_loss))
        
        if verbose > 0:
            print("epoch: {} loss: {:.3f}".format(epoch + 1, np.mean(step_wise_loss)))

    return model, epoch_wise_loss


def evaluate_model_simple(pred, truth, is_one_hot=True, return_dict=True):
    """
    Evaluate the prediction results of a model with 7 different metrics
    Metrics:
        Confusion Matrix
        F1 Macro
        F1 Micro
        F1 Weighted
        Precision
        Recall 
        Kappa (sklearn.metrics.cohen_kappa_score)

    Parameters:
        pred
            predictions made by the model

        truth
            the ground-truth labels
        
        is_one_hot=True
            whether the predictions and ground-truth labels are one-hot encoded or not

        return_dict=True
            whether to return the results in dictionary form (return a tuple if False)

    Return:
        results
            dictionary with 7 entries if return_dict=True
            tuple of size 7 if return_dict=False
    """

    if is_one_hot:
        truth_argmax = np.argmax(truth, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
    else:
        truth_argmax = truth
        pred_argmax = pred

    test_cm = sklearn.metrics.confusion_matrix(truth_argmax, pred_argmax)
    test_auroc = sklearn.metrics.roc_auc_score(truth, pred, multi_class='ovr')
    test_auprc = sklearn.metrics.average_precision_score(truth, pred)
    test_acc = sklearn.metrics.accuracy_score(truth_argmax, pred_argmax)
    test_f1 = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro')
    test_precision = sklearn.metrics.precision_score(truth_argmax, pred_argmax, average='macro')
    test_recall = sklearn.metrics.recall_score(truth_argmax, pred_argmax, average='macro')
    test_kappa = sklearn.metrics.cohen_kappa_score(truth_argmax, pred_argmax)

    test_f1_micro = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='micro')
    test_f1_weighted = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='weighted')

    if return_dict:
        return {
            'Accuracy': test_acc,
            'AUROC': test_auroc,
            'AUPRC': test_auprc,
            'Confusion Matrix': test_cm, 
            'F1 Macro': test_f1, 
            'F1 Micro': test_f1_micro, 
            'F1 Weighted': test_f1_weighted, 
            'Precision': test_precision, 
            'Recall': test_recall, 
            'Kappa': test_kappa
        }
    else:
        return (test_acc, test_auroc, test_cm, test_f1, test_f1_micro, test_f1_weighted, test_precision, test_recall, test_kappa)

"""
The following section of this file includes software licensed under the Apache License 2.0, by The SimCLR Authors 2020, modified by C. I. Tang.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""

@tf.function
def NT_Xent_loss(hidden_features_transform_1, hidden_features_transform_2, normalize=True, temperature=1.0, weights=1.0):
    """
    The normalised temperature-scaled cross entropy loss function of SimCLR Contrastive training
    Reference: Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.
    https://github.com/google-research/simclr/blob/master/objective.py

    Parameters:
        hidden_features_transform_1
            the features (activations) extracted from the inputs after applying transformation 1
            e.g. model(transform_1(X))
        
        hidden_features_transform_2
            the features (activations) extracted from the inputs after applying transformation 2
            e.g. model(transform_2(X))

        normalize = True
            normalise the activations if true

        temperature
            hyperparameter, the scaling factor of the logits
        
        weights
            weights of different samples

    Return:
        loss
            the value of the NT_Xent_loss
    """
    LARGE_NUM = 1e9
    entropy_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    batch_size = tf.shape(hidden_features_transform_1)[0]
    
    h1 = hidden_features_transform_1
    h2 = hidden_features_transform_2
    if normalize:
        h1 = tf.math.l2_normalize(h1, axis=1)
        h2 = tf.math.l2_normalize(h2, axis=1)

    labels = tf.range(batch_size)
    masks = tf.one_hot(tf.range(batch_size), batch_size)
    
    logits_aa = tf.matmul(h1, h1, transpose_b=True) / temperature
    # Suppresses the logit of the repeated sample, which is in the diagonal of logit_aa
    # i.e. the product of h1[x] . h1[x]
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(h2, h2, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(h1, h2, transpose_b=True) / temperature
    logits_ba = tf.matmul(h2, h1, transpose_b=True) / temperature

    
    loss_a = entropy_function(labels, tf.concat([logits_ab, logits_aa], 1), sample_weight=weights)
    loss_b = entropy_function(labels, tf.concat([logits_ba, logits_bb], 1), sample_weight=weights)
    loss = loss_a + loss_b

    return loss
