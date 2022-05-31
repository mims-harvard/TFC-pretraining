import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

seed = 5
tf.random.set_seed(seed)
np.random.seed(seed)

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# Library scripts
import raw_data_processing
import data_pre_processing
import simclr_models
import simclr_utitlities
import transformations

working_directory = 'sleepEDF/'
dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

# Load preprocessed data
data_folder = 'pFD_B'
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
           np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
           np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))

input_shape = (178, 1)

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

# SIMCLR _finetuning
output_shape = 3 # edit this to be the number of label classes
total_epochs = 100
batch_size = 16

base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
pretrained_model = simclr_models.attach_simclr_head(base_model)

pretrained_model = tf.keras.models.load_model(os.path.join(working_directory, f'{seed}_simclr.hdf5'))

linear_evaluation_model = simclr_models.create_linear_model_from_base_model(pretrained_model, output_shape=output_shape)

linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_finetuning.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
    monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False, verbose=0
)

training_history = linear_evaluation_model.fit(
    x = np_train[0],
    y = np_train[1],
    batch_size=batch_size,
    shuffle=True,
    epochs=total_epochs,
    callbacks=[best_model_callback],
    validation_data=np_val
)

linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)
print(np_test[0].shape, np_test[1].shape)
print("Model with lowest validation Loss:")
print(simclr_utitlities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Model in last epoch")
print(simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))
