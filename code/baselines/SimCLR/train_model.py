import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

seed = 2
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

working_directory = 'test_run/'
dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

# Load preprocessed data
data_folder = 'test_run'
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
           np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
           np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))
input_shape = (1500, 1)

# SIMCLR training
batch_size = 256
decay_steps = 1000
epochs = 5
temperature = 0.1
transform_funcs = [
#     transformations.rotation_transform_vectorized # Use rotation trasnformation
    transformations.scaling_transform_vectorized,
    transformations.negate_transform_vectorized
]
transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)

# trasnformation_indices = [2] # Use rotation trasnformation only
# trasnformation_indices = [1, 2] # Use Scaling and rotation trasnformation

# trasnform_funcs_vectorized = [
#     transformations.noise_transform_vectorized,
#     transformations.scaling_transform_vectorized,
#     transformations.rotation_transform_vectorized,
#     transformations.negate_transform_vectorized,
#     transformations.time_flip_transform_vectorized,
#     transformations.time_segment_permutation_transform_improved,
#     transformations.time_warp_transform_low_cost,
#     transformations.channel_shuffle_transform_vectorized
# ]
# transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = simclr_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()

trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer, batch_size, transformation_function, temperature=temperature, epochs=epochs, is_trasnform_function_vectorized=True, verbose=1)

simclr_model_save_path = f"{seed}_simclr.hdf5"
trained_simclr_model.save(simclr_model_save_path)

#plt.figure(figsize=(12,8))
#plt.plot(epoch_losses)
#plt.ylabel("Loss")
#plt.xlabel("Epoch")
#plt.show()
#plt.savefig(os.path.join(working_directory, "epoch_loss_plot.png"))

