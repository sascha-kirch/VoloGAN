import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow_datasets as tfds
import shutil
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
import time
import os
import io

#Define Name for Model and define callback! 
NAME = "Pix2Pix-{}".format(int(time.time()))
#tensorboard_callback = TensorBoard(
#    log_dir='logs/{}'.format(NAME), 
#    update_freq = 'epoch',
#    histogram_freq = 1, 
#    profile_batch = '100, 200'
#    )


#print Tensoflow version
print("TensorFlow version: ", tf.__version__)

# Check for GPU and set Memory groth to true!
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

log_dir = 'logs/{}'.format(NAME)
summary_writer = tf.summary.create_file_writer(log_dir)

