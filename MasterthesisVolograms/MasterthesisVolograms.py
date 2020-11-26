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

NAME = "Sascha-first-model-{}".format(int(time.time()))
tensorboard_callback = TensorBoard(
    log_dir='logs/{}'.format(NAME), 
    update_freq = 'epoch',
    histogram_freq = 1, 
    profile_batch = '100, 200'
    )


#"C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\tensorboard\main.py"
#os.system('py "C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python37_64\\Lib\\site-packages\\tensorboard\\main.py" --logdir=logs/ --host localhost')
#log_dir = os.path.join('logs','fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
#print(log_dir)
#os.makedirs(log_dir, exist_ok=True)


# Check on which device the task is running
#tf.debugging.set_log_device_placement(True)

#tf.test.is_built_with_cuda()
#tf.test.is_gpu_available()#cuda_only=False, min_cuda_compute_capability=None)

# GPU is automatically used when applicable
#a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#c = tf.matmul(a, b)
#print(c)


# Place tensors on the CPU manually. This also tries to use all available Cores!
#with tf.device('/CPU:0'):
#  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#  c = tf.matmul(a, b)
#print(c)

# Commented out IPython magic to ensure Python compatibility.
#try:
  # %tensorflow_version only exists in Colab.
#   %tensorflow_version 2.x
#except Exception:
#  pass

print("TensorFlow version: ", tf.__version__)

device_name = tf.test.gpu_device_name()
if not device_name:
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)  
amplitude = 20
slope = 0.09
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
# Update with noise
series += noise(time, noise_level, seed=42)

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset

window_size = 30
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.SGD(lr=8e-6, momentum=0.9)


model.compile(
    loss="mse", 
    optimizer=optimizer
    )


model.fit(dataset,
          epochs=500,
          verbose=2,
          callbacks = [tensorboard_callback])
