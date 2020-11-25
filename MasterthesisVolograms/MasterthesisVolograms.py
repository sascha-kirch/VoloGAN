import tensorflow as tf
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing
import tensorflow_datasets as tfds
import shutil
from datetime import datetime

import os


# Check on which device the task is running
tf.debugging.set_log_device_placement(True)

# tf.test.is_built_with_cuda()
# tf.test.is_gpu_available()#cuda_only=False, min_cuda_compute_capability=None)

# GPU is automatically used when applicable
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
print(c)


# Place tensors on the CPU manually. This also tries to use all available Cores!
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  c = tf.matmul(a, b)
print(c)