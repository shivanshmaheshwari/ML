import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))

# Disable GPU support
# tf.config.set_visible_devices([], 'GPU')
# print(tf.config.list_physical_devices('GPU'))  

# # Enable GPU support
# tf.config.set_visible_devices(tf.config.list_physical_devices('GPU'), 'GPU')
# print(tf.config.list_physical_devices('GPU'))

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # memory growth
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)