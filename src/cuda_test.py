import tensorflow.compat.v2 as tf

# checking for CUDA support
if  tf.test.is_built_with_cuda():
    print("Cuda is supported.")
else:
    print("Cuda is not supported")
print("Tensor flow version used: " + tf.__version__)

# checking for available GPUs:
if  tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
    print("The available GPUs printed above.")
else:
    print("No available GPUs.")
