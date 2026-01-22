import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np

# 1. Create two sample tensors

tensor1 = tf.constant([[1, 2], [3, 4]])
tensor2 = tf.constant([[5, 6], [7, 8]])

# 2. Perform element-wise addition
added_tensors = tf.add(tensor1, tensor2)
print("Element-wise Addition:\n", added_tensors.numpy()) 
# element-wise addition of tensor1 and tensor2
#  [[ 6  8]
#   [10 12]]

# 3. Perform element-wise multiplication
multiplied_tensors = tf.multiply(tensor1, tensor2)
print("Element-wise Multiplication:\n", multiplied_tensors.numpy())
# element-wise multiplication of tensor1 and tensor2
#  [[ 5 12]
#   [21 32]]

# 4. Reshape a tensor
reshaped_tensor = tf.reshape(tensor1, [1, 4])
print("Reshaped Tensor:\n", reshaped_tensor.numpy())
# Tensor 1 is reshaped from 2x2 to 1x4.
#  [[1 2 3 4]]

# 5. Slicing and indexing a tensor
sliced_tensor = tensor2[:, 1]
indexed_value = tensor1[0, 1]
print("Indexed Value (tensor1[0,1]):", indexed_value.numpy())
print("Sliced Tensor (second column of tensor2):\n", sliced_tensor.numpy())

# Slicing to get the second column of tensor2 results in [6 8]
# Indexing to get the element at first row, second column of tensor1 results in 2

# 6. Compute the mean of a tensor

tensor4 = tf.constant([ 1, 4, 2, 3])
mean_tensor = tf.reduce_mean(tensor4)
print("Mean of tensor4 int:\n", mean_tensor.numpy())
# Integers are truncuated, so output is 2
tensor4 = tf.constant([ 1.0, 4.0, 2.0, 3.0])
mean_tensor = tf.reduce_mean(tensor4)
print("Mean of tensor4 float:\n", mean_tensor.numpy())
# if float is used we get 2.5

# 7. Splitting a tensor
tensor5 = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
split_tensors = tf.split(tensor5, num_or_size_splits=2, axis=1)
print("Split Tensors:\n", [t.numpy() for t in split_tensors])

#tensor 5 is split into two tensors along the second axis (columns).
# Split Tensors:
#[array([[1, 2],
#        [5, 6]]), 
# array([[3, 4],
#        [7, 8]])]

# 8. Basic math operations

print("tensor1",tensor1.numpy(), "tensor2",tensor2.numpy()) # tensorstensor1 [[1 2] [3 4]] tensor2 [[5 6] [7 8]]
print("addition",tf.add(tensor1, 4))  # Addition [[ 5  6] [ 7  8]]
print("subtraction",tf.subtract(tensor2, 10))  # Subtraction [[-5 -4] [-3 -2]]
print("multiplication",tf.multiply(tensor1, tensor2))  # Multiplication [[ 5 12] [21 32]]_