import tensorflow as tf

'''
Tensorflow Version and Devices
'''

# Print TF version
print("\nTensorflow version: ", tf.__version__)

# Check available devices (CPU / GPU)
print("\nAvailable devices: ", tf.config.list_physical_devices())

'''
Constants (Immutable Tensors) and Variables
'''

# tf.constant creates an immutable tensor (like a number or array you can't change)
a = tf.constant(3)
b = tf.constant(4)

# Tensorflow supports element-wise operations
c = a + b
print("\na + b = ", c)

# Other operations . . .
print("\na ^ 2 = ", tf.pow(a, 2))

# tf.Variable creates a tensor whose value can be changed
x = tf.Variable(5.0)
print("\nInitial x: ", x.numpy()) # .numpy() gets the value out of the tensor

# assign a new value
x.assign(10.0)
print("\nAfter assign:", x.numpy())

# Assign using computation
x.assign_add(3.0) # There also exists assign_sub()
print("\nAfter add 3:", x.numpy())

'''
tf.function (Graph Mode)
'''

# tf.function turns a Python function into a Tensorflow graph
@tf.function
def multiply_and_add_one(x, y):
    return x * y + 1

# The following prints: tf.Tensor(7.0, shape=(), dtype=float32)
# If you include .numpy(), then it will extract the value and just print 7.0
result = multiply_and_add_one(tf.constant(2.0), tf.constant(3.0))
print("\nGraph result:", result)

result = multiply_and_add_one(tf.constant(2), tf.constant(3))
print("\nGraph result:", result)

result = multiply_and_add_one(2, 3)
print("\nGraph result:", result)

'''
Basic Math with Matrices
'''

# Creating 2 x 2 matrices
mat1 = tf.constant([[1, 2], [3, 4]])
mat2 = tf.constant([[5, 6], [7, 8]])

# Element-wise addition
mat_sum = mat1 + mat2
print("\nMatric sum:", mat_sum.numpy())

'''
TODO: Create two 1D tensors of length 5 using tf.constant.
Add, subtract, and multiply them. Print results.
Then try using tf.Variable and tf.function on them.
'''

one_dim_1 = tf.constant([1, 2, 3, 4, 5])
one_dim_2 = tf.constant([6, 7, 8, 9, 10])

mat_sum = one_dim_1 + one_dim_2
mat_sub = one_dim_1 - one_dim_2
mat_mult = one_dim_1 * one_dim_2

print("\nAdd:", mat_sum.numpy())
print("\nSubtract:", mat_sub.numpy())
print("\Multiplication:", mat_mult.numpy())


@tf.function
def add_mats(x, y):
    return x + y

@tf.function
def sub_mats(x, y):
    return x - y

@tf.function
def mult_mats(x, y):
    return x * y

one_dim_1 = tf.Variable([1, 2, 3, 4])
one_dim_2 = tf.Variable([5, 6, 7, 8])

result = add_mats(one_dim_1, one_dim_2)
print("\nAdd Variables:", result.numpy())

result = sub_mats(one_dim_1, one_dim_2)
print("\nSubtract Variables:", result.numpy())

result = mult_mats(one_dim_1, one_dim_2)
print("\nMultiply Variables:", result.numpy())