import tensorflow as tf

# Basic constant operations
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph.
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))

# Basic Operations with variable as graph input
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Launch the default graph.
with tf.Session() as sess:
    # Run every operation with variable input
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# Matrix Multiplication from TensorFlow official tutorial
matrix1 = tf.constant([[3., 3.]])
# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])

# Matrix multiplication.
product = tf.matmul(matrix1, matrix2)

# The output of the op is returned in 'result' as a numpy `ndarray` object.
with tf.Session() as sess:
    result = sess.run(product)
    print('Matrix multiplication: ', result)
