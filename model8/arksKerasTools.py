import keras
import tensorflow as tf
from configurations import Config


"""Version 2017.12.17"""


class StupidMatrixMultiplicationLayer(object):
    """
    creates a layer that multiplies two matrices.
    yes this is a stupid class.
    Only works with tf backend.

    USAGE:
    mul_layer = MatrixMultiplicationLayer.create_layer()

    returns a layer that takes a list of TWO 2d matrices as input and multiplies them
    """

    @staticmethod
    def __matmul(matrices):
        x = matrices[0]
        y = matrices[1]
        return tf.matmul(x, y)

    @staticmethod
    def __compute_output_shape(input_shape):
        dim1 = input_shape[0][-2]
        dim2 = input_shape[1][-1]
        return [None, dim1, dim2]

    @staticmethod
    def create_layer():
        return keras.layers.Lambda(StupidMatrixMultiplicationLayer.__matmul,
                                   output_shape=StupidMatrixMultiplicationLayer.__compute_output_shape)


def weighted_cross_entropy(y_true, y_pred):
    """
    ! For keras uses, this function needs to fix weight for keras uses with functools.partial
    :param y_true: [n,m]
    :param y_pred: [n,m]
    weights: shape [1, m]. weights to be added to each category
    :return:
    """
    weights = Config.Weight_on_a1_cross_entropy
    n = tf.cast(tf.shape(y_true)[0], tf.float32)
    loss = tf.reduce_sum(-tf.log(y_pred) * y_true * weights) / n
    return loss


def normal_cross_entropy(y_true, y_pred):
    """
    :param y_true: [n,m]
    :param y_pred: [n,m]
    :return:
    """
    n = tf.cast(tf.shape(y_true)[0], tf.float32)
    loss = tf.reduce_sum(-tf.log(y_pred) * y_true) / n
    return loss


'''
#TEST
a = keras.layers.Input(shape=[1, 2], dtype='float', name='a')
b = keras.layers.Input(shape=[2, 1], dtype='float', name='b')
#s=keras.layers.Lambda(lambda x: tf.transpose(tf.transpose(x)))(a)
matmul = StupidMatrixMultiplicationLayer.create_layer()
s = matmul([a, b])
#r = StupidMatrixMultiplicationLayer.compute_output_shape(K.shape([a, b]))
model = keras.models.Model(inputs=[a, b], outputs=s)

aa = np.array([[[1,1]], [[2,2]]])
bb = np.array([[[3],[3]], [[4], [5]]])

cc = model.predict({'a':aa, 'b':bb})
'''

"""
a = keras.layers.Input(shape=[1, 6], dtype='float', name='a')
s = keras.layers.Reshape(target_shape=(2, -1))(a)

model = keras.models.Model(inputs=a, outputs=s)

aa = np.array([[[1, 2, 3, 4,5,6]], [[2,3,4,5,6,7]]])

cc = model.predict({'a':aa})
"""