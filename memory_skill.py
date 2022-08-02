import logging
import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.contrib import layers
from utils import getLogger
from tensorflow.contrib import framework as contrib_framework
# Code reused from https://github.com/ckyeungac/DeepIRT.git

# set logger
logger = getLogger('Deep-IRT-model-HN')
nest = contrib_framework.nest

_BIAS_VARIABLE_NAME = 'biasmogreifer'
_WEIGHTS_VARIABLE_NAME = 'kernelmogrifer'

_BIAS_VARIABLE_NAME1 = 'biasmogreifer1'
_WEIGHTS_VARIABLE_NAME1 = 'kernelmogrifer1'

_BIAS_VARIABLE_NAME2 = 'biasmogreifer2'
_WEIGHTS_VARIABLE_NAME2 = 'kernelmogrifer2'


class MemoryHeadGroup():
    def __init__(self, memory_size, memory_state_dim, delta_1, delta_2, num_pattern,rounds,batch_size, is_write, name="DKVMN-Head"):
        self.name = name
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.num_pattern = num_pattern
        self.rounds = rounds
        self.batch_size = batch_size
        self.is_write = is_write

    def correlation_weight(self, embedded_query_vector, key_memory_matrix):
        """
        Given a batch of queries, calculate the similarity between the query and
        each key-memory slot via inner dot product. Then, calculate the weighting
        of each memory slot by softmax function.

        Parameters:
            - embedded_query_vector (k): Shape (batch_size, key_memory_state_dim)
            - key_memory_matrix (D_k): Shape (memory_size, key_memory_state_dim)
        Result:
            - correlation_weight (w): Shape (batch_size, memory_size)
        """
        embedding_result = tf.matmul(
            embedded_query_vector, tf.transpose(key_memory_matrix)
        )
        correlation_weight = tf.nn.softmax(embedding_result)
        return correlation_weight

    def read(self, value_memory_matrix, correlation_weight):
        """
        Given the correlation_weight, read the value-memory in each memory slot
        by weighted sum. This operation is assumpted to be done in batch manner.

        Parameters:
            - value_memory_matrix (D_v): Shape (batch_size, memory_size, value_memory_state_dim)
            - correlation_weight (w): Shape (batch_size, memory_size)
        Result:
            - read_result (r): Shape (batch_size, value_memory_state_dim)
        """
        value_memory_matrix_reshaped = tf.reshape(value_memory_matrix, [-1, self.memory_state_dim])
        correlation_weight_reshaped = tf.reshape(correlation_weight, [-1, 1])

        _read_result = tf.multiply(value_memory_matrix_reshaped, correlation_weight_reshaped)  # row-wise multiplication
        read_result = tf.reshape(_read_result, [-1, self.memory_size, self.memory_state_dim])
        read_result = tf.reduce_sum(read_result, axis=1, keepdims=False)
        return read_result

    def linear(self, args, output_size, rounds, bias, bias_start=0.0, initializer=None, scope=None):
        with tf.variable_scope(scope or 'linear', initializer=initializer):
            return self._linear(args, output_size, bias, rounds, bias_start)

    def _linear(self, args, output_size, bias, rounds, bias_start=0.0):
        """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

        Args:
          args: a 2D Tensor or a list of 2D, batch x n, Tensors.
          output_size: int, second dimension of W[i].
          bias: boolean, whether to add a bias term or not.
          bias_start: starting value to initialize the bias; 0 by default.

        Returns:
          A 2D Tensor with shape [batch x output_size] equal to
          sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

        Raises:
          ValueError: if some of the arguments has unspecified or wrong shape.
        """

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError('`args` must be specified')
        if not nest.is_sequence(args):
            args = [args]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size = 0
        shapes = [a.get_shape() for a in args]
        for shape in shapes:
            if shape.ndims != 2:
                raise ValueError('linear is expecting 2D arguments: %s' % shapes)
            if shape[1].value is None:
                raise ValueError('linear expects shape[1] to be provided for shape %s, '
                                 'but saw %s' % (shape, shape[1]))
            else:
                total_arg_size += shape[1].value

        dtype = [a.dtype for a in args][0]

        # Now the computation.
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as outer_scope:

            weights = tf.get_variable(
                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
            if len(args) == 1:
                res = tf.matmul(args[0], weights)
            else:
                res = tf.matmul(tf.concat(args, 1), weights)
            if not bias:
                return res
            with tf.variable_scope(outer_scope, reuse=tf.AUTO_REUSE) as inner_scope:

                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                    _BIAS_VARIABLE_NAME, [output_size],
                    dtype=dtype,
                    initializer=tf.constant_initializer(bias_start, dtype=dtype))

            return tf.add(res, biases)

    def linear_lstm(self, arg1, arg2, output_size, bias, bias_start=0.0, initializer=None, scope=None):
        with tf.variable_scope(scope or 'linear_lstm', initializer=initializer):
            return self._linear_lstm(arg1, arg2, output_size, bias, bias_start)

    def _linear_lstm(self, arg1, arg2, output_size, bias, bias_start=0.0):

        if arg1 is None or (nest.is_sequence(arg1) and not arg1):
            raise ValueError('`arg1` must be specified')
        if not nest.is_sequence(arg1):
            arg1 = [arg1]

        if arg2 is None or (nest.is_sequence(arg2) and not arg2):
            raise ValueError('`arg2` must be specified')
        if not nest.is_sequence(arg2):
            arg2 = [arg2]

        # Calculate the total size of arguments on dimension 1.
        total_arg_size1 = 0
        total_arg_size2 = 0
        shapes1 = [a.get_shape() for a in arg1]
        shapes2 = [a.get_shape() for a in arg2]
        for shape in shapes1:
            if shape.ndims != 2:
                raise ValueError('linear is expecting 2D arguments: %s' % shapes1)
            if shape[1].value is None:
                raise ValueError('linear expects shape[1] to be provided for shape %s, '
                                 'but saw %s' % (shape, shape[1]))
            else:
                total_arg_size1 += shape[1].value

        for shape in shapes2:
            if shape.ndims != 2:
                raise ValueError('linear is expecting 2D arguments: %s' % shapes2)
            if shape[1].value is None:
                raise ValueError('linear expects shape[1] to be provided for shape %s, '
                                 'but saw %s' % (shape, shape[1]))
            else:
                total_arg_size2 += shape[1].value

        dtype1 = [a.dtype for a in arg1][0]
        dtype2 = [a.dtype for a in arg2][0]

        # Now the computation.
        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as outer_scope:

            weights1 = tf.get_variable(
                _WEIGHTS_VARIABLE_NAME1, [total_arg_size1, output_size], dtype=dtype1)
            weights2 = tf.get_variable(
                _WEIGHTS_VARIABLE_NAME2, [total_arg_size2, output_size], dtype=dtype2)
            if len(arg1) == 1:
                res1 = tf.matmul(arg1[0], weights1)
            else:
                res1 = tf.matmul(tf.concat(arg1, 1), weights1)

            if len(arg2) == 1:
                res2 = tf.matmul(arg2[0], weights2)
            else:
                res2 = tf.matmul(tf.concat(arg2, 1), weights2)

            res = res1 + res2
            if not bias:
                return res
            with tf.variable_scope(outer_scope, reuse=tf.AUTO_REUSE) as inner_scope:

                inner_scope.set_partitioner(None)
                biases = tf.get_variable(
                    _BIAS_VARIABLE_NAME, [output_size],
                    dtype=dtype1,
                    initializer=tf.constant_initializer(bias_start, dtype=dtype1))

            return tf.add(res, biases)

    def mogrifer(self, embedded_content_vector, value_memory_matrix, delta_1, delta_2, rounds):
        i = 0
        while i <= rounds - 1:
            fm_name = 'fm_' + str(i)
            # [32,100]
            if i % 2 == 0:
                embedded_content_vector = embedded_content_vector * tf.transpose(delta_1 * tf.sigmoid(
                    self.linear(tf.transpose(tf.reshape(value_memory_matrix, [-1, self.memory_state_dim])), self.batch_size, 2,
                                bias=True, scope=fm_name)))


            else:
                # [32,50,100]
                value_memory_matrix = value_memory_matrix * delta_2 * tf.sigmoid(
                    tf.reshape(self.linear(embedded_content_vector, self.memory_state_dim, 2, bias=True, scope=fm_name),
                               [-1, 1, self.memory_state_dim]))
            i = i + 1
        return embedded_content_vector, value_memory_matrix

    def write(self, value_memory_matrix, correlation_weight, embedded_content_vector, memory_matrix_pre_list,
              reuse=False):
        """
        Update the value_memory_matrix based on the correlation weight and embedded result vector.

        Parameters:
            - value_memory_matrix (D_v): Shape (batch_size, memory_size, value_memory_state_dim)
            - correlation_weight (w): Shape (batch_size, memory_size)
            - embedded_content_vector (v): Shape (batch_size, value_memory_state_dim)
            - reuse: indicate whether the weight should be reuse during training.
        Return:
            - new_value_memory_matrix: Shape (batch_size, memory_size, value_memory_state_dim)
        """
        assert self.is_write

        value_memory_matrix_pre = value_memory_matrix
        # Adding the previous pattern to the current memory
        for i in range(len(memory_matrix_pre_list)):
            value_memory_matrix = tf.concat([value_memory_matrix, memory_matrix_pre_list[i]], 2)
        if self.num_pattern > 0:
        # If changing to the pattern 0,switch the activation function to None
            value_memory_matrix = layers.fully_connected(
                inputs=value_memory_matrix,
                num_outputs=self.memory_state_dim,
                scope=self.name + '/con_preoperation',
                reuse=reuse,
                activation_fn=tf.sigmoid
            )

        embedded_content_vector, value_memory_matrix = self.mogrifer(embedded_content_vector, value_memory_matrix,
                                                                     self.delta_1, self.delta_1, self.rounds)

        value_memory_matrix_reshaped_hpre = tf.reshape(value_memory_matrix, [self.batch_size, -1])

        erase_signal = tf.sigmoid(self.linear_lstm(embedded_content_vector, value_memory_matrix_reshaped_hpre,
                                                   self.memory_state_dim, bias=True,
                                                   scope=self.name + '/EraseOperation', ))

        zt_signal = tf.sigmoid(self.linear_lstm(embedded_content_vector, value_memory_matrix_reshaped_hpre,
                                                self.memory_state_dim, bias=True,
                                                scope=self.name + '/ZtOperation', ))

        add_signal = tf.tanh(self.linear_lstm(value_memory_matrix_reshaped_hpre, zt_signal, self.memory_state_dim,
                                              bias=True, scope=self.name + '/AddOperation', ))

        add_reshaped = tf.reshape(add_signal, [-1, 1, self.memory_state_dim])
        erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.memory_state_dim])
        # reshape from (batch_size, memory_size) to (batch_size, memory_size, 1)
        cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])

        # erase_mul/add_mul: Shape (batch_size, memory_size, value_memory_state_dim)
        add_mul = tf.multiply(add_reshaped, cw_reshaped)
        erase_mul = tf.multiply(erase_reshaped, cw_reshaped)
        # Update value memory
        new_value_memory_matrix = value_memory_matrix * (1 - erase_mul)  # erase memory
        new_value_memory_matrix += add_mul

        return new_value_memory_matrix, value_memory_matrix_pre


class DKVMN():
    def __init__(self, memory_size, key_memory_state_dim, value_memory_state_dim, num_pattern, delta_1, delta_2, rounds,batch_size,
                 init_key_memory=None, init_value_memory=None, name="DKVMN"):
        self.name = name
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim
        self.num_pattern = num_pattern
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.rounds = rounds
        self.batch_size = batch_size

        self.key_head = MemoryHeadGroup(
            self.memory_size, self.key_memory_state_dim, self.delta_1, self.delta_2, self.num_pattern, self.rounds,self.batch_size,
            name=self.name + '-KeyHead', is_write=False
        )
        self.value_head = MemoryHeadGroup(
            self.memory_size, self.value_memory_state_dim, self.delta_1, self.delta_2, self.num_pattern, self.rounds,self.batch_size,
            name=self.name + '-ValueHead', is_write=True
        )

        self.key_memory_matrix = init_key_memory
        self.value_memory_matrix = init_value_memory

    def attention(self, embedded_query_vector):
        correlation_weight = self.key_head.correlation_weight(
            embedded_query_vector=embedded_query_vector,
            key_memory_matrix=self.key_memory_matrix
        )
        return correlation_weight

    def read(self, correlation_weight):
        read_content = self.value_head.read(
            value_memory_matrix=self.value_memory_matrix,
            correlation_weight=correlation_weight
        )
        return read_content

    def write(self, correlation_weight, embedded_result_vector, memory_matrix_pre_list, reuse):
        self.value_memory_matrix, value_memory_matrix_pre = self.value_head.write(
            value_memory_matrix=self.value_memory_matrix,
            correlation_weight=correlation_weight,
            embedded_content_vector=embedded_result_vector,
            memory_matrix_pre_list=memory_matrix_pre_list,
            reuse=reuse
        )
        return self.value_memory_matrix, value_memory_matrix_pre
