from tensorflow.keras import models
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, pipeline, AutoModel
import resources.smart_cond as sc
import gc
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Embedding, Activation, Dropout, Softmax, Layer, InputSpec, Input, Dense, Lambda, TimeDistributed, Concatenate, Add
from tensorflow.keras import initializers, regularizers, constraints, Model
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow import nn


class Multimodal_Fusion(Layer):
    def __init__(self, num_latents=880+50, name="multimodal_fusion"):
        super(Multimodal_Fusion, self).__init__(name=name)

        # Latents
        self.num_latents = num_latents
        self.latents = self.add_weight(name="latents", shape=(
            1, num_latents, 50), initializer='random_normal', trainable=True)
        self.scale_a = self.add_weight(name="scale_a", shape=(
            1,), initializer='zeros', trainable=True)
        self.scale_v = self.add_weight(name="scale_v", shape=(
            1,), initializer='zeros', trainable=True)

    # requires q,k,v to have same dim
    def multimodal_fusion_attention(self, q, k, v):
        B, N, C = q.shape
        B, _, _ = k.shape
        attn = tf.matmul(q, k, transpose_b=True) * (C ** -0.5)  # scaling
        attn = tf.nn.softmax(attn, axis=-1)
        x = tf.matmul(attn, v)
        # x = tf.reshape(x, (B, N, C))
        return x

    # Latent Fusion
    def multimodal_fusion(self, pysio_embs, text_embs):
        # shapes
        B, N, C = pysio_embs.shape
        # concat all the tokens
        concat_ = tf.concat([pysio_embs, text_embs], axis=1)
        # cross attention (AV -->> latents)
        # X = tf.broadcast_to(self.latents, [B, N, C])
        fused_latents = self.multimodal_fusion_attention(
            self.latents, concat_, concat_)
        # cross attention (latents -->> AV)
        pysio_embs = pysio_embs + self.scale_a * \
            self.multimodal_fusion_attention(
                pysio_embs, fused_latents, fused_latents)
        text_embs = text_embs + self.scale_v * \
            self.multimodal_fusion_attention(
                text_embs, fused_latents, fused_latents)
        return pysio_embs, text_embs

    def call(self, x, y):

        # Bottleneck Fusion
        x, y = self.multimodal_fusion(x, y)

        return tf.concat([x, y], axis=1)


class CVE(Layer):
    def __init__(self, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(CVE, self).__init__()

    def build(self, input_shape):
        self.W1 = self.add_weight(name='CVE_W1',
                                  shape=(1, self.hid_units),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.b1 = self.add_weight(name='CVE_b1',
                                  shape=(self.hid_units,),
                                  initializer='zeros',
                                  trainable=True)
        self.W2 = self.add_weight(name='CVE_W2',
                                  shape=(self.hid_units, self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(CVE, self).build(input_shape)

    def call(self, x):
        x = K.expand_dims(x, axis=-1)
        x = K.dot(K.tanh(K.bias_add(K.dot(x, self.W1), self.b1)), self.W2)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)


class Attention(Layer):

    def __init__(self, hid_dim):
        self.hid_dim = hid_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        self.W = self.add_weight(shape=(d, self.hid_dim), name='Att_W',
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.hid_dim,), name='Att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.hid_dim, 1), name='Att_u',
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e30):
        attn_weights = K.dot(
            K.tanh(K.bias_add(K.dot(x, self.W), self.b)), self.u)
        mask = K.expand_dims(mask, axis=-1)
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)


class Transformer(Layer):

    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(Transformer, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        if self.dk == None:
            self.dk = d//self.h
        if self.dv == None:
            self.dv = d//self.h
        if self.dff == None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wq',
                                  initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wk',
                                  initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.N, self.h, d, self.dv), name='Wv',
                                  initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.N, self.dv*self.h, d), name='Wo',
                                  initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.N, d, self.dff), name='W1',
                                  initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.N, self.dff), name='b1',
                                  initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.N, self.dff, d), name='W2',
                                  initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.N, d), name='b2',
                                  initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2*self.N,), name='gamma',
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2*self.N,), name='beta',
                                    initializer='zeros', trainable=True)
        super(Transformer, self).build(input_shape)

    def call(self, x, mask, mask_value=-1e-30):
        mask = K.expand_dims(mask, axis=-2)
        for i in range(self.N):
            # MHA
            mha_ops = []
            for j in range(self.h):
                q = K.dot(x, self.Wq[i, j, :, :])
                k = K.permute_dimensions(
                    K.dot(x, self.Wk[i, j, :, :]), (0, 2, 1))
                v = K.dot(x, self.Wv[i, j, :, :])
                A = K.batch_dot(q, k)
                # Mask unobserved steps.
                A = mask*A + (1-mask)*mask_value
                # Mask for attention dropout.

                def dropped_A():
                    dp_mask = K.cast(
                        (K.random_uniform(shape=array_ops.shape(A)) >= self.dropout), K.floatx())
                    return A*dp_mask + (1-dp_mask)*mask_value
                A = sc.smart_cond(K.learning_phase(), dropped_A,
                                  lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A, v))
            conc = K.concatenate(mha_ops, axis=-1)
            proj = K.dot(conc, self.Wo[i, :, :])
            # Dropout.
            proj = sc.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),
                                 lambda: array_ops.identity(proj))
            # Add & LN
            x = x+proj
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i] + self.beta[2*i]
            # FFN
            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i, :, :]), self.b1[i, :])),
                                      self.W2[i, :, :]), self.b2[i, :,])
            # Dropout.
            ffn_op = sc.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),
                                   lambda: array_ops.identity(ffn_op))
            # Add & LN
            x = x+ffn_op
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i+1] + self.beta[2*i+1]
        return x

    def compute_output_shape(self, input_shape):
        return input_shape