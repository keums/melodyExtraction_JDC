
import keras.backend as KK
from keras import backend as K
from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Dropout,\
                        LSTM,Reshape,Bidirectional,TimeDistributed,Input,add,concatenate,Lambda

import math

def ResNet_Block(input,block_id,filterNum):
    ''' Create a ResNet block
    Args:
        input: input tensor
        filterNum: number of output filters
    Returns: a keras tensor
    '''
    x = BatchNormalization()(input)
    x = LeakyReLU(0.01)(x)
    x = MaxPooling2D((1, 4))(x)

    init = Conv2D(filterNum, (1, 1), name='conv'+str(block_id)+'_1x1', padding='same', kernel_initializer='he_normal', use_bias=False)(x)
    x = Conv2D(filterNum, (3, 3), name='conv'+str(block_id)+'_1',padding='same',kernel_initializer='he_normal',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.01)(x)
    x = Conv2D(filterNum, (3, 3),  name='conv'+str(block_id)+'_2',padding='same',kernel_initializer='he_normal',use_bias=False)(x)

    x = add([init, x])
    return x


def melody_ResNet_joint_add(options):
    num_output = int(45 * 2 ** (math.log(options.resolution, 2)) + 2)
    input = Input(shape=(options.input_size, options.num_spec, 1))

    block_1 = Conv2D(64, (3, 3), name='conv1_1', padding='same', kernel_initializer='he_normal', use_bias=False,
                     kernel_regularizer=l2(1e-5))(input)
    block_1 = BatchNormalization()(block_1)
    block_1 = LeakyReLU(0.01)(block_1)
    block_1 = Conv2D(64, (3, 3), name='conv1_2', padding='same', kernel_initializer='he_normal', use_bias=False,
                     kernel_regularizer=l2(1e-5))(block_1)

    block_2 = ResNet_Block(input=block_1, block_id=2, filterNum=128)
    block_3 = ResNet_Block(input=block_2, block_id=3, filterNum=192)
    block_4 = ResNet_Block(input=block_3, block_id=4, filterNum=256)

    block_4 = BatchNormalization()(block_4)
    block_4 = LeakyReLU(0.01)(block_4)
    block_4 = MaxPooling2D((1, 4))(block_4)
    block_4 = Dropout(0.5)(block_4)

    numOutput_P = 2 * block_4._keras_shape[3]
    output = Reshape((options.input_size, numOutput_P))(block_4)

    output = Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.3, dropout=0.3))(output)
    output = TimeDistributed(Dense(num_output))(output)
    output = TimeDistributed(Activation("softmax"), name='output')(output)

    block_1 = MaxPooling2D((1, 4 ** 4))(block_1)
    block_2 = MaxPooling2D((1, 4 ** 3))(block_2)
    block_3 = MaxPooling2D((1, 4 ** 2))(block_3)

    joint = concatenate([block_1, block_2, block_3, block_4])
    joint = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', use_bias=False,
                   kernel_regularizer=l2(1e-5))(joint)
    joint = BatchNormalization()(joint)
    joint = LeakyReLU(0.01)(joint)
    joint = Dropout(0.5)(joint)

    num_V = joint._keras_shape[3] * 2
    output_V = Reshape((options.input_size, num_V))(joint)

    output_V = Bidirectional(LSTM(32, return_sequences=True, stateful=False, recurrent_dropout=0.3, dropout=0.3))(
        output_V)
    output_V = TimeDistributed(Dense(2))(output_V)
    output_V = TimeDistributed(Activation("softmax"))(output_V)

    output_NS = Lambda(lambda x: x[:, :, 0])(output)
    output_NS = Reshape((options.input_size, 1))(output_NS)
    output_S = Lambda(lambda x: 1 - x[:, :, 0])(output)
    output_S = Reshape((options.input_size, 1))(output_S)
    output_VV = concatenate([output_NS, output_S])

    output_V = add([output_V, output_VV])
    output_V = TimeDistributed(Activation("softmax"), name='output_V')(output_V)

    model = Model(inputs=input, outputs=[output, output_V])
    return model