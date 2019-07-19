'''
History:
- miniVggNet was originally written by Adrian Rosebrock on 11 February 2019 in his
  https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/ using the Sequential Keras model.

- All other CNNs are taken from Deep Learning for Computer Vision with Python.

- All CNNs of this file were rewritten by me to remove the Sequential model, as I need to force the input layer name for the quantization process.

- Note that the sequence of layers "CONV-> RELU -> BN" in LeNet and miniVggNet CNN has to be replaced by the sequence "CONV-> BN -> RELU" 
  as the first  one does not allow the merging of all the layers during the compilation phase (and also might have potential issues 
  during 8-bit quantization).

Date: 9 July 2019 by daniele.bagni@xilinx.com

'''

''' #DB: sequential model original code

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class MiniVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# first CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# second CONV => RELU => CONV => RELU => POOL layer set
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
'''



# import the necessary packages
from keras.models import Model
from keras.layers import Input
from keras.layers import add
from keras.regularizers import l2
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import concatenate
from keras import backend as K


class ConvNetFactory:
        def __init__(self):
                pass

        @staticmethod
        def build(name, *args, **kargs):
                # define the network (i.e., string => function) mappings
                mappings = {"LeNet"        : ConvNetFactory.LeNet,
                            "miniVggNet"   : ConvNetFactory.miniVggNet,
                            "miniGoogleNet": ConvNetFactory.miniGoogleNet,
                            "miniResNet"   : ConvNetFactory.miniResNet}
                # grab the builder function from the mappings dictionary
                builder = mappings.get(name, None)
                # if the builder is None, then there is not a function that can be used to build to the network, so return None
                if builder is None:
                        return None
                # otherwise, build the network architecture
                return builder(*args, **kargs)

        @staticmethod
        def LeNet(width, height, depth, classes=10, reg=0.0001, bnEps=2e-5, bnMom=0.9, dropout=False, BN=True, FC=True, **kargs):
                # initialize the input shape to be "channels last" and the channels dimension itself
                inputShape = (height, width, depth)
                chanDim = -1
                # if we are using "channels first", update the input shape and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1
                inputs = Input(shape=inputShape, name="conv2d_1_input")
                # define the first set of CONV => ACTIVATION => POOL layer
                x = Conv2D(20, (5, 5), padding="same", kernel_regularizer=l2(reg))(inputs)
                if BN:
                        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)                        
                x = MaxPooling2D(pool_size=(2, 2))(x)
                if dropout:
                        x = Dropout(0.25)(x)

                x = Conv2D(50, (5, 5), padding="same", kernel_regularizer=l2(reg))(inputs)
                if BN:
                        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)                      
                x = MaxPooling2D(pool_size=(2, 2))(x)
                if dropout:
                        x = Dropout(0.25)(x)
                if FC:
                        #define the first FC => ACTIVATION layers
                        x = Flatten()(x)
                        x = Dense(500, kernel_regularizer=l2(reg))(x)
                        x = Activation("relu")(x)
                        # define the second FC layer
                        x = Dense(classes, kernel_regularizer= l2(reg))(x)
                        # lastly, define the soft-max classifier
                        x = Activation("softmax")(x)
                else: # CONV layers to replace the FC layers
                        kernel_size = (width//2, height//2)
                        x = Conv2D(500, kernel_size)(x) # input data has size 14x14 #DB
                        x = Conv2D(500, (1, 1))(x)
                        x = Activation("relu")(x)
                        x = Conv2D(classes, (1, 1))(x)
                        x = Flatten()(x)
                        x = Activation("softmax")(x)

                # create the model
                model = Model(inputs, x, name="LeNet")
                return model

        @staticmethod
        def miniVggNet(width, height, depth, classes=10, reg=0.0001, bnEps=2e-5, bnMom=0.9,dropout=False, BN=True, FC=True, **kargs):
                # initialize the input shape to be "channels last" and the
                # channels dimension itself
                inputShape = (height, width, depth)
                chanDim = -1

                # if we are using "channels first", update the input shape and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1

                # set the inpu			"minivggnet": cnn.MiniVGGNet}t and apply BN
                inputs = Input(shape=inputShape, name="conv2d_1_input")

                # apply a single CONV layer
                x = Conv2D(32, (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(inputs)
                #x = Activation("relu")(x)
                #x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)

                x = Conv2D(32, (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
                #x = Activation("relu")(x)
                #x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)

                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.25)(x)

                # apply a single CONV layer
                x = Conv2D(64, (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
                #x = Activation("relu")(x)
                #x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)

                x = Conv2D(64, (3, 3), use_bias=False, padding="same", kernel_regularizer=l2(reg))(x)
                #x = Activation("relu")(x)
                #x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                x = Activation("relu")(x)
                
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.25)(x)

                if FC:
                        x = Flatten()(x)
                        x = Dense(512, kernel_regularizer=l2(reg))(x)
                        #x = Activation("relu")(x)                                              
                        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                        x = Activation("relu")(x)                      
                        x = Dropout(0.50)(x)
                        x = Dense(classes, kernel_regularizer= l2(reg))(x)
                        x = Activation("softmax")(x)
                else: #replace FC layers with CONV layers
                        kernel_size = (width//4, height//4)
                        x = Conv2D(512, kernel_size)(x) # input data has size 7x7 #DB
                        x = Conv2D(512, (1, 1))(x)
                        #x = Activation("relu")(x)
                        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
                        x = Activation("relu")(x)                                              
                        x = Conv2D(classes, (1, 1))(x)
                        x = Flatten()(x)
                        x = Activation("softmax")(x)


                # create the model
                model = Model(inputs, x, name="miniVggNet")
                return model

        # miniResNet
        @staticmethod
        def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
                # the shortcut branch of the ResNet module should be initialized as the input (identity) data
                shortcut = data
                # the first block of the ResNet module are the 1x1 CONVs
                bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(data)
                act1 = Activation("relu")(bn1)
                conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
                # the second block of the ResNet module are the 3x3 CONVs
                bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(conv1)
                act2 = Activation("relu")(bn2)
                conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False,
                               kernel_regularizer=l2(reg))(act2)
                # the third block of the ResNet module is another set of 1x1 CONVs
                bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
                act3 = Activation("relu")(bn3)
                conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)
                # if we are to reduce the spatial size, apply a CONV layer to the shortcut
                if red:
                        shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False,
                                          kernel_regularizer=l2(reg))(act1)
                # add together the shortcut and the final CONV
                x = add([conv3, shortcut])
                # return the addition as the output of the ResNet module
                return x

        @staticmethod
        def miniResNet(width, height, depth, classes, stages, filters,
                       reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset="cifar"):
                # initialize the input shape to be "channels last" and the
                # channels dimension itself
                inputShape = (height, width, depth)
                chanDim = -1
                # if we are using "channels first", update the input shape
                # and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1
                # set the input and apply BN
                inputs = Input(shape=inputShape, name="conv2d_1_input")
                x = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(inputs)
                # apply a single CONV layer
                x = Conv2D(filters[0], (3, 3), use_bias=False,
                           padding="same", kernel_regularizer=l2(reg))(x)
                # loop over the number of stages
                for i in range(0, len(stages)):
                        # initialize the stride, then apply a residual module
                        # used to reduce the spatial size of the input volume
                        stride = (1, 1) if i == 0 else (2, 2)
                        x = ConvNetFactory.residual_module(x, filters[i + 1], stride,
                                                       chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
                        # loop over the number of layers in the stage
                        for j in range(0, stages[i] - 1):
                                # apply a ResNet module
                                x = ConvNetFactory.residual_module(x, filters[i + 1],
                                                               (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
                # apply BN => ACT => POOL
                x = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(x)
                x = Activation("relu")(x)
                x = AveragePooling2D((8, 8))(x)
                # softmax classifier
                x = Flatten()(x)
                x = Dense(classes, kernel_regularizer=l2(reg))(x)
                x = Activation("softmax")(x)

                # create the model
                model = Model(inputs, x, name="miniResNet")
                # return the constructed network architecture
                return model

        # miniGoogleNet
        @staticmethod
        def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
                # define a CONV => BN => RELU pattern
                x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
                x = BatchNormalization(axis=chanDim)(x)
                x = Activation("relu")(x)
                # return the block
                return x

        @staticmethod
        def inception_module(x, numK1x1, numK3x3, chanDim):
                # define two CONV modules, then concatenate across the
                # channel dimension
                conv_1x1 = ConvNetFactory.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
                conv_3x3 = ConvNetFactory.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
                x = concatenate([conv_1x1, conv_3x3], axis=chanDim)
                # return the block
                return x

        @staticmethod
        def downsample_module(x, K, chanDim):
                # define the CONV module and POOL, then concatenate
                # across the channel dimensions
                conv_3x3 = ConvNetFactory.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
                pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
                x = concatenate([conv_3x3, pool], axis=chanDim)
                # return the block
                return x

        @staticmethod
        def miniGoogleNet(width, height, depth, classes,  FC=True):
                # initialize the input shape to be "channels last" and the
                # channels dimension itself
                inputShape = (height, width, depth)
                chanDim = -1
                # if we are using "channels first", update the input shape
                # and channels dimension
                if K.image_data_format() == "channels_first":
                        inputShape = (depth, height, width)
                        chanDim = 1
                # define the model input and first CONV module
                inputs = Input(shape=inputShape, name="conv2d_1_input")
                x = ConvNetFactory.conv_module(inputs, 96, 3, 3, (1, 1),chanDim)
                # two Inception modules followed by a downsample module
                x = ConvNetFactory.inception_module(x, 32, 32, chanDim)
                x = ConvNetFactory.inception_module(x, 32, 48, chanDim)
                x = ConvNetFactory.downsample_module(x, 80, chanDim)
                # four Inception modules followed by a downsample module
                x = ConvNetFactory.inception_module(x, 112, 48, chanDim)
                x = ConvNetFactory.inception_module(x,  96, 64, chanDim)
                x = ConvNetFactory.inception_module(x,  80, 80, chanDim)
                x = ConvNetFactory.inception_module(x,  48, 96, chanDim)
                x = ConvNetFactory.downsample_module(x, 96, chanDim)
                # two Inception modules followed by global POOL and dropout
                x = ConvNetFactory.inception_module(x, 176, 160, chanDim)
                x = ConvNetFactory.inception_module(x, 176, 160, chanDim)
                x = AveragePooling2D((7, 7))(x)
                x = Dropout(0.5)(x)
                # softmax classifier
                x = Flatten()(x)
                x = Dense(classes)(x)
                x = Activation("softmax")(x)
                # create the model
                model = Model(inputs, x, name="miniGoogleNet")
                # return the constructed network architecture
                return model
