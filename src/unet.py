from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import keras

@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth = 1.0):
    for i in range(2):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / 2
    return total_loss

def double_conv(x, n_filters):
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample(x, n_filters):
    f = double_conv(x, n_filters)
    p = MaxPool2D(2)(f)
    return f, p

def upsample(x, conv_features, n_filters):
    x = Conv2DTranspose(n_filters, 3, 2, padding = "same")(x)
    x = concatenate([x, conv_features])
    x = double_conv(x, n_filters)
    return x

def build_model(img_size = (128, 128, 3), num_classes = 2):
    inputs = Input(shape = img_size)

    f1, p1 = downsample(inputs, 64)
    f2, p2 = downsample(p1, 128)
    f3, p3 = downsample(p2, 256)
    f4, p4 = downsample(p3, 512)

    bottleneck = double_conv(p4, 1024)

    u6 = upsample(bottleneck, f4, 512)
    u7 = upsample(u6, f3, 256)
    u8 = upsample(u7, f2, 128)
    u9 = upsample(u8, f1, 64)

    outputs = Conv2D(num_classes, 1, padding = "same", activation = 'sigmoid')(u9)
    unet_model = tf.keras.Model(inputs, outputs, name = "U-Net")

    return unet_model