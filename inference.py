import argparse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from src.dataset import get_image


def load_model(path='log/checkpoint.keras'):
    model = tf.keras.models.load_model(path)
    return model


def inference(image_path):
    model = load_model()
    image = np.array(get_image(image_path, (128, 128, 3)))
    image = np.expand_dims(image, 0)
    prediction = model.predict(image)
    prediction_mask = np.argmax(prediction, axis=3)[0, :, :]

    plt.title('Prediction')
    plt.axis('off')
    plt.imshow(prediction_mask, cmap='gray')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--image_path', metavar='str', type=str,
                        help='Path to the image')

    args = parser.parse_args()
    inference(args.image_path)
