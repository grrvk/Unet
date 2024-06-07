import argparse
import os
from src.dataset import dir_path, get_all_datasets
from src.unet import dice_coef, build_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, CSVLogger


def prepare_log_folder(log_dir='log'):
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def set_callbacks(log_dir):
    checkpointer = ModelCheckpoint(filepath=f'{log_dir}/checkpoint.keras',
                                   verbose=1,
                                   save_best_only=True)

    csv_logger = CSVLogger(f'{log_dir}/log.csv', append=True, separator=';')
    return [checkpointer, csv_logger]


def train(path: str, LR: float, loss: str, epochs: int, batch_size: int):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = get_all_datasets(path)
    log_dir = prepare_log_folder()

    optimizer = Adam(learning_rate=LR)
    metrics = [dice_coef, tf.keras.metrics.MeanIoU(num_classes=2)]

    unet_model = build_model()
    unet_model.compile(optimizer=optimizer,
                       loss=loss,
                       metrics=metrics)

    callbacks = set_callbacks(log_dir)
    history = unet_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
                             verbose=1, validation_data=(X_val, Y_val),
                             callbacks=callbacks)

    unet_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    results = unet_model.evaluate(X_test, Y_test, batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run train')
    parser.add_argument('--dataset_path', metavar='str', type=dir_path,
                        help='General path to the dataset')
    parser.add_argument('--LR', metavar='N', type=float,
                        help='Learning rate')
    parser.add_argument('--loss', metavar='str', type=str,
                        help='Loss function')
    parser.add_argument('--epochs', metavar='N', type=int,
                        help='Number of epochs to run train')
    parser.add_argument('--batch_size', metavar='N', type=int,
                        help='Size of a batch')

    args = parser.parse_args()
    train(args.dataset_path, args.LR, args.loss, args.epochs, args.batch_size)