import os.path

from model import BrazilianIdModel
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MAE, MSE, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy


class TrainBrIdModel:
    """
         @@Author Willian Antunes
        Classe customizada para efetuar treino do modelo.
    """

    def __init__(self, **kwargs):
        super(TrainBrIdModel, self).__init__()
        self._dataset_path = kwargs.get("dataset_directory") if kwargs.get("dataset_directory") else "dataset/"
        self._image_size = kwargs.get("image_size") if kwargs.get("image_size") else (150, 150)
        self._batch_size = kwargs.get("batch_size") if kwargs.get("batch_size") else 32
        self._epochs = kwargs.get("epochs") if kwargs.get("epochs") else 5
        self._loss_func = kwargs.get("loss_function") if kwargs.get("loss_function") else 'categorical_crossentropy'
        self._metrics = kwargs.get("metrics") if kwargs.get("metrics") else ["accuracy"]
        self._verbose = kwargs.get("verbose") if kwargs.get("verbose") else 1
        self._devices = kwargs.get("devices")

    def _load_dataset(self):
        """
            Definições para construação do dataset
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           validation_split=0.25,
                                           fill_mode='nearest',
                                           )

        train_generator = train_datagen.flow_from_directory(
            directory=self._dataset_path,
            target_size=self._image_size,
            batch_size=self._batch_size,
            color_mode='rgb',
            class_mode='categorical',
            subset='training',
            shuffle=True,
        )

        validation_generator = train_datagen.flow_from_directory(
            directory=self._dataset_path,
            target_size=self._image_size,
            batch_size=self._batch_size,
            color_mode='rgb',
            class_mode='categorical',
            subset='validation',
            shuffle=True,
        )

        return train_generator, validation_generator

    def _get_strategy_multigpu(self):
        """
            Lista de dispositivos disponiveis para treino;
            exemplo: devices=["/gpu:0", "/gpu:1", "/gpu:2"]
            conforme a documentação do tensorflow
        """

        if self._devices is not None:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self._devices)

            return mirrored_strategy

        strategy = tf.distribute.MirroredStrategy()
        print('Número de gpus: {}'.format(strategy.num_replicas_in_sync))

        return strategy

    def train_model(self):
        strategy = self._get_strategy_multigpu()
        train_generator, validation_generator = self._load_dataset()
        logdir = "logs/scalars/" + str(datetime.now().strftime("%Y%m%d- %H%M%S"))

        callbacks = [TensorBoard(log_dir=logdir),
                     EarlyStopping(patience=2),
                     ModelCheckpoint(
                         filepath='weights/BrazilianID_{epoch:02d}_{val_loss:.4f}.h5',
                         monitor='val_loss',
                         save_freq='epoch',
                         verbose=self._verbose,
                         save_best_only=True,
                         save_weights_only=False
                     )]

        with strategy.scope():
            model = BrazilianIdModel()

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss=self._loss_func,
                      metrics=self._metrics)

        model.summary()

        model.fit(train_generator, epochs=self._epochs, callbacks=callbacks, validation_data=validation_generator, verbose=self._verbose)


if __name__ == "__main__":
    '''
        Debug......
    '''
    train_model = TrainBrIdModel(dataset_directory='/home/willian/Projects/docket/BID Dataset/',
                                 batch_size=64, epochs=7)
    train_model.train_model()
