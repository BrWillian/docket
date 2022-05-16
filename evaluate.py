import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import BrazilianIdModel
import numpy as np
from tensorflow.keras.preprocessing import image

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class EvaluateBrIdModel(object):
    '''
        @@Author Willian Antunes
        Realiza Avaliação do modelo.
    '''

    def __init__(self, **kwargs):
        super(EvaluateBrIdModel, self).__init__()
        self._model_path = kwargs.get("path_model") if kwargs.get("path_model") else "weights/BrazilianID_weights_b2.h5"
        self._model = BrazilianIdModel()
        self._classes_names = ['CNH_Aberta', 'CNH_Frente', 'CNH_Verso', 'CPF_Frente', 'CPF_Verso', 'RG_Aberto', 'RG_Frente', 'RG_Verso']

    def _load_datagen(self, directory):
        '''
            Carregar a base de imagens.
        '''

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_generator = test_datagen.flow_from_directory(
            directory,
            target_size=(150, 150),
            batch_size=1,
            class_mode=None,
            shuffle=False,
            color_mode='rgb'
        )

        return test_generator

    def get_result_from_directory(self, directory):
        '''
            Carrega o modelo e faz a predição.
        '''
        directory = directory if directory else "documentos/"

        self._model.load_weights(filepath=self._model_path)

        test_datage = self._load_datagen(directory=directory)

        results = self._model.predict(test_datage)

        res_with_classes = []

        for res in results: res_with_classes.append(self._classes_names[np.argmax(res)])

        return res_with_classes

    def get_result_single_image(self, img):
        """
            Exemplo para chama com apenas uma imagem.
            :param img:
            :return: predict
        """
        img = image.load_img(img, target_size=(150, 150))

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        self._model.load_weights(filepath=self._model_path)

        results = self._classes_names[np.argmax(self._model.predict(img))]

        return results


if __name__ == "__main__":
    BrIdModel = EvaluateBrIdModel(path_model='weights/BrazilianID_weights_b.h5')
    result = BrIdModel.get_result_from_directory(directory='/home/willian/PycharmProjects/docket/documentos/')
    #result = BrIdModel.get_result_single_image('/home/willian/PycharmProjects/docket/documentos/CNH_Aberta/aberta.jpg')

    print(result)