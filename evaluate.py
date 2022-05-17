import os
import glob
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
        self._model_path = kwargs.get("path_model") if kwargs.get(
            "path_model") else "weights/BrazilianID_01_0.0837.h5"
        self._model = BrazilianIdModel()
        self._classes_names = ['CNH_Aberta', 'CNH_Frente', 'CNH_Verso', 'CPF_Frente', 'CPF_Verso', 'RG_Aberto',
                               'RG_Frente', 'RG_Verso']

    def get_result_from_directory(self, directory):
        '''
            Carrega o modelo e faz a predição utilizando generator do tensorflow para varias imagens.
        '''
        list_of_results = []
        for filename in glob.iglob(directory + '**/**', recursive=True):
            if os.path.isfile(filename) and filename.endswith('.jpg'):
                result = self.get_result_single_image(filename)

                list_of_results.append([filename.split('/')[-1:][0], result])

        return list_of_results

    def get_result_single_image(self, img):
        """
            Exemplo para chama com apenas uma imagem.
            :param img:
            :return: predict
        """
        img = image.load_img(img, target_size=(150, 150), color_mode='rgb')

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        self._model.load_weights(filepath=self._model_path)

        results = self._classes_names[np.argmax(self._model.predict(img))]

        return results


if __name__ == "__main__":
    '''
        Debug......
    '''
    BrIdModel = EvaluateBrIdModel(path_model='weights/BrazilianID.h5')
    result = BrIdModel.get_result_from_directory(directory='documentos/CNH/')
    # result = BrIdModel.get_result_single_image('/home/willian/PycharmProjects/docket/documentos/CNH_Aberta/aberta.jpg')

    print(result)
