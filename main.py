from __future__ import print_function
from train import TrainBrIdModel
from evaluate import EvaluateBrIdModel
import os
import tensorflow as tf
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--treino', type=bool, help='Modo treino (realizar treino)', default=False, required=True)

    parser.add_argument('--dataset_directory', type=str, help='coloque o caminho do seu dataset'
                                                              ' de acordo com a documentação '
                                                              'disponivel no README.md', default='dataset/')
    parser.add_argument('--image_size', type=tuple, help='coloque o tamanho da imagem para treino.', default=(150, 150))

    parser.add_argument('--batch_size', type=int, help='Tamanho do batch_size', default=32)

    parser.add_argument('--epochs', type=int, help='Número de epocas para treino.', default=10)

    parser.add_argument('--loss_function', type=str, help='função de erro utilizada no treino. '
                                                          '(todas disponiveis de acordo com a documentação do tensorflow',
                        default='categorical_crossentropy')

    parser.add_argument('--metrics', type=list, help='metricas utilizadas para efeturar treino.'
                                                     ' (todas disponiveis de acordo com '
                                                     'a documentação do tensorflow.', default=['accuracy'])

    parser.add_argument('--verbose', type=int, help='Nivel de verbosidade para treino (0, 1, 2).', default=1)

    parser.add_argument('--devices', type=list, help='Caso treino for com multigpu\'s apontar quais dispositivos '
                                                     'utilizar. Exemplo: devices=["/gpu:0", "/gpu:1", "/gpu:2"]')

    parser.add_argument('--teste', type=bool, help='Modo teste (realiza teste com batch de imagens ou single).'
                        , default=True, required=True)

    parser.add_argument('--path_model', type=str, help='Caminho onde está disponivel o modelo para teste.',
                        default='weights/BrazilianID_01_0.0837.h5')

    parser.add_argument('--teste_varias', type=str, help='Caminho onde está disponivel diretorio para imagens de teste,'
                                                         ' de acordo com a documentação disponivel no README.md'
                                                         ' (passar diretorio de imagens para predição)',
                        default='documentos/')

    parser.add_argument('--single_teste', type=bool,
                        help='Realiza predict em apenas uma imagem. (passar caminho da imagem)')

    args = parser.parse_args()

    if args.treino:
        train_model = TrainBrIdModel(dataset_directory=args.dataset_directory, image_size=args.image_size,
                                     batch_size=args.batch_size, epochs=args.epochs, loss_function=args.loss_function,
                                     metrics=args.metrics, verbose=args.verbose, devices=args.devices)
        train_model.train_model()

    if args.teste:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        BrIdModel = EvaluateBrIdModel(path_model=args.path_model)

        if args.teste_varias:
            result = BrIdModel.get_result_from_directory(directory=args.teste_varias)

            print(result)


if __name__ == "__main__":
    main()
