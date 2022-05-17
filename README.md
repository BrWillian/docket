# Desafio Docket

Bem-vindo/a ao repositorio contendo esse desafio.  

## :scroll: Sobre o teste
O teste consiste em codificar um modelo de machine learning, com finalidade de criar uma proposta de classificação de amostras de documentos RG, CPF e CNH.

## :clipboard: Requisitos

### Requisitos obrigatórios
Este repositorio além dos requisitos proposto pelo desafio contêm outros requisitos necessarios para estruturação do projeto.

* Documentação
  * Instruções de instalação, inicialização e testes
  * Descrição sobre as tecnologias utilizadas no projeto
* Desenvolvimento do dataset de imagens artificiais relacionadas aos documentos.
* Treinar um modelo capaz de classificar os documentos relacionados.
* Validação da solução.
  * Caracteristicas do modelo.
  * Testes

## :computer: Utilização

### Modo de utilização
* Realizar instalação das depêndencias \
`$ pip3 install -r requeriments.txt`
* Após instalação das depêndencias podemos consultar a mini documentação guia \
`python3 main.py --help`

<pre>
usage: main.py [-h] [--treino] [--no-treino] [--diretorio_dataset DIRETORIO_DATASET] [--image_size IMAGE_SIZE] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--loss_function LOSS_FUNCTION] [--metrics METRICS] [--verbose VERBOSE] [--devices DEVICES] [--teste]
               [--no-teste] [--path_model PATH_MODEL] [--diretorio_teste DIRETORIO_TESTE] [--single_teste SINGLE_TESTE]
               [--download_pesos DOWNLOAD_PESOS]

optional arguments:
  -h, --help            show this help message and exit
  --treino              Modo treino (realizar treino) (default: False)
  --no-treino           Modo treino (não realizar treino) (default: False)
  --diretorio_dataset DIRETORIO_DATASET
                        coloque o caminho do seu dataset de acordo com a documentação disponivel no README.md (default: dataset/)
  --image_size IMAGE_SIZE
                        coloque o tamanho da imagem para treino. (default: (150, 150))
  --batch_size BATCH_SIZE
                        Tamanho do batch_size (default: 32)
  --epochs EPOCHS       Número de epocas para treino. (default: 10)
  --loss_function LOSS_FUNCTION
                        função de erro utilizada no treino. (todas disponiveis de acordo com a documentação do tensorflow (default:
                        categorical_crossentropy)
  --metrics METRICS     metricas utilizadas para efeturar treino. (todas disponiveis de acordo com a documentação do tensorflow. (default:
                        ['accuracy'])
  --verbose VERBOSE     Nivel de verbosidade para treino (0, 1, 2). (default: 1)
  --devices DEVICES     Caso treino for com multigpu's apontar quais dispositivos utilizar. Exemplo: devices=["/gpu:0", "/gpu:1", "/gpu:2"]
                        (default: None)
  --teste               Modo teste (realiza teste com batch de imagens ou single). (default: False)
  --no-teste            Modo teste (não realiza teste com batch de imagens ou single). (default: False)
  --path_model PATH_MODEL
                        Caminho onde está disponivel o modelo para teste. (default: weights/BrazilianID_01_0.0837.h5)
  --diretorio_teste DIRETORIO_TESTE
                        Caminho onde está disponivel diretorio para imagens de teste, de acordo com a documentação disponivel no README.md
                        (passar diretorio de imagens para predição) (default: documentos/)
  --single_teste SINGLE_TESTE
                        Realiza predict em apenas uma imagem. (passar caminho da imagem) (default: None)
  --download_pesos DOWNLOAD_PESOS
                        Faz download dos pesos de forma automatica. (default: True)
</pre>

### Exemplos de uso
* Primeiramente deve ser efetuado o download dos pesos padrões.\
`$ python3 main.py --no-treino --no-teste --download_pesos=True`

* Link para download de forma manual. 
[Link para download](https://drive.google.com/file/d/1nYNIq7tX8RtP49Y7czGfETtzgTAPFZIH/view?usp=sharing)

* Para realizar treino segue abaixo codigo exemplo.\
`$ python3 main.py --treino --diretorio_dataset=dataset/ --epochs=10 --batch_size=64`
* O código acima realiza o treino básico, porém podem ser alteradas todas caracteriscas do treino conforme o --help mostra.
* O modelo a seguir e dotado de 8 classes distintas sendo elas: <pre>'CNH_Aberta', 'CNH_Frente', 'CNH_Verso', 'CPF_Frente', 'CPF_Verso', 'RG_Aberto','RG_Frente', 'RG_Verso'</pre>
* Para realizar teste temos 2 funções disponíveis um para teste em massa onde terá o returno de uma lista de listas contendo o nome da imagem infêrida é o resultado: exemplo <pre>[['imagem1.jpg', 'RG_Frente'],['image2.jpg', 'RG_Verso'] ...]</pre> E outra para inferência de apenas uma imagem.
* A função para infêrencia em massa será necessario passar o diretorio e todas imagens contendo no diretorio e subdiretorios serão inferidas segue exemplo de uso:\
`$ python3 main.py --no-treino --teste --diretorio_teste=documentos/ --path_model=weights/BrazilianID_07_0.5606.h5`
* E para inferência de apenas uma imagem segue o exemplo de uso:\
`$ python3 --no-treino --teste --single_teste='documentos/CNH/eu_frente.jpg'`

## :bulb: Exemplos

### Exemplos de imagens
* As imagens utilizadas neste projeto foram geradas de forma artificial, portando todas imagens apresentadas neste projeto são ficticias ou ofucadas.
* Segue abaixo alguns modelos de imagens utilizadas quanto no treino e no teste do modelo.

> \
> <img src="https://github.com/BrWillian/docket/blob/main/documentos/CNH/00003644_in.jpg?raw=true" hspace="50" width="250">
> <img src="https://github.com/BrWillian/docket/blob/main/documentos/CNH/00007257_in.jpg?raw=true" width="250"><p>
> <center>CNH Frente Verso</center>


> \
> <img src="https://github.com/BrWillian/docket/blob/main/documentos/CPF/00010912_in.jpg?raw=true" width="150" hspace="75" >
> <img src="https://github.com/BrWillian/docket/blob/main/documentos/RG/00025937_in.jpg?raw=true" width="250">
> <p align="center">CPF Frente  RG Verso</p>[
]()
* Todo o dataset possui orientações variadas (vertical, horizontal) e pode ser requisitado para análise.
* Na pasta documentos/ segue algumas imagens exemplo que foram utilizadas em teste.

### Recomendações
* Pelo dataset possui apenas imagens rachuradas, é recomendado rachurar para melhor performance. conforme abaixo
> \
> <img src="https://github.com/BrWillian/docket/blob/main/documentos/CNH/eu_frente.jpg?raw=true" width="225" hspace="225"><p>
> <p align="center"> CNH Frente</p>
Boa sorte! :boom:

---
