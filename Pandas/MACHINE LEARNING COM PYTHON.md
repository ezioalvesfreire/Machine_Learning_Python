
# MACHINE LEARNING COM PYTHON FAZENDO USO DA BIBLIOTECA PANDAS

<p>A principal tarefa para engenheiros quando desejam trabalhar com Machine Learning e realizar análise de dados para determinar a viabilidade de encontrar tendências, e então criar um "pipeline" efieciete para treinar o modelo. Este processo envolve
usar bibliotecas como NumPy e Pandas para manipular dados, em conjunto com frameworks como TensorFlow/Keras/PyTorch.</p>

<p>O pandas é muito útil para trabalhar com dataframes, (dataframes é basicamente uma tabela que contém linhas e colunas).</p>


![Dataframe](img/alunos_df-1.png)
<p>Neste tutorial utilizarei o Jupyter Notebook como ferramenta para desenvolver e executar os códigos.</p>

## Guia de instalação
<p>Para que você possa estar acompanhando este tutorial e também executando em sua máquina este passo a passo, certifique se de que já tenha o Python instalado em seu computador, para consultar se o Python já está instalado, abra o Prompt de comando CMD do windows, e execute o comando python, com isso deverá aparecer Python e sua versão, no MAC ou Linux acesse o Terminal e ao invés de digitar apenas Python, digite Pytonh3.
</p>

<p>caso não tenha o Python instalado, acesse o site oficial do Python, <https://www.python.org/> 
em seguida clique em download e será guiado para página de download  da versão, conforme a ilustra a imagem a seguir, para instalar a última versão clique no botão amarelo conforme aponta a seta 02, então o download será inicializado, após concluir o download, vá até a pasta Python que foi baixada (possivelmente estará na pasta downloads), e clique no executável para efetuar a instalação, todo o processo consiste basicamente clicar  em install, next, next… e finish.</p>

![Download_Python](img/Download_Python.png)

<p>Para instalar o Jupyter notebook, basta instalar o Anacondas que já está com Jupyter Notebook integrado.
Instalando o Anaconda, para isso acesse o site do Anaconda, <https://anaconda.org/>
</p>
<p>Clique em Download anaconda </p>

![Download_Anaconda](img/Download_Anaconda-1.png)
<p>Após clicar em download Anaconda, que irá te guiar para página seguinte,  role a página até o rodapé e selecione a versão que mais se enquadra com o seu sistema operacional, no meu caso estou usando o windows 64-bit</p>

![Download_Anaconda](img/Download_Anaconda-2.png)

<p>Após efetuado o download, vá até a pasta baixada e clique no executável para efetuar a instalação.</P>

![Download_Anaconda](img/Download_Anaconda-3.png)
<br>
![Download_Anaconda](img/Download_Anaconda-4.png)

<p>Leia e aceite o termo de licença.</P>

![Download_Anaconda](img/Download_Anaconda-5.png)
<br>
![Download_Anaconda](img/Download_Anaconda-6.png)

![Download_Anaconda](img/Download_Anaconda-7.png)

![Download_Anaconda](img/Download_Anaconda-8.png)



![Download_Anaconda](img/Download_Anaconda-9.png)

<p>Aguarde a conclusão do Download.</P>

![Download_Anaconda](img/Download_Anaconda-10.png)

![Download_Anaconda](img/Download_Anaconda-11.png)
<p>Clicque em finish, para finalizar a instalação.</P>
![Download_Anaconda](Download_Anaconda-12.png)
<p>Usuário  Windows, após instalação do Anaconda, vá para o campo de pesquisa, no canto inferior esquerdo e digite Anaconda navigator e clique em: Anaconda Navigator para abri-lo.</p>
<p>Veja a quantidade de ferramentas disponíveis do pacote anacondas e ainda contamos com uma série de bibliotecas já embutida.</p>

![Ferramentas_Anaconda](img/ferramentas-1.png)

![Ferramentas_Anaconda](img/ferramentas-2.png)

![Ferramentas_Anaconda](img/ferramentas-3.png)
<p>A ferramenta que iremos utilizar para executar o nosso código será o Jupyter Notebook
com Anaconda Navigator aberto basta clicar em Launch no quadro Jupyter notebook, conforme ilustra imagem abaixo:</p>

![Jupyter](img/jupyter-1.png)

<p>outra opção para usuário do Windows abrir o Jupyter, basta buscar por Jupyter Notebook na area de pesquisa no canto inferior esquerdo e selecionar Jupyter Notebook, e também pode optar por abrir o prompt de comando do windows e digitar Jupyter notebook e teclar enter, ao ambiente para desenvolvimento estará aberto conforme ilustra a figura abaixo.</p>

![Jupyter](img/jupyter-2.png)

<p>Em seguida clique em New, no canto superior direito, e depois clique em Python 3 (ipykemel).</p>

![Jupyter](img/jupyter-3.png)

<p>para execução dos nossos código será necessario a instalação da biblioteca Pandas, portanto abra o prompt de comando CMD ou equivalente e execute o comando >>> pip install pandas</p>

![Jupyter](img/install_pandas.png)



## Execução dos códigos

<p>Pronto já podemos iniciar o nosso trabalho com o código.</p>

### Obs: Para execução dos códigos, teremos como base instruções do canal Didática Tech
### Acesse o site: [Didática Tech](https://didatica.tech/)
### Canal [Didática Tech no YouTube](https://www.youtube.com/watch?v=an54pc9BW4I)


-------------------------------------------------------------------------------------------------

### Criando um dataframe a partir de um dicionário python
<p>nota: dicionário é um conjunto de dados contendo chave e valor, chave sempre a esquerda dos : (dois pontos), e logicamente o valor a direita dos dois pontos, obs: o valor pode receber uma lista, seja strings, inteiros, floats etc.</p>
Obs2: dicionário está sempre entre chaves{}, lista entre colchetes[] e tuplas entre parênteses().
</br>

```python
import pandas as pd  #importação da biblioteca Pandas do Python
```

```python

alunos = {'Nome':['joão', 'Maria', 'Bernardo', 'Paulo', 'Lorena'],    # Dicionário alunos
            'Nota':[5, 7.2, 5.7, 10, 8.9],
            'Aprovado':['não', 'sim', 'não', 'sim', 'sim']}
```
```python
print(alunos) #imprima o dicionário alunos, para verificar como ficou

```
```python
{'Nome': ['joão', 'Maria', 'Bernardo', 'Paulo', 'Lorena'], 'Nota': [5, 7.2, 5.7, 10, 8.9], 'Aprovado':   ['não', 'sim', 'não', 'sim', 'sim']} # resultado da impressão

```

```python
alunos_df = pd.DataFrame(alunos)  # criação de um DataFrame a partir do dicionário alunos sendo atribuiído a variável alunos_df
```
```python
print(alunos_df)
```
```python
       Nome  Nota Aprovado
0      joão   5.0      não
1     Maria   7.2      sim
2  Bernardo   5.7      não
3     Paulo  10.0      sim
4    Lorena   8.9      sim
```


## Efetivamente trabalhando com Machine Learning