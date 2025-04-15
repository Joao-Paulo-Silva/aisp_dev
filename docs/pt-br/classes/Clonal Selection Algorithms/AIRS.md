O AIRS é um algoritmo de classificação inspirado no processo de seleção clonal. A versão
implementada nesta classe é inspirada na sua versão simplificada, o AIRS2, descrito em 
[Brabazon, O’Neill e McGarraghy (2015)](#ref1).
Nesta classe, há uma adaptação para dados de valor real e uma opção secundária para
características binárias.


---

# AIRS (Sistema Imunológico Artificial de Reconhecimento)

Esta classe estende a classe [**Base**](../../advanced-guides/Base%20Classes%20Reference/csa/Classifier.md).

## Construtor AIRS:

A classe ``AIRS`` tem a finalidade de realizar a classificação utilizando metáforas de seleção e expansão clonal.
**Attributes:**
* **n_resources** (``float``): Total amount of available resources. Defaults to 10.
* **rate_clonal** (``float``): Maximum number of possible clones of a class. This quantity is multiplied by (cell stimulus * rate_hypermutation) to define the number of clones. Defaults to 10.
* **rate_hypermutation** (``int``): The rate of mutated clones derived from rate_clonal as a scalar factor. Defaults to 0.75.
* **affinity_threshold_scalar** (``float``): Normalized affinity threshold. Defaults to 0.75.
* **k** (``int``): The number of K nearest neighbors that will be used to choose a label in the prediction. Defaults to 10.
* **max_iters** (``int``): Maximum number of interactions in the refinement process of the ABR set exposed to aᵢ. Defaults to 100.
* resource_amplified (``float``): Resource consumption amplifier is multiplied with the incentive to subtract resources. Defaults to 1.0 without amplification.
* **metric** (Literal["manhattan", "minkowski", "euclidean"]): Way to calculate the distance between the detector and the sample:
    * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: \
    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
    * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: \
    ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
    * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: 
    ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|). 

    **Defaults to "Euclidean".**
    
* **algorithm** (Literal["continuous-features", "binary-features"]): [description]. Defaults to "continuous-features".
* seed (int): Semente para a geração randômica dos valores nos detectores. Defaults to None.

- ``**kwargs``:
        - p (``float``): Este parâmetro armazena o valor de ``p`` utilizada na distância \
            de Minkowski. O padrão é ``2``, o que significa distância euclidiana normalizada. \
            Diferentes valores de p levam a diferentes variantes da distância de Minkowski \
            [saiba mais](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).


**Outras variáveis iniciadas:**

* *cells_memory* (``dict``): Esta variável armazena uma lista de células de memória por classe.
* *affinity_threshold* (``dict``): Define o limiar de afinidade entre os antígenos.
* *classes* (``npt.NDArray``): lista de classes de saída.

---

### Função fit(...)

A função ``fit(...)`` gera os detectores para os não próprios com relação às amostras:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```
Nela é realizado o treinamento de acordo com ``X`` e ``y``, usando o método de Sistema Imunológico Artificial de Reconhecimento (SIAR).

**Os parâmetros de entrada são:**
* ``X``: array com as características das amostras com **N** amostras (linhas) e **N** características  (colunas), normalizados para valores entre [0, 1]. 
* ``y``: array com as classes de saídas disposto em **N** amostras que são relacionadas ao ``X``.
* ``verbose``: boolean com valor default ``True``, determina se o feedback da geração dos detectores será imprimido.

*Retorna a instância da classe.*

---

### Função predict(...)

A função ``predict(...)`` realiza a previsão das classes utilizando as células de memória geradas:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**O parâmetro de entrada:**
 
* ``X``: array com as características para a previsão, com **N** amostras (Linhas) e **N** colunas.

**Retorna:** 
* ``C``: Um array de previsão com as classes de saída para as características informadas. 
* ``None``: se não houver detectores.

---

### Função score(...):

A função "score(...)" calcula a precisão do modelo treinado por meio da realização de previsões e do cálculo da acurácia.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

retorna a acurácia, do tipo ``float``.

---

## Métodos privados

### Função _refinement_ABR(...):

A função "_refinement_ABR(...)" faz o processo de refinamento do conjunto ABR ate que o valor médio da estimulação seja maior que o limite definido (``affinity_threshold_scalar``)

```python
def _refinement_ABR(self, ai: npt.NDArray, c_match: _Cell, abr_list: List[_ABR]) -> _Cell:
```

Parameters:
* c_match (``_Cell``): Célula com a maior estimulação com relação a aᵢ
* abr_list (``List[_ABR]``): Conjunto ABR.


Retorna a célula (_Cell) com maior estimulação de ABR

---

### Função _cells_affinity_threshold(...):

A função "_cells_affinity_threshold(...)" calcula o limite de afinidade com base na afinidade média entre instâncias de treinamento, onde aᵢ e aⱼ são um par de antígenos, e a afinidade é medida pela distância (Euclidiana, Manhattan, Minkowski, Hamming).

```python
def _cells_affinity_threshold(self, antigens_list: npt.NDArray):
```

**Seguindo a fórmula:**

$$
\text{affinity}_{\text{threshold}} = \frac{
\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{affinity}(a_i, a_j)}{n(n-1)/2}
$$

Parameters:
* antigens_list (``NDArray``): Lista de antígenos de treinamento.

---

### Função _affinity(...):

A função "_affinity(...)" calcula o estimulo entre dois vetores usando métricas.

```python
def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
```

Parameters:
* u (``npt.NDArray``): Coordenadas do primeiro ponto.
* v (``npt.NDArray``): Coordenadas do segundo ponto.

Retorna a taxa de estimulo entre os vetores.

---

### Função _init_memory_c(...):

A função "_init_memory_c(...)" inicializa as células de memória escolhendo `n_antigens_selected` aleatoriamente da lista de antígenos de treinamento.

```python
def _init_memory_c(self, antigens_list: npt.NDArray) -> List[_Cell]:
```

Parameters:
* antigens_list (``NDArray``): Lista de antígenos de treinamento.



---

### Função __slice_index_list_by_class(...):

A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que está sendo treinada:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Retorna um dicionario com as classes como chave e os índices em ``X`` das amostras.

# Classes Auxiliares: ``_Cell`` e ``_ABR``

## Classe _Cell:

Esta classe representa uma célula-B de memoria.

### Construtor: 

Parameters:
* size (``Optional[int]``): O número de características do vetor. Se `vector` for `None`, um vetor aleatório é gerado. Defaults to None.
* vector (``Optional[npt.NDArray]``): Um vetor de características da célula.  Defaults to None.
* algorithm (``Literal["continuous-features", "binary-features"]``): O tipo de algoritmo para amostras contínuas ou binárias. Defaults to "continuous-features".

---

### Função hyper_clonal_mutate(...):

A função "hyper_clonal_mutate(...)" realiza a clonagem de N características a partir das características de uma célula, gerando um conjunto de vetores mutados.

Parameters:
* n (``int``): Número de clones a serem gerados a partir de mutações na célula original.

```python
def hyper_clonal_mutate(self, n: int) -> npt.NDArray
```

Retorna um array contendo N vetores mutados da célula original.

---

## Classe _ABR (Herda a classe _Cell)

Esta classe representa indivíduos do conjunto de células reconhecedoras (ABR), herda características de uma célula-B, adicionando o consumo de recursos

### Construtor: 

 Um membro do conjunto de ABR, herda características de uma `_Cell` com adição do consumo de recursos.

Parameters:
* size (``Optional[int]``): O número de características do vetor. Se `vector` for `None`, um vetor aleatório é gerado. Defaults to None.
* vector (``Optional[npt.NDArray]``): Um vetor de características da célula. Defaults to None.
* stimulation (``Optional[float]``): A taxa de estímulo da célula antígenos. Defaults to None.
* algorithm (``Literal["continuous-features", "binary-features"]``): O tipo de algoritmo para amostras contínuas ou binárias. Defaults to "continuous-features".

---

### Função set_resource(...):

A função "set_resource(...)" consome os recursos e retorna a quantidade de recursos restantes após o consumo.

Parameters:
* resource (```float```): Quantidade inicial de recursos.
* amplified (``float``): Amplificador para o consumo de recursos pela célula. É multiplicado pelo estímulo da célula. O valor padrão é 1.

```python
def set_resource(self, resource: float, amplified: float = 1) -> float:
```

Retorna a quantidade de recursos restantes após o consumo.

---

# References

<br id='ref1'/>

> 1. BRABAZON, Anthony; O’NEILL, Michael; MCGARRAGHY, Seán. Natural Computing Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8. Disponível em: http://dx.doi.org/10.1007/978-3-662-43631-8.

<br id='ref2'/>