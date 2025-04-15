# AIRS (Artificial Immune Recognition System)

This class extends the [**Base**](../../advanced-guides/Base%20Classes%20Reference/csa/Classifier.md) class.

## AIRS Constructor:

The ``AIRS`` class aims to perform classification using metaphors of selection and clonal expansion.  
**Attributes:**
* **n_resources** (``float``): Total amount of available resources. Defaults to 10.
* **rate_clonal** (``float``): Maximum number of possible clones of a class. This quantity is multiplied by (cell stimulus * rate_hypermutation) to define the number of clones. Defaults to 10.
* **rate_hypermutation** (``int``): The rate of mutated clones derived from rate_clonal as a scalar factor. Defaults to 0.75.
* **affinity_threshold_scalar** (``float``): Normalized affinity threshold. Defaults to 0.75.
* **k** (``int``): The number of K nearest neighbors that will be used to choose a label in the prediction. Defaults to 10.
* **max_iters** (``int``): Maximum number of interactions in the refinement process of the ABR set exposed to aᵢ. Defaults to 100.
* **resource_amplified** (``float``): Resource consumption amplifier is multiplied with the incentive to subtract resources. Defaults to 1.0 without amplification.
* **metric** (Literal["manhattan", "minkowski", "euclidean"]): Way to calculate the distance between the detector and the sample:
    * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression:  
    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
    * ``'minkowski'`` ➜ The calculation of the distance is given by the expression:  
    ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
    * ``'manhattan'`` ➜ The calculation of the distance is given by the expression:  
    ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).  
    Defaults to "Euclidean".
* **algorithm** (Literal["continuous-features", "binary-features"]): [description]. Defaults to "continuous-features".
* **seed** (int): Seed for the random generation of detector values. Defaults to None.

- ``**kwargs``:
    - **p** (``float``): This parameter stores the value of ``p`` used in the Minkowski distance. The default is ``2``, which means normalized Euclidean distance. Different values of p lead to different variants of the Minkowski distance. [Learn more](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).

**Other initialized variables:**
* **cells_memory** (``dict``): This variable stores a list of memory cells by class.
* **affinity_threshold** (``dict``): Defines the affinity threshold between antigens.
* **classes** (``npt.NDArray``): List of output classes.

---

### Function fit(...)

The ``fit(...)`` function generates detectors for the non-owners relative to the samples:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```
It performs the training according to ``X`` and ``y``, using the method Artificial Immune Recognition System (``AIRS``).

**Input parameters:**
* **X**: Array with sample features, with **N** samples (rows) and **N** features (columns), normalized to values between [0, 1].
* **y**: Array with output classes corresponding to **N** samples related to ``X``.
* **verbose**: Boolean, default ``True``, determines if the feedback from the detector generation will be printed.

*Returns the class instance.*

---

### Function predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**Input parameter:**
* **X**: Array with the features for prediction, with **N** samples (rows) and **N** columns.

**Returns:**
* **C**: An array of predictions with the output classes for the given features.
* **None**: If there are no detectors.

---

### Function score(...):

The ``score(...)`` function calculates the accuracy of the trained model by making predictions and calculating the accuracy.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

Returns accuracy as a ``float``.

---

## Private Methods

### Function _refinement_ABR(...):

The function "_refinement_ABR(...)" refines the ABR set until the average stimulation value exceeds the defined threshold (``affinity_threshold_scalar``).

Parameters:
* **c_match** (``_Cell``): Cell with the highest stimulation relative to aᵢ.
* **abr_list** (``List[_ABR]``): ABR set.

```python
def _refinement_ABR(self, ai: npt.NDArray, c_match: _Cell, abr_list: List[_ABR]) -> _Cell:
```

Returns the cell (_Cell) with the highest ABR stimulation.

---

### Function _cells_affinity_threshold(...):

The function "_cells_affinity_threshold(...)" calculates the affinity threshold based on the average affinity between training instances, where aᵢ and aⱼ are a pair of antigens, and affinity is measured by distance (Euclidean, Manhattan, Minkowski, Hamming).  
**Following the formula:**

$$
\text{affinity}_{\text{threshold}} = \frac{
\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} \text{affinity}(a_i, a_j)}{n(n-1)/2}
$$

Parameters:
* **antigens_list** (``NDArray``): List of training antigens.

```python
def _cells_affinity_threshold(self, antigens_list: npt.NDArray):
```

---

### Function _affinity(...):

The function "_affinity(...)" calculates the stimulus between two vectors using metrics.

Parameters:
* **u** (``npt.NDArray``): Coordinates of the first point.
* **v** (``npt.NDArray``): Coordinates of the second point.

```python
def _affinity(self, u: npt.NDArray, v: npt.NDArray) -> float:
```

Returns the stimulus rate between the vectors.

---

### Function _init_memory_c(...):

The function "_init_memory_c(...)" initializes memory cells by randomly selecting `n_antigens_selected` from the list of training antigens.

Parameters:
* **antigens_list** (``NDArray``): List of training antigens.

```python
def _init_memory_c(self, antigens_list: npt.NDArray) -> List[_Cell]:
```

---

### Function __slice_index_list_by_class(...):

The function ``__slice_index_list_by_class(...)`` separates the indices of the rows according to the output class, to iterate over the sample array, only at the positions where the output corresponds to the class being trained:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with classes as keys and indices in ``X`` of the samples.

---

# Auxiliary Classes: ``_Cell`` and ``_ABR``

## _Cell Class:

### Constructor:

Parameters:
* size (``Optional[int]``): The number of features in the vector. If `vector` is `None`, a random vector is generated. Defaults to None.
* vector (``Optional[npt.NDArray]``): A feature vector of the cell. Defaults to None.
* algorithm (``Literal["continuous-features", "binary-features"]``): The type of algorithm for continuous or binary samples. Defaults to "continuous-features".

---

### Function hyper_clonal_mutate(...):

Parameters:
* n (``int``): The number of clones to be generated from mutations in the original cell.

```python
def hyper_clonal_mutate(self, n: int) -> npt.NDArray
```

Returns an array containing N mutated vectors from the original cell.

---

## _ABR Class (Inherits from _Cell)

### Constructor:

Parameters:
* size (``Optional[int]``): The number of features in the vector. If `vector` is `None`, a random vector is generated. Defaults to None.
* vector (``Optional[npt.NDArray]``): A feature vector of the cell. Defaults to None.
* stimulation (``Optional[float]``): The stimulus rate of the antigen cell. Defaults to None.
* algorithm (``Literal["continuous-features", "binary-features"]``): The type of algorithm for continuous or binary samples. Defaults to "continuous-features".

---

### Function set_resource(...):

Parameters:
* resource (```float```) : The initial amount of resources.
* amplified (``float``): Amplifier for resource consumption by the cell. It is multiplied by the cell's stimulus. The default value is 1.

```python
def set_resource(self, resource: float, amplified: float = 1) -> float:
```

Returns the remaining amount of resources after consumption.


---

# References

<br id='ref1'/>

> 1. BRABAZON, Anthony; O’NEILL, Michael; MCGARRAGHY, Seán. Natural Computing Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8. Disponível em: http://dx.doi.org/10.1007/978-3-662-43631-8.

<br id='ref2'/>