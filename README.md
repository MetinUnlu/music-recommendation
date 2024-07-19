This project is prepared as the final project of Mining Massive Dataset course in Verona University

----
# Recommendation Systems

Recommendation systems plays a much more crucial role in many new service platforms. Movie recommendation, e-commerce item recommendation, music recommendation and even small short video recommendations in platforms such as Instagram and TikTok.
To increase user experience, recommendation systems search and brings what the user may enjoy, deriving from what they enjoyed previously.

Social media platforms, such as Instagram, leverage **vast amounts** of user-generated data to power their recommendation systems. Every user interaction, from viewing stories to engaging with short-form videos, contributes to a rich dataset. These interactions might include:

1. Story engagement (viewing or skipping)
2. Video watch patterns (repeat views, duration)
3. Content appreciation (likes, comments)
4. Social sharing behaviors

A typical user session, lasting just a few minutes, can produce hundreds of data points. When scaled to a user base of billions, this results in an enormous volume of data.
The challenge for these platforms lies in efficiently collecting and processing this data. The goal is to transform these interactions into meaningful insights that drive personalized content recommendations for each user.

In this project we will cover:
- NVIDIA Merlin Recommender System
- Collaborative Filtering
- Content-Based Recommendation System

Collaborative Filtering and Content-Based Recommendation systems are well known recommendation solutions that can be interpreted. New deep-learning-based solutions are becoming more popular each day, these systems can work with massive data efficiently, have great recommendation accuracy however one disadvantage of this method is that the solution is a black box.
Since interpretation is crucial for the Mining Massive Dataset course, the implemented method is Content-Based Recommendation System and Collaborative Filtering. However, concisely, the Nvidia Merlin Framework, which has industrial-level capabilities in recommendation using deep learning is presented below.

### NVIDIA Merlin Recommender 

NVIDIA Merlin is a scalable and GPU-accelerated solution for recommendation problems, making it easy to build recommender systems from end to end. NVİDİA Merlin also provides working case-studies in a pipelined format where the input for data can be adjusted to any recommendation-related data, which are provided in notebook format with docker environment. NVIDIA Merlin has components, each one is open-source library. One of them is NVTabular, this library can process and easily manipulate terabyte-size datasets that are used to train deep learning based recommender systems. The library offers a high-level API that can define complex data transformation workflows. And two libraries which are Merlin Models and Merlin Systems provides tools for training and serving with Triton Inference Server, which is an open-source inference-serving software that streamlines AI inferencing.

**[NVTabular](https://github.com/NVIDIA-Merlin/NVTabular)**
**[Merlin Models](https://github.com/NVIDIA-Merlin/models)**
**[Merlin Systems](https://github.com/NVIDIA-Merlin/systems)**

And example is shown below:


```python
import merlin.models.tf as mm
from merlin.io.dataset import Dataset

train = Dataset(PATH_TO_TRAIN_DATA)
valid = Dataset(PATH_TO_VALID_DATA)

model = mm.DLRMModel(
    train.schema,                                                   # 1
    embedding_dim=64,
    bottom_block=mm.MLPBlock([128, 64]),                            # 2
    top_block=mm.MLPBlock([128, 64, 32]),
    prediction_tasks=mm.BinaryClassificationTask(train.schema)      # 3
)

model.compile(optimizer="adagrad", run_eagerly=False)
model.fit(train, validation_data=valid, batch_size=1024)
eval_metrics = model.evaluate(valid, batch_size=1024, return_dict=True)
```

In this example the Merlin Models library is used to train RecSys architectures like [DLRM](http://arxiv.org/abs/1906.00091) for a dataset.

NVIDIA Merlin uses Dask and RAPIDS cuDF. DASK**[Dask](https://www.dask.org/)**<br> is a Python library for parallel and distributed computing. It allows computing to scale your data workflows from a single machine to a cluster, multiple CPU cores can be used to process computation in parallel and it also gives easy capability to adjust RAM dedicated to each solver. **[RAPIDS cuDF](https://github.com/rapidsai/cudf)**<br> is library of NVIDIA which has the main capabilities of pandas Dataframe but powered with GPU loading and computing. cuDF leverages libcudf, a blazing-fast C++/CUDA dataframe library and the Apache Arrow columnar format to provide a GPU-accelerated pandas API.

For our subject of course, the solution is necessary to be interpretable, thus we move on to next recommendation system.

### Collaborative Filtering

Collaborative filtering is an information retrieval technique that suggests items to users by analyzing the interaction of other users who share similar preferences and behaviors (URL1). In another word, collaborative filtering uses the system that groups the users who show the same behavior and use general group characteristics to recommend items to a target user. 

The main problems in collaborative filtering are:
Collaborative filtering, while powerful for recommendation systems, faces several key challenges:

1. **Cold Start Problem**: This occurs when there is insufficient data about new users or items, making it difficult to provide accurate recommendations¹.

2. **Sparsity**: Often, user-item interaction matrices are sparse, meaning most users have rated only a few items. This lack of data can hinder the system's ability to find similar users or items².

3. **Scalability**: As the number of users and items grows, the computational resources required to process and generate recommendations increase significantly.

4. **Popularity Bias**: The system tends to favor popular items, which can lead to less diverse recommendations and overlook niche items that might be of interest to users³.

In this study, different methods available in literature and open-source have been tested.

#### 1. Koren Neighborhood Model

To facilitate global optimization, Koren proposed a neighborhood model with global weights independent of a specific user. The similarity weight is solved via optimization rather than correlation matrix. Unlike some other models that used user-specific interpolation weights, this model uses global weights. These weights are learned from the data through optimization.⁴

This approach emphasizes the influence of missing ratings and allows for more flexible and accurate predictions.

The code and design is explained in the notebook. [[View Notebook]](https://github.com/MetinUnlu/music-recommendation/blob/master/Collaborative-notebooks/koren-Neighborhood%20.ipynb)

Koren Neighborhood Model Mean Squared Error(MSE): 0.117

#### 2. Surprise Library

Surprise is a Python library containing collaborative-filtering-based recommendation systems. The library is well-built and implementation is really easy and straightforward. 

Provide various ready-to-use [prediction
  algorithms](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)
  such as [baseline
  algorithms](https://surprise.readthedocs.io/en/stable/basic_algorithms.html),
  [neighborhood
  methods](https://surprise.readthedocs.io/en/stable/knn_inspired.html), matrix
  factorization-based (
  [SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD),
  [PMF](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note),
  [SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp),
  [NMF](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)),
  and [many
  others](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html).
  Also, various [similarity
  measures](https://surprise.readthedocs.io/en/stable/similarities.html)
  (cosine, MSD, pearson...) are built-in.

In Surprise library it is possible to see Koren neighborhood model as well, which is shown in the notebook. The SVD⁵ and kNNBaseline⁴

The code and design is explained in the notebook. [[View Notebook]](https://github.com/MetinUnlu/music-recommendation/blob/master/Collaborative-notebooks/surp-recommender.ipynb)

**SVD method with Surprise Library MSE: 0.1107**

**kNN Baseline(Koren Neighborhood Model) MSE: 0.1208**

### 3. LightFM Recommender

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

The advantage of using LightFM is the matrix factorization that can work with sparse data efficiently, giving us the ability of using all the data we have available. On previous methods, only the 0.27% percent of data could be used, which corresponds to 10000 row of interactions. With LightFM we can work with all 3651141 rows. The LightFM uses its own Dataset object to load this data and stored data is in this format:

<692376x28597 sparse matrix of type '<class 'numpy.int32'>'
	with 3651141 stored elements in COOrdinate format>

### 4. [Implicit](https://github.com/benfred/implicit) Library

The final and main method used for the dataset is the building recommender model with Implicit Library. Similar to LightFM, Implicit also uses sparse matrixes to store and compute the data. Implicit offers many different recommender models, has easy-to-use and customize code structure, making it the main choice for the project.

Implicit provides fast Python implementations of several different popular recommendation algorithms for
implicit feedback datasets:

 * Alternating Least Squares as described in the papers [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) and [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf).

 * [Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618.pdf).

 * [Logistic Matrix Factorization](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

 * Item-Item Nearest Neighbour models using Cosine, TFIDF or BM25 as a distance metric.

All models have multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS and BPR models both have custom CUDA
kernels - enabling fitting on compatible GPU's. Approximate nearest neighbours libraries such as [Annoy](https://github.com/spotify/annoy), [NMSLIB](https://github.com/searchivarius/nmslib)
and [Faiss](https://github.com/facebookresearch/faiss) can also be used by Implicit to [speed up
making recommendations](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/).

The tested methods are AlternatingLeastSquares, BayesianPersonalizedRanking, LogisticMatrixFactorization.

Each method was also tested with different parameters. The resulting scores are like the following:

| Method   | Model                        | Training Parameters                                                                                          | Precision | Map    | NDCG   | AUC   |
|----------|------------------------------|-------------------------------------------------------------------------------------------------------------|-----------|--------|--------|-------|
| Implicit | AlternatingLeastSquares      | factors=50, iterations=1, regularization=0.01, use_gpu=implicit.gpu.HAS_CUDA                                 | 0.03463   | 0.01734| 0.02554| 0.52038|
| Implicit | AlternatingLeastSquares      | factors=50, iterations=30, regularization=0.01, use_gpu=implicit.gpu.HAS_CUDA                                | 0.0709    | 0.03791| 0.05182| 0.53452|
| Implicit | BayesianPersonalizedRanking  | factors=50, iterations=100, regularization=0.01                                                             | 0.1336    | 0.0636 | 0.0885 | 0.5627 |
| Implicit | BayesianPersonalizedRanking  | factors=50, iterations=100, regularization=0.01                                                             | 0.16506   | 0.0854 | 0.11548| 0.58127|
| Implicit | BayesianPersonalizedRanking  | factors=50, iterations=1200, regularization=0.01                                                            | 0.1799    | 0.092  | 0.12448| 0.58753|
| Implicit | LogisticMatrixFactorization  | factors=50, iterations=20, regularization=0.01                                                              | 0.0094847 | 0.0039269| 0.006447| 0.50585|
| Implicit | LogisticMatrixFactorization  | factors=50, iterations=350, regularization=0.01                                                             | 0.03328   | 0.01167| 0.0183 | 0.51384|

Best evaluation is achieved with Bayesian Personalized Ranking, 1200 Iterations.


references:
----
¹ Shao, Y., & Xie, Y. H. (2019, November). Research on cold-start problem of collaborative filtering algorithm. In Proceedings of the 3rd International Conference on Big Data Research (pp. 67-71).

² Huang, Z., Chen, H., & Zeng, D. (2004). Applying associative retrieval techniques to alleviate the sparsity problem in collaborative filtering. ACM Transactions on Information Systems (TOIS), 22(1), 116-142.

³ Zhu, Z., He, Y., Zhao, X., Zhang, Y., Wang, J., & Caverlee, J. (2021, March). Popularity-opportunity bias in collaborative filtering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining (pp. 85-93).

⁴ Yehuda Koren. Factor in the neighbors: scalable and accurate collaborative filtering. 2010. URL: https://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf.

⁵ Francesco Ricci, Lior Rokach, Bracha Shapira, and Paul B. Kantor. Recommender Systems Handbook. 1st edition, 2010.
