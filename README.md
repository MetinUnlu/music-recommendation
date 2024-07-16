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

While recommendation systems have become increasingly sophisticated, they still face several hurdles. Two significant problems are the cold-start issue and data sparsity.
The cold-start problem occurs when the system encounters new users or items with little to no historical data. For new users, the system lacks information about their preferences, making it difficult to provide accurate recommendations. Similarly, newly added items have no interaction history, so the system struggles to suggest them to appropriate users.
Data sparsity is another common issue. Despite the vast amount of data generated, most users interact with only a tiny fraction of available items. This results in a sparse user-item interaction matrix, where most entries are empty. Sparse data can lead to less reliable predictions and lower-quality recommendations.

In this project we will cover:
- NVIDIA Merlin Recommender System
- Collaborative Filtering
- Content-Based Recommendation System

Collaborative Filtering and Content-Based Recommendation systems are well known recommendation solutions that can be interpretted. New deep-learning based solutions are becoming more popular each day, this systems can work with massive data efficiently, have great recommendation accuracy however one disadvantage of this method is that the solution is a black box.
For the Mining Massive Dataset course, since interpretation is crucial, the implemented method is Content-Based Recommendation System and Collaborative Filtering. However, in concise manner, Nvidia Merlin Framework, which has industrial level capabilities in recommendation using deep learning is presented below.

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

#### Koren Neighborhood Model

To facilitate global optimization, Koren proposed a neighborhood model with global weights independent of a specific user. The similarity weight is solved via optimization rather than correlation matrix. Unlike some other models that used user-specific interpolation weights, this model uses global weights (denoted as \( w_{ij} \)) that are independent of specific users. These weights are learned from the data through optimization.

This approach emphasizes the influence of missing ratings and allows for more flexible and accurate predictions.




references:
----
¹ Shao, Y., & Xie, Y. H. (2019, November). Research on cold-start problem of collaborative filtering algorithm. In Proceedings of the 3rd International Conference on Big Data Research (pp. 67-71).
² Huang, Z., Chen, H., & Zeng, D. (2004). Applying associative retrieval techniques to alleviate the sparsity problem in collaborative filtering. ACM Transactions on Information Systems (TOIS), 22(1), 116-142.
³ Zhu, Z., He, Y., Zhao, X., Zhang, Y., Wang, J., & Caverlee, J. (2021, March). Popularity-opportunity bias in collaborative filtering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining (pp. 85-93).
