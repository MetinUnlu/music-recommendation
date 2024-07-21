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

**A small information on dataset**

The dataset is taken from Kaggle, [Million Song Dataset + Spotify + Last.fm](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm?select=Music+Info.csv)

The interaction data of user listening history is normalized for each users music playcounts. After normalization if user has more than one listening count for a music count, the remaining music listenings with one listening count will normalized to zero. To decrease the dimension of data and since 0 would not deliver an information for models, all the rows where normalized playcount is 0. This decrease the row count of dataset from 9.711.301 to 3.651.141. Note that in this process no unique user is removed. This data manipulation can be observed in the [notebook](https://github.com/MetinUnlu/music-recommendation/blob/master/Collaborative-notebooks/dataset_invest.ipynb).

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

#### 3. LightFM Recommender

LightFM is a Python implementation of a number of popular recommendation algorithms for both implicit and explicit feedback, including efficient implementation of BPR and WARP ranking losses. It's easy to use, fast (via multithreaded model estimation), and produces high quality results.

It also makes it possible to incorporate both item and user metadata into the traditional matrix factorization algorithms. It represents each user and item as the sum of the latent representations of their features, thus allowing recommendations to generalise to new items (via item features) and to new users (via user features).

The advantage of using LightFM is the matrix factorization that can work with sparse data efficiently, giving us the ability of using all the data we have available. On previous methods, only the 0.27% percent of data could be used, which corresponds to 10000 row of interactions. With LightFM we can work with all 3651141 rows. The LightFM uses its own Dataset object to load this data and stored data is in this format:

<692376x28597 sparse matrix of type '<class 'numpy.int32'>'
	with 3651141 stored elements in COOrdinate format>

#### 4. [Implicit](https://github.com/benfred/implicit) Library

The final and main method used for the dataset is the recommender model with Implicit Library. Similar to LightFM, Implicit also uses sparse matrixes to store and compute the data, giving us opportunities to use all the 3 million rows. Implicit offers many different recommender models, has an easy-to-use and customizable code structure, making it the main choice for the project.

Implicit provides fast Python implementations of several different popular recommendation algorithms for
implicit feedback datasets:

 * Alternating Least Squares as described in the papers [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) ⁶ and [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf) ⁷.

 * [Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618.pdf) ⁸.

 * [Logistic Matrix Factorization](https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf)

 * Item-Item Nearest Neighbour models using Cosine, TFIDF or BM25 as a distance metric.

All models have multi-threaded training routines, using Cython and OpenMP to fit the models in
parallel among all available CPU cores.  In addition, the ALS and BPR models both have custom CUDA
kernels - enabling fitting on compatible GPU's. Approximate nearest neighbours libraries such as [Annoy](https://github.com/spotify/annoy), [NMSLIB](https://github.com/searchivarius/nmslib)
and [Faiss](https://github.com/facebookresearch/faiss) can also be used by Implicit to [speed up
making recommendations](https://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/).

The tested methods are Alternating Least Squares, Bayesian Personalized Ranking, and Logistic Matrix Factorization.

Each method was also tested with different parameters. The resulting scores are like the following:

| **Model**                   | **Training Parameters**                                                       | **Precision** | **Map**   | **NDCG** | **AUC** |
|-----------------------------|-------------------------------------------------------------------------------|---------------|-----------|----------|---------|
| AlternatingLeastSquares     | factors=50, iterations=1, regularization=0.01, use_gpu=implicit.gpu.HAS_CUDA  | 0.03463       | 0.01734   | 0.02554  | 0.52038 |
| AlternatingLeastSquares     | factors=50, iterations=30, regularization=0.01, use_gpu=implicit.gpu.HAS_CUDA | 0.0709        | 0.03791   | 0.05182  | 0.53452 |
| BayesianPersonalizedRanking | factors=50, iterations=100, regularization=0.01                               | 0.1336        | 0.0636    | 0.0885   | 0.5627  |
| BayesianPersonalizedRanking | factors=50, iterations=100, regularization=0.01                               | 0.16506       | 0.0854    | 0.11548  | 0.58127 |
| BayesianPersonalizedRanking | factors=50, iterations=1200, regularization=0.01                              | 0.1799        | 0.092     | 0.12448  | 0.58753 |
| LogisticMatrixFactorization | factors=50, iterations=20, regularization=0.01                                | 0.0094847     | 0.0039269 | 0.006447 | 0.50585 |
| LogisticMatrixFactorization | factors=50, iterations=350, regularization=0.01                               | 0.03328       | 0.01167   | 0.0183   | 0.51384 |


Best evaluation is achieved with Bayesian Personalized Ranking, 1200 Iterations.

#### 4.1 Implicit Method Implementation

```python
music_path="../data/music_info.csv"
df_path="../data/normalized_filtered_user_listening.csv"
    
# load user artists matrix
user_artists = load_user_artists(Path(df_path))

# instantiate artist retriever
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(Path(music_path),Path(df_path))

# instantiate ALS using implicit
implict_model = implicit.als.AlternatingLeastSquares(
    factors=50, iterations=30, regularization=0.01,  use_gpu=implicit.gpu.HAS_CUDA
)
train, test= train_test_split(user_artists, train_percentage=0.8)

# instantiate recommender, fit, and recommend
recommender = ImplicitRecommender(artist_retriever, implict_model)
recommender.fit(train)
```

The code snippet of an example of how to load data and train using Impilicit is shown above. The user-item interaction and item datasets are loaded using method of the library. When the data is loaded correctly, the implicit library model traning methods can easily be defined and trained. [Full code](https://github.com/MetinUnlu/music-recommendation/blob/master/music-recommendation/jupyter_data.ipynb). 

The data loading may differ from dataset to dataset, thus for this project, modification in data loading method was done. [Modified Implicit Data Loader](https://github.com/MetinUnlu/music-recommendation/blob/master/music-recommendation/implicitMusic.py)

After the training of the model, it can be saved and loaded easily:

```python
# Saving
recommender.implicit_model.save('800-BPS')

# Loading:
implict_model = implicit.cpu.bpr.BayesianPersonalizedRanking().load('800-BPS.npz')
```

In my modified code, recommender returns: music_id, artists, tracks, scores

And an example returned recommender results are shown below (
```python
music_id, artists, tracks, scores = recommender.recommend(1413, user_artists, n=5)
print(music_id)
    # print results
for artist, track, score in zip(artists, tracks, scores):
    print(f"{artist} by {track}: {score}")
```
Output:
```markdown
[13450 18488  1296  4494   960]
Love You Lately by Daniel Powter: 4.25557279586792
L'impasse by Coralie Clément: 4.092816352844238
Live Like We're Dying by Kris Allen: 3.987139940261841
What's Left of Me by blessthefall: 3.9387805461883545
From Where You Are by Lifehouse: 3.9146549701690674
```

This model gives really accurate recommendations. However in interaction dataset the number of users with three or less music interaction is more than 50% of dataset. For such users, the recommendation will be biased and will have cold-start problem. To remedy this, this model is used in uniform with a simple Content-Based Recommender.

#### Content-Based Recommender: kNN Cosine Similarity

For Content-Based recommender, a really simple method is to use kNN cosine similarity which can return most similar items for given reference item with distance value as well. 

Initial trial was to create complete Cosine Similarity matrix of music dataset using sklearn cosine_similarity, however since dataset contains a lot of music, required RAM to process complete matrix is 19 GB. [Notebook](https://github.com/MetinUnlu/music-recommendation/blob/master/Content-Based-notebooks/CossineSimilarity.ipynb) However complete matrix is not required and not even efficient. Using sklearn.neighbors NearestNeighbors method with metrix='cosine' and number of neighbors set to 10, much smaller, required version of the matrix is created. [Notebook](https://github.com/MetinUnlu/music-recommendation/blob/master/Content-Based-notebooks/kNN-CossineSimilarity.ipynb)

From the NearestNeighbors method we return distances and indices as csv file to use it together with Collaborative Filter.

### Collaborative Filtering and Content-Based Recommendation

From the Implicit method we saved our model and can be loaded easily. Similarly the distance and indices for similarty matrix from Content-Based model is saved as csv.

Objective is to use it together in uniformity. We can directly get recommendation from Collaborative Filtering, however for Content-Based I have added the following rule:

```python
def kNN_recommend(N,df,df_music,indices,distances,n_recommendations=5):
    all_recommendations={}
    tracks= df[df['user_id']==users[N]]['track_id']
    weights=df[df['user_id']==users[N]]['normalized_playcount']
    for i in range(len(tracks)):
        track=df_music[df_music['track_id']==tracks.iloc[i]].index[0]
        recommended_ind, recommended_dist=get_recommendations(track,indices,distances,n_recommendations)
        recommended_dist=recommended_dist/weights.iloc[i]
        for n in range(len(recommended_ind)):
            all_recommendations[recommended_ind[n]]=recommended_dist[n]
    recommendation_list=heapq.nsmallest(n_recommendations, all_recommendations, key=all_recommendations.get)
    return(recommendation_list)
```

We will use the function above to return recommendation for Content-Based model, here we load the all the unique music the given user have listened and how many times they have listened the music. Since I have already normalized the playcount for each user uniquely, dataset contains normalized_playcount which act as weight. Then for each unique music the user have listened, it returns the 5 (or n) neighboring music distances and each distance value is divided by the weight of given unique music. Note that the weight, normalized playcount, is between 0 and 1, which will increase the distance when divided. This creates a dictionary containing all the distances for all the unique music the user listened. Finally, the function returns from this dictionary *n* smallest distance-valued music list. [Notebook](https://github.com/MetinUnlu/music-recommendation/blob/master/music-recommendation/Collab-and-Content-recommender.ipynb)

To combine both collaborative and content-based method, we can use thresholds. In my [recommendation system](https://github.com/MetinUnlu/music-recommendation/blob/master/music-recommendation/Collab-and-Content-recommender.ipynb), I have used the following:

- If the unique listened music count is <=3: Return 10 recommendation from Content-Based model
- If the unique listened music count is >3 and <=5: Return 5 recommendation from Content-Based model, 5 from Collaborative Filtering Model
- If the unique listened music count is >5: Return 3 recommendation from Content-Based model, 7 from Collaborative Filtering Model

And example output is given below:

```markdown
For user who have listened 5 music

 Recommended Musics:
Some Kinda Love by The Velvet Underground (Collaborative Filtering Based)
Age of Consent by New Order (Collaborative Filtering Based)
I'm Sleeping in a Submarine by Arcade Fire (Collaborative Filtering Based)
Childhood Remembered by Kevin Kern (Collaborative Filtering Based)
Thieves Like Us by New Order (Collaborative Filtering Based)
Kein Mitleid by Eisbrecher (Content Based)
Tostaky (Le Continent) by Noir Désir (Content Based)
Easy Love by MSTRKRFT (Content Based)
Avantasia by Avantasia (Content Based)
Mysterious Skies by ATB (Content Based)
```






references:
----
¹ Shao, Y., & Xie, Y. H. (2019, November). Research on cold-start problem of collaborative filtering algorithm. In Proceedings of the 3rd International Conference on Big Data Research (pp. 67-71).

² Huang, Z., Chen, H., & Zeng, D. (2004). Applying associative retrieval techniques to alleviate the sparsity problem in collaborative filtering. ACM Transactions on Information Systems (TOIS), 22(1), 116-142.

³ Zhu, Z., He, Y., Zhao, X., Zhang, Y., Wang, J., & Caverlee, J. (2021, March). Popularity-opportunity bias in collaborative filtering. In Proceedings of the 14th ACM International Conference on Web Search and Data Mining (pp. 85-93).

⁴ Yehuda Koren. Factor in the neighbors: scalable and accurate collaborative filtering. 2010. URL: https://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf.

⁵ Francesco Ricci, Lior Rokach, Bracha Shapira, and Paul B. Kantor. Recommender Systems Handbook. 1st edition, 2010.

⁶ Hu, Y., Koren, Y., & Volinsky, C. (2008, December). Collaborative filtering for implicit feedback datasets. In 2008 Eighth IEEE International Conference on data mining (pp. 263-272). Ieee.

⁷ Takács, G., Pilászy, I., & Tikk, D. (2011, October). Applications of the conjugate gradient method for implicit feedback collaborative filtering. In Proceedings of the fifth ACM conference on Recommender systems (pp. 297-300).

⁸ Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2012). BPR: Bayesian personalized ranking from implicit feedback. arXiv preprint arXiv:1205.2618.

⁹ Johnson, C. C. (2014). Logistic matrix factorization for implicit feedback data. Advances in Neural Information Processing Systems, 27(78), 1-9.
