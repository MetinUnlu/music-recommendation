This project is prepared as the final project of Mining Massive Dataset course in Verona University

----
# Recommenedation Systems

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
- Nvidia Merlin Recommender System
- Collaborative Filtering
- Content-Based Recommendation System

Collaborative Filtering and Content-Based Recommendation systems are well known recommendation solutions that can be interpretted. New deep-learning based solutions are becoming more popular each day, this systems can work with massive data efficiently, have great recommendation accuracy however one disadvantage of this method is that the solution is a black box.
For the Mining Massive Dataset course, since interpretation is crucial, the implemented method is Content-Based Recommendation System and Collaborative Filtering. However, in concise manner, Nvidia Merlin Framework, which has industrial level capabilities in recommendation using deep learning is presented below.

### Nvidia Merlin Recommender 
