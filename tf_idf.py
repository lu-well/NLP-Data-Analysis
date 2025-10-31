# import required libraries and packages
from data_cleansing import clean_data
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# create vectorizer
tfidf_vectorizer = TfidfVectorizer()

# fit data to tf-idf matrix using vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_data)

# create dataframe from results to read and check it worked
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)
print(tfidf_df)

# generate silhouette score for the bag of words matrix
silhouette_scores = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(tfidf_matrix)
    score = silhouette_score(tfidf_matrix, kmeans.labels_)
    silhouette_scores.append(score)

# plot the results in a graph to find the optimal number of clusters
plt.plot(range(2, 15), silhouette_scores)
plt.show()

# graph shows that the optimal number of clusters is 11
