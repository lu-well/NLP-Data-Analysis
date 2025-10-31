# import required libraries and packages
from tf_idf import tfidf_matrix
from tf_idf import tfidf_vectorizer
from sklearn.decomposition import LatentDirichletAllocation

# apply LDA with 11 topics
lda = LatentDirichletAllocation(n_components=11, random_state=42)
lda.fit(tfidf_matrix)

# show which topics exist and which words are connected to them
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx + 1}:")
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]  # Top 7 words
    print(" ".join(top_words))
