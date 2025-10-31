from data_cleansing import clean_data
from tf_idf import tfidf_matrix
from tf_idf import tfidf_vectorizer
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# create NMF model and fit tfidf matrix to it
nmf_model = NMF(n_components=11, random_state=42)
W = nmf_model.fit_transform(tfidf_matrix)
H = nmf_model.components_

# find top 7 words for each topic
feature_names = tfidf_vectorizer.get_feature_names_out()
n_top_words = 7


# function to display all topics and top words
def display_topics(H, W, feature_names, documents, n_top_words):
    for topic_idx, topic in enumerate(H):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Topic {topic_idx+1}: {', '.join(top_features)}")
        # Show top document for this topic
        top_doc_idx = np.argmax(W[:, topic_idx])
        print(f"  Example doc: {documents[top_doc_idx]}\n")


# show top 7 words for each of the 11 topics
display_topics(H, W, feature_names, clean_data, n_top_words)

# find top words per topic outside the function
top_words_per_topic = []
for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]  # Top 7 words
    top_words_per_topic.extend(top_words)

# join all top words for the 11 topics
all_top_words = ' '.join(top_words_per_topic)

# plot and visualise the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_top_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
