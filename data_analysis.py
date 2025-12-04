# import required libraries and packages
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
import numpy as np
from wordcloud import WordCloud
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

# read csv file
original_data = pd.read_csv("Datasetprojpowerbi.csv", index_col=False)  # modify file path for raw data input

# remove irrelevant columns, convert everything to lower case, remove punctuation
data = original_data['Reports'].str.lower()
no_punc_data = data.apply(lambda x: re.sub(r'[^\w\s]', '', x))

# remove standard stop words from nltk library
nltk.download('stopwords')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('english'))


# define function to remove stop words
def remove_stop_words(text):
    words = word_tokenize(text)  # Tokenize the text
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# apply function to remove stop words
semi_clean_data = no_punc_data.apply(remove_stop_words)

# find unique words and how often these appear in the data
word_counts = Counter(word for row in semi_clean_data for word in row.split())

# load this information into a dataframe and sort by frequency
df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
df = df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# export to csv to check for other stop words to be removed
df.to_csv(""\word_counts.csv", index=False)   # modify file path to export csv

# import csv of additional stop words discovered
other_stop_words = pd.read_csv("stopwords.csv", index_col=False)   # modify file path for raw data input

# convert to list
osw_data = other_stop_words['stopwords'].values.astype(str).tolist()

# define rule to remove additional stop words
stop_words_removal = r'\b(?:' + '|'.join(map(re.escape, osw_data)) + r')\b'

# use rule to remove additional stop words
clean_data = semi_clean_data.str.replace(stop_words_removal, '', case=False, regex=True)

# create BoW vectorizer
vectorizer = CountVectorizer()

# fit data to bag of words matrix using vectorizer
bag_of_words = vectorizer.fit_transform(clean_data)

# create dataframe from results to read and check it worked
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=vectorizer.get_feature_names_out())

# generate silhouette score for the bag of words matrix
silhouette_scores = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(bag_of_words)
    score = silhouette_score(bag_of_words, kmeans.labels_)
    silhouette_scores.append(score)

# plot the results in a graph to find the optimal number of clusters
plt.plot(range(2, 15), silhouette_scores)
plt.show()

# create TD-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# fit data to tf-idf matrix using vectorizer
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_data)

# create dataframe from results to read and check it worked
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

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

# graphs show that the optimal number of clusters as per BoW is 3, but this is too few to
# provide a reasonable overview of the student complaints, therefore I will continue
# the analysis using the TF-IDF matrix instead, as the silhouette score for this method is
# 11 which is a more reasonable number of topics

# apply LDA with 11 topics
lda = LatentDirichletAllocation(n_components=11, random_state=42)
lda.fit(tfidf_matrix)

# show which topics exist and which words are connected to them
feature_names = tfidf_vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f"Topic #{topic_idx + 1}:")
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]  # Top 7 words
    print(" ".join(top_words))

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

# as the topics identified by the NMF method are easier to interpret and label as distinct topics,
# further analysis will be continued with the NMF results

# find top words per topic outside the NMF function
top_words_per_topic = []
for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]  # Top 7 words
    top_words_per_topic.extend(top_words)

# join all top words for the 11 topics
all_top_words = ' '.join(top_words_per_topic)

# as the topics identified by the NMF method are easier to interpret and label as distinct topics,
# further analysis will be continued with the NMF results

# plot and visualise a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_top_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# from original data, take the Reports column for calculating coherence score
reports_data = original_data['Reports']

# tokenize the text and set other necessary parameters for coherence model
tokenized_texts = [doc.split() for doc in reports_data]
dictionary = Dictionary(tokenized_texts)
corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

# list all topic keywords
topics = [[["access", "limited", "academic", "software", "technology", "complete", "difficult"]],
          [["students", "university", "resources", "international", "provide", "feel", "support"]],
          [["academic", "time", "work", "responsibilities", "commitments", "balance", "workload"]],
          [["options", "offer", "cantine", "cafeteria", "vegetarian", "vegan", "affordable"]],
          [["health", "mental", "care", "affordable", "insurance", "afford", "concerned"]],
          [["online", "classes", "professors", "difficult", "inperson", "availability", "lack"]],
          [["materials", "accessing", "finding", "difficulty", "databases", "hard", "coursework"]],
          [["job", "opportunities", "career", "field", "internship", "internships", "market"]],
          [["financial", "pay", "aid", "worried", "student", "medical", "expenses"]],
          [["food", "campus", "dining", "hall", "quality", "prices", "high"]],
          [["stress", "causing", "lot", "anxiety", "workload", "college", "pressure"]]]

# loop through topics to calculate and print umass coherence scores
for topics in topics:
    coherence_model_1 = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence='u_mass'
    )

    umass_score = coherence_model_1.get_coherence()

    print(f"UMass Coherence Score: {umass_score:.4f}")

# convert everything to lower case, remove punctuation for labelling
original_data['Reports'] = original_data['Reports'].str.lower()
original_data['Reports'] = original_data['Reports'].str.replace('[^\w\s]', '', regex=True)

# top words for each topic for checking frequencies of keywords in each complaint
topic_1 = ['access', 'limited', 'academic', 'software', 'technology', 'complete', 'difficult']
topic_2 = ['students', 'university', 'resources', 'international', 'provide', 'feel', 'support']
topic_3 = ['academic', 'time', 'work', 'responsibilities', 'commitments', 'balance', 'workload']
topic_4 = ['options', 'offer', 'cantine', 'cafeteria', 'vegetarian', 'vegan', 'affordable']
topic_5 = ['health', 'mental', 'care', 'affordable', 'insurance', 'afford', 'concerned']
topic_6 = ['online', 'classes', 'professors', 'difficult', 'inperson', 'availability', 'lack']
topic_7 = ['materials', 'accessing', 'finding', 'difficulty', 'databases', 'hard', 'coursework']
topic_8 = ['job', 'opportunities', 'career', 'field', 'internship', 'internships', 'market']
topic_9 = ['financial', 'pay', 'aid', 'worried', 'student', 'medical', 'expenses']
topic_10 = ['food', 'campus', 'dining', 'hall', 'quality', 'prices', 'high']
topic_11 = ['stress', 'causing', 'lot', 'anxiety', 'workload', 'college', 'pressure']

# change lists to sets
word_set_1 = set(topic_1)
word_set_2 = set(topic_2)
word_set_3 = set(topic_3)
word_set_4 = set(topic_4)
word_set_5 = set(topic_5)
word_set_6 = set(topic_6)
word_set_7 = set(topic_7)
word_set_8 = set(topic_8)
word_set_9 = set(topic_9)
word_set_10 = set(topic_10)
word_set_11 = set(topic_11)


# functions to count how many words from each topic are in each report
def count_topic_1_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_1 for word in text.split())


def count_topic_2_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_2 for word in text.split())


def count_topic_3_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_3 for word in text.split())


def count_topic_4_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_4 for word in text.split())


def count_topic_5_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_5 for word in text.split())


def count_topic_6_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_6 for word in text.split())


def count_topic_7_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_7 for word in text.split())


def count_topic_8_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_8 for word in text.split())


def count_topic_9_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_9 for word in text.split())


def count_topic_10_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_10 for word in text.split())


def count_topic_11_words(text):
    if not isinstance(text, str):
        return 0
    return sum(word in word_set_11 for word in text.split())


# add information in new columns on dataframe
original_data['Topic 1: Technology'] = original_data['Reports'].apply(count_topic_1_words)
original_data['Topic 2: Student Support'] = original_data['Reports'].apply(count_topic_2_words)
original_data['Topic 3: Work-life Balance'] = original_data['Reports'].apply(count_topic_3_words)
original_data['Topic 4: Food Options'] = original_data['Reports'].apply(count_topic_4_words)
original_data['Topic 5: Mental Health'] = original_data['Reports'].apply(count_topic_5_words)
original_data['Topic 6: Teaching'] = original_data['Reports'].apply(count_topic_6_words)
original_data['Topic 7: Resources'] = original_data['Reports'].apply(count_topic_7_words)
original_data['Topic 8: Future Opportunities'] = original_data['Reports'].apply(count_topic_8_words)
original_data['Topic 9: Finances'] = original_data['Reports'].apply(count_topic_9_words)
original_data['Topic 10: Food Prices'] = original_data['Reports'].apply(count_topic_10_words)
original_data['Topic 11: Study Stress'] = original_data['Reports'].apply(count_topic_11_words)

# select columns from B to L (inclusive)
cols_range = original_data.loc[:, 'Topic 1: Technology':'Topic 11: Study Stress']

# find the column with the highest value in each row, or unclassified if none
main_topic = cols_range.idxmax(axis=1, skipna=True)
unclassified_topic = (cols_range.isna() | (cols_range <= 0)).all(axis=1)
main_topic[unclassified_topic] = 'Unclassified'

original_data['Main Topic'] = main_topic

# drop irrelevant columns
matrix_columns = ["Reports", 'Topic 1: Technology', 'Topic 2: Student Support', 'Topic 3: Work-life Balance',
                  'Topic 4: Food Options', 'Topic 5: Mental Health', 'Topic 6: Teaching', 'Topic 7: Resources',
                  'Topic 8: Future Opportunities', 'Topic 9: Finances', 'Topic 10: Food Prices',
                  'Topic 11: Study Stress', 'Main Topic']

label_matrix_final = original_data[matrix_columns]

count_topics = label_matrix_final['Main Topic'].value_counts()

# plot bar chart of categories and their values within the dataset
count_topics.plot(kind='bar', color='skyblue')
plt.title('Value Counts of Categories')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.show()
