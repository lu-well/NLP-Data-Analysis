# import required libraries and packages
import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# read csv file
data = pd.read_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
                   r"\Datasetprojpowerbi.csv", index_col=False)

# check what the file looks like
print(data.head())

# remove irrelevant columns, convert everything to lower case, remove punctuation
data = data['Reports'].str.lower()
data = data.apply(lambda x: re.sub(r'[^\w\s]', '', x))

# check that these updates worked
print(data.head())

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
data = data.apply(remove_stop_words)

# check to see that the function worked
print(data.head())

# find unique words and how often these appear in the data
word_counts = Counter(word for row in data for word in row.split())

# load this information into a dataframe and sort by frequency
df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
df = df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

# export to csv to check for other stop words to be removed
df.to_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
          r"\word_counts.csv", index=False)

# import csv of additional stop words discovered
other_stop_words = pd.read_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
                               r"\stopwords.csv", index_col=False)

# convert to list
osw_data = other_stop_words['stopwords'].values.astype(str).tolist()
print(osw_data)

# define rule to remove additional stop words
stop_words_removal = r'\b(?:' + '|'.join(map(re.escape, osw_data)) + r')\b'

# use rule to remove additional stop words
clean_data = data.str.replace(stop_words_removal, '', case=False, regex=True)

# export to csv and print data head to check if it worked
clean_data.to_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
                  r"\cleandata.csv", index=False)
print(clean_data.head())
