# Natural Language Processing of a Complaints Database
In this project a dataset consisting of unstructured text complaints and reports
from students at a university is analysed to extract the most prevalent topics, with 
the use of NLP techniques.

## Dataset
https://www.kaggle.com/datasets/omarsobhy14/university-students-complaints-and-reports

## Data Cleansing
Only the "Reports" column was needed for the analysis and other columns were therefore dropped.
To pre-process the data, all words were converted to lowercase and
punctuation was removed. Stop words from the nltk library were deleted,
then a manual check carried out to remove any further stop words and
characters (e.g. numbers which provided no insight to the complaint sentiment).

## Vectorisation
Bag of words and TF-IDF techniques from the Scikit learn library were used to vectorise the text.
To establish the optimal number of topics, I found the silhouette score after using each method and
compared results. The Bag of Words silhouette score showed that the reports should be clustered into 3 groups,
whereas the result when using the TF-IDF vectorised data was 11 topics.
As 3 topics would be too generic for such a varied dataset, I continued the analysis with the 
TF-IDF results and 11 topics.
<img width="640" height="480" alt="silhouette score" src="https://github.com/user-attachments/assets/f99f1d0a-916a-4d52-b3e2-b8ae35cd463f" />

## Semantic Analysis Techniques
I used Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) to extract 7 keywords from 
the top 11 topics in the vectorised data. Both methods worked satisfactorily, and successfully produced a variety
of topics. The keywords from the LDA results were more ambiguous, seemingly
mixing subjects such as sports and food, or technical and medical issues into one topic, making it more challenging
to classify them into one subject area per topic. As the NMF results produced more clearly defined topics, I used these
to create a word cloud and check how many of the complaints from the original
dataset could be labelled with the keywords from the topics.

## Result Interpretation
The word cloud produced from the keywords across the 11 topics distinctly shows "workload", "difficult" and "academic"
as the largest results. As these words are present in the top 3 most prevalent topics, we can see that
the word cloud is a reasonable representation of the data.
<img width="566" height="296" alt="wordcloud final" src="https://github.com/user-attachments/assets/aa342e6c-a513-4e61-b4ef-a9cc4b7fd643" />


When assigning labels to the complaints, only 2.59% were left uncategorised.
I also used the results to label the reports and generate a bar chart of
how often each topic appears.
<img width="646" height="480" alt="bar graph topic distribution" src="https://github.com/user-attachments/assets/471b16e1-a9f2-4655-8266-e7b432d08313" />

This has limitations as some reports can be classified
under more than one topic, and for this I recommend a manual check to ensure that the topic
chosen is appropriate. The labelling method could also be expanded to include more keywords
or the keywords could be adjusted to lessen the occurrence of ambiguously labelled complaints.
However, "difficult" and "academic" can still be seen as 
rather ambiguous, which indicates that some human intervention to adjust the topic keywords would be beneficial.
For example, the word "professors" is one of the keywords for the teaching topic, but the singular equivalent
"professor" is not. If this were added, and the word "difficult" removed from the technology topic for being too
generalised and applicable to a wide range of topics, we would see a shift in the number of complaints assigned to each
topic. From the data cleansing part of this project, it was also evident that some human input is important in NLP
projects to capture unexpected stop words which are not included in standard stop word sets. This could also be taken
further with spell-checking words, as this can also significantly affect results.

## Conclusion
In this project, I found that TF-IDF and NMF were the most suitable NLP techniques for topic extraction. Although they
worked well, a sense check for topic keywords is a sensible next step, to ensure that complaints are being categorised
as accurately as possible. The NLP analysis should also be repeated regularly to capture data shift and new emerging
topics from future complaints.
