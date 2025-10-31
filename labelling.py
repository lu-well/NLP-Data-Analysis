# import required libraries and packages
import pandas as pd

# upload original dataset
original_data = pd.read_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
                            r"\Datasetprojpowerbi.csv", index_col=False)

# list each topic and its keywords
topics = {
    'Technology': ['access', 'limited', 'academic', 'software', 'technology', 'complete', 'difficult'],
    'Student Support': ['students', 'university', 'resources', 'international', 'provide', 'feel', 'support'],
    'Work-life Balance': ['academic', 'time', 'work', 'responsibilities', 'commitments', 'balance', 'workload'],
    'Canteen Options': ['options', 'offer', 'cantine', 'cafeteria', 'vegetarian', 'vegan', 'affordable'],
    'Mental Health': ['health', 'mental', 'care', 'affordable', 'insurance', 'afford', 'concerned'],
    'Teaching': ['online', 'classes', 'professors', 'difficult', 'inperson', 'availability', 'lack'],
    'Resources': ['materials', 'accessing', 'finding', 'difficulty', 'databases', 'hard', 'coursework'],
    'Future Opportunities': ['job', 'opportunities', 'career', 'field', 'internship', 'internships', 'market'],
    'Finances': ['financial', 'pay', 'aid', 'worried', 'student', 'medical', 'expenses'],
    'Food Prices': ['food', 'campus', 'dining', 'hall', 'quality', 'prices', 'high'],
    'Study Stress': ['stress', 'causing', 'lot', 'anxiety', 'workload', 'college', 'pressure']
}


# create labelling function
def label_topic(complaint):
    for topic, keywords in topics.items():
        if any(keyword in complaint.lower() for keyword in keywords):
            return topic
    return 'Uncategorised'


# apply the labelling function to the dataset
original_data['Topic'] = original_data['Reports'].apply(label_topic)

# print head of dataset to check that it worked
print(original_data.head())

# export to csv to check that labelling worked
original_data.to_csv(r"C:\Users\lucyl\Desktop\University\Course Books\Data Analysis Project"
                     r"\labelled_data.csv", index=False)

# count number and percentages of complaints associated with each topic
count_topics = original_data['Topic'].value_counts()
topic_percentages = original_data['Topic'].value_counts(normalize=True) * 100

# display counts and percentages of topics
result = pd.DataFrame({'Count': count_topics, 'Percentage': topic_percentages})
print(result)
