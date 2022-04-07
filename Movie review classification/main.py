import pandas
from imblearn.under_sampling import  RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

df_review = pandas.read_csv('IMDB Dataset.csv')

rus = RandomUnderSampler(random_state=0)
df_review_bal, df_review_bal['sentiment']=rus.fit_resample(df_review[['review']], df_review['sentiment'])

train, test = train_test_split(df_review_bal, test_size=0.33, random_state=42)

train_x, train_y = train['review'], train['sentiment']
test_x, test_y = test['review'], test['sentiment']

tfidf = TfidfVectorizer(stop_words='english')
train_x_vector = tfidf.fit_transform(train_x)

pandas.DataFrame.sparse.from_spmatrix(train_x_vector, index=train_x.index, columns=tfidf.get_feature_names_out())

test_x_vector = tfidf.transform(test_x)

svc = SVC(kernel='linear')
svc.fit(train_x_vector, train_y)

print("Input review here:")
review = input()

prediction = svc.predict(tfidf.transform([review]))

print("\nPrediction: ", prediction[0])