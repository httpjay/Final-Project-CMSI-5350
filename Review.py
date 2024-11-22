import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
import pickle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score





file_path = r'.\data\fake reviews dataset.csv'  

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("Data loaded successfully!")  
else:
    print(f"File not found: {file_path}")


plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='label', palette='Set2', legend=False)
plt.title('distribution of target classes')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

print(data.isnull().sum())
print(data['label'].value_counts())



import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Supprimer les balises HTML
    text = re.sub(r'[^\w\s]', '', text)  # Supprimer les caractères spéciaux
    text = text.lower()  # Transformer en minuscules
    return text

data['cleaned_text'] = data['text_'].apply(clean_text)
print("The data has been successfully cleaned and stored in the 'cleaned_text' column.")


data['text_length'] = data['cleaned_text'].apply(len)


plt.figure(figsize=(10, 6))
sns.histplot(data['text_length'], bins=30, kde=True)
plt.title('text length distribution')
plt.xlabel('Length of the text')
plt.ylabel('frequency')
plt.show()


nltk.download('stopwords') 
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words] 
    return tokens
data['tokens'] = data['cleaned_text'].apply(tokenize)


from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_top_words(data, category):
    tokens = [token for tokens_list in data[data['label'] == category]['tokens'] for token in tokens_list]
    word_counts = Counter(tokens)
    most_common = word_counts.most_common(10)
    
    words, counts = zip(*most_common)
    df_words = pd.DataFrame({'word': words, 'count': counts})
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_words, x='count', y='word', palette='Blues_d')
    plt.title(f'Most Frequent in Category {category}')
    plt.xlabel('frequency')
    plt.ylabel('Words')
    plt.show()

for label in data['label'].unique():
    plot_top_words(data, label)


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(data['cleaned_text']).toarray()  
y = data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.metrics import classification_report

# y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred))

# def train_classifier(clf,X_train,y_train,X_test,y_test):
#     clf.fit(X_train,y_train)
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test,y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted')
    
#     return accuracy,precision

# accuracy_scores = []
# precision_scores = []

# for name,clf in clfs.items():    
#     current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
#     accuracy_scores.append(current_accuracy)
#     precision_scores.append(current_precision)

# performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)

# performance_df.reset_index(drop = True)

"------------------------------------------------------------------"
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

classifier=MultinomialNB()

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], normalize=False)
plt.show()



from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier(n_iter=50)
linear_clf.fit(X_train, y_train)
pred = linear_clf.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE Data', 'REAL Data'])
plt.show()



"------------------------------------------------------------------"
# This we will use for voting classifer!!!!

# mnb = MultinomialNB()
# lrc = LogisticRegression(solver='liblinear', penalty='l1')
# rfc = RandomForestClassifier(n_estimators=50, random_state=2)

# voting = VotingClassifier(estimators=[('LR', lrc), ('nb', mnb), ('RF', rfc)],voting='soft')

# voting.fit(X_train,y_train)

# y_pred = voting.predict(X_test)
# print("Accuracy",accuracy_score(y_test,y_pred))
# print("Precision",precision_score(y_test,y_pred))

# estimators = [('LR', lrc), ('nb', mnb), ('RF', rfc)]
# final_estimator = RandomForestClassifier(n_estimators=50, random_state=2)

# clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)

# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy",accuracy_score(y_test,y_pred))
# print("Precision",precision_score(y_test,y_pred))

# pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
# pickle.dump(mnb,open('model.pkl','wb'))

"----------------------------------------------------------------------"