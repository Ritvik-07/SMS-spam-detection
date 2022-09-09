#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv(r'D:\data sets\spam.csv', encoding = 'Windows-1252')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df = df.drop(columns = df[['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']], axis  = 1)


# In[6]:


df.head()


# In[7]:


df.rename(columns = {'v1' : 'target', 'v2' : 'text'}, inplace = True)


# In[8]:


df.head()


# In[9]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


# In[10]:


df['target'] = encoder.fit_transform(df['target'])


# In[11]:


df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df = df.drop_duplicates(keep = 'first')


# In[15]:


df.duplicated().sum()


# In[16]:


df.shape


# In[17]:


df['target'].value_counts()


# In[18]:


import plotly
import plotly.graph_objects as go
import plotly.express as px


# In[19]:


import matplotlib.pyplot as plt 


# In[20]:


fig = go.Figure(data=[go.Pie(labels=['ham', 'spam'],
                             values=df['target'].value_counts())])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,
                  marker=dict(colors=['#210070', 'red'], line=dict(color='#000000', width=2)))
fig.show()


# In[21]:


import nltk


# In[22]:


df['num_character'] = df['text'].apply(len)


# In[23]:


#number of words
df['text'].apply(lambda x : nltk.word_tokenize(x))


# In[24]:


df['num_words'] = df['text'].apply(lambda x : len(nltk.word_tokenize(x)))


# In[25]:


df.head()


# In[26]:


#number of sentences
df.text.apply(lambda x : len(nltk.sent_tokenize(x)))


# In[27]:


df['num_sentences'] = df.text.apply(lambda x : len(nltk.sent_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df.num_character.describe()


# In[31]:


df.head()


# In[32]:


df[['num_character', 'num_words', 'num_sentences']].describe().style.background_gradient(cmap='PuBu')


# In[33]:


#ham
df[df['target'] == 0][['num_character', 'num_words', 'num_sentences']].describe().style.background_gradient(cmap='PuBu')


# In[34]:


#spam
df[df['target'] == 1][['num_character', 'num_words', 'num_sentences']].describe().style.background_gradient(cmap='Blues')


# In[35]:


import seaborn as sns


# In[36]:


plt.figure(figsize = (12, 8))
sns.histplot(df[df['target'] == 0]['num_character'], color = '#210070', label = 'ham')
sns.histplot(df[df['target'] == 1]['num_character'], color = 'red', label = 'spam')
plt.legend()


# In[37]:


plt.figure(figsize = (12, 8))
sns.histplot(df[df['target'] == 0]['num_words'], color = '#213970', label = 'ham')
sns.histplot(df[df['target'] == 1]['num_words'], color = 'red', label = 'spam')
plt.legend()


# In[38]:


plt.figure(figsize = (12, 6))
sns.pairplot(df, hue = 'target', height = 5)
plt.show()


# In[39]:


df.corr()


# In[40]:


plt.figure(figsize = (7, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'Reds')


# # Data preprocessing
# 1)lowercase
# 2)tokenization
# 3)Removing special character
# 4)Removing special characters and stopwords
# 5)steming

# In[41]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[42]:


from nltk.corpus import stopwords


# In[43]:


import string
string.punctuation


# In[44]:


def transformed_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    list = []
    for i in text:
        if i.isalnum():
            list.append(i)
            
    text = list[:]
    list.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            list.append(i)
            
    text = list[:]
    list.clear()
    
    for i in text:
        list.append(ps.stem(i))
            
    
    return ' '.join(list)


# In[45]:


transformed_text('I am a Boy?, and i Love Dancing.')


# In[46]:


df['text'][10]


# In[47]:


transformed_text('I\'m gonna be home soon and i don\'t want to talk about this stuff anymore tonight, k? I\'ve cried enough today')


# In[48]:


df['transformed_text'] = df['text'].apply(transformed_text)


# In[49]:


df.head()


# # word cloud in spam and ham

# In[50]:


from wordcloud import WordCloud
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = 'black')


# In[51]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))


# In[52]:


plt.figure(figsize = (12, 6))
plt.imshow(spam_wc)


# In[53]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep = " "))


# In[54]:


plt.figure(figsize = (12, 6))
plt.imshow(ham_wc)


# In[55]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text']:
    for word in msg.split():
        spam_corpus.append(word)


# In[56]:


spam_corpus


# In[57]:


len(spam_corpus)


# In[58]:


from collections import Counter 
Counter(spam_corpus).most_common(30)


# In[59]:


import warnings
warnings.filterwarnings("ignore")


# In[60]:


plt.figure(figsize = (12, 6))
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0], pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# In[61]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text']:
    for word in msg.split():
        ham_corpus.append(word)


# In[62]:


len(ham_corpus)


# In[63]:


pd.DataFrame(Counter(ham_corpus).most_common(30))


# In[64]:


plt.figure(figsize = (12, 6))
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0], pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation = 'vertical')
plt.show()


# # Model building

# In[65]:


# Text Vectorization
# using Bag of Words
df.head()


# In[66]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[67]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[68]:


X.shape


# In[69]:


y = df['target'].values


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[72]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[74]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[182]:


gnb.fit(X_train,y_train)
y_pred1 = gnb.predict(X_test)
print('---accuracy Score---')
print(accuracy_score(y_test,y_pred1))
print('\n')
print('---confusion Matrics---')
print(confusion_matrix(y_test,y_pred1))
print('\n')
print('---precision Score---')
print(precision_score(y_test,y_pred1))


# In[75]:


mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
print('---accuracy Score---')
print(accuracy_score(y_test,y_pred2))
print('\n')
print('---confusion Matrics---')
print(confusion_matrix(y_test,y_pred2))
print('\n')
print('---precision Score---')
print(precision_score(y_test,y_pred2))


# In[143]:


bnb.fit(X_train,y_train)
y_pred3 = bnb.predict(X_test)
print('---accuracy Score---')
print(accuracy_score(y_test,y_pred3))
print('\n')
print('---confusion Matrics---')
print(confusion_matrix(y_test,y_pred3))
print('\n')
print('---precision Score---')
print(precision_score(y_test,y_pred3))


# In[144]:


# tfidf --> MNB


# In[145]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# In[146]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)


# In[147]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}


# In[148]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[149]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[150]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[151]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[152]:


performance_df


# In[153]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1


# In[154]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[164]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_max_ft_3000':accuracy_scores,'Precision_max_ft_3000':precision_scores}).sort_values('Precision_max_ft_3000',ascending = False)


# In[156]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_scaling':accuracy_scores,'Precision_scaling':precision_scores}).sort_values('Precision_scaling',ascending=False)


# In[165]:


new_df = performance_df.merge(temp_df,on='Algorithm')


# In[166]:


new_df_scaled = new_df.merge(temp_df,on='Algorithm')


# In[167]:


temp_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy_num_chars':accuracy_scores,'Precision_num_chars':precision_scores}).sort_values('Precision_num_chars',ascending = True)


# In[168]:


new_df_scaled.merge(temp_df,on='Algorithm')


# In[161]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[162]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[163]:


voting.fit(X_train,y_train)


# In[169]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[170]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[171]:


from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[172]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[76]:


import pickle
pickle.dump(tfidf,open('vectorizer2.pkl','wb'))
pickle.dump(mnb,open('model2.pkl','wb'))


# In[87]:


df['text'][2]


# In[86]:


df['target'][2]


# In[ ]:




