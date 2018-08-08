
# coding: utf-8

# In[1]:


import os
import fnmatch
from textblob import TextBlob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk import pos_tag, pos_tag_sents
import regex as re
import operator
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import pickle


# In[2]:


#You will first iterate over all the text files and get the absolute path of all the text files 
#through which you will then extract the corresponding labels.


# In[3]:


path = "op_spam_train/"
label = []

configFiles = [os.path.join(subdir, f)
               for subdir, dirs, files in os.walk(path)
               for f in fnmatch.filter(files, '*.txt')]


# In[4]:


len(configFiles)


# In[5]:


configFiles[1]


# In[6]:


#So, from the above output, you can observe that in order to extract the labels we need some kind of filter.
#And for that, you will use Regex also known as a regular expression.


# In[7]:


for i in configFiles:
    c=re.search('(trut|deceptiv)\w',i)
    label.append(c.group())


# In[8]:


labels=pd.DataFrame(label,columns=['Labels'])


# In[9]:


labels.head(10)


# In[10]:


##Fetch all the reviews and append in a list


# In[11]:


review=[]
directory=os.path.join(path)
for subdir,dirs,files in os.walk(directory):
    for file in files:
        if fnmatch.filter(files,"*.txt"):
            f=open(os.path.join(subdir,file),'r')
            a=f.read()
            review.append(a)
reviews=pd.DataFrame(review,columns=['HotelReviews'])


# In[12]:


reviews.head(10)


# In[13]:


##Merge the review dataframe and label dataframe


# In[14]:


result=pd.merge(reviews,labels,right_index=True,left_index=True)
result['HotelReviews']=result['HotelReviews'].map(lambda x:x.lower())


# In[15]:


result.head()


# In[16]:


#Remove stopwords from the Hotel Reviews column


# In[17]:


import nltk
nltk.download('stopwords')


# In[18]:


stop=stopwords.words('english')
result['review_without_stopwords']=result['HotelReviews'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))


# In[19]:


result.head()


# In[20]:


#Extract parts of speech from Hotel Reviews which will be fed as a Feature Input to the model


# In[21]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# In[22]:


def pos(review_without_stopwords):
    return TextBlob(review_without_stopwords).tags


# In[23]:


os=result.review_without_stopwords.apply(pos)
os1=pd.DataFrame(os)


# In[24]:


os1.head()


# In[25]:


os1['pos']=os1['review_without_stopwords'].map(lambda x :" ".join(["/".join(x) for x in x]))


# In[26]:


result =pd.merge(result,os1,right_index=True,left_index=True)
result.head()


# In[27]:


###Training Phase
##Split the Data into two parts 80% train and 20% test data


# In[28]:


review_train,review_test,label_train,label_test=train_test_split(result['pos'],result['Labels'],test_size=0.2,random_state=13)


# In[29]:


#Vectorize the Training and Testing data using TfidfVectorizer


# In[30]:


tf_vect = TfidfVectorizer(lowercase=True,use_idf=True,smooth_idf=True,sublinear_tf=False)
X_train_tf=tf_vect.fit_transform(review_train)
X_test_tf=tf_vect.transform(review_test)


# In[31]:


def svc_param_selection(X,y,nfolds):
    Cs=[0.001,0.01,0.1,1,10]
    gammas=[0.001,0.01,0.1,1]
    param_grid={'C':Cs,'gamma':gammas}
    grid_search=GridSearchCV(svm.SVC(kernel='linear'),param_grid,cv=nfolds)
    grid_search.fit(X,y)
    return grid_search.best_params_


# In[32]:


svc_param_selection(X_train_tf,label_train,5)


# In[33]:


clf=svm.SVC(C=10,gamma=0.001,kernel='linear')
clf.fit(X_train_tf,label_train)
pred=clf.predict(X_test_tf)


# In[34]:


##Save the Tfidf vectorizer and the ML model


# In[35]:


with open('vectorizer.pickle','wb') as fin:
    pickle.dump(tf_vect,fin)


# In[36]:


with open('mlmodel.pickle','wb') as f:
    pickle.dump(clf,f)


# In[37]:


#Load the Tfidf vectorizer and the ML model


# In[38]:


pkl=open('mlmodel.pickle','rb')
clf=pickle.load(pkl)
vec=open('vectorizer.pickle','rb')
tf_vect=pickle.load(vec)


# In[39]:


def test_string(s):
    X_test_tf=tf_vect.transform([s])
    y_predict=clf.predict(X_test_tf)
    return y_predict


# In[40]:


#Predict on the Test Data


# In[41]:


X_test_tf=tf_vect.transform(review_test)
pred=clf.predict(X_test_tf)


# In[42]:


print(metrics.accuracy_score(label_test,pred))


# In[43]:


print(confusion_matrix(label_test,pred))


# In[44]:


print(classification_report(label_test,pred))


# In[45]:


#Test the model with two reviews from Yelp


# In[46]:


test_string("The hotel was good.The room had a 27-inch Samsung led tv, a microwave.The room had a double bed")


# In[47]:


test_string("My family and I are huge fans of this place. The staff is super nice, and the food is great. The chicken is very good, and the garlic sauce is perfect. Ice cream topped with fruit is delicious too. Highly recommended!")


# In[54]:


test_string("very boring mcdoanlds good")


# In[66]:


# Test the Model using different Random States


# In[118]:


review_train, review_test, label_train, label_test = train_test_split(result['pos'],result['Labels'], test_size=0.2,random_state=20)


# In[119]:


X_test_tf=tf_vect.transform(review_test)
pred=clf.predict(X_test_tf)


# In[120]:


print(metrics.accuracy_score(label_test,pred))


# In[121]:


print (confusion_matrix(label_test, pred))


# In[122]:


print (classification_report(label_test, pred))

