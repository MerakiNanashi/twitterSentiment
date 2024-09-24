# %%
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
%matplotlib inline
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
cols = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("/home/mera/Desktop/twitterSentiment/.venv/archive/training.1600000.processed.noemoticon.csv",header=None, names=cols)

# %% [markdown]
# # **Context**
# This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the twitter api . The tweets have been annotated (0 = negative, 4 = positive) and they can be used to detect sentiment .
# 
# **Content**
# 
# It contains the following 6 fields:
# 
# target: the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
# 
# ids: The id of the tweet ( 2087)
# 
# date: the date of the tweet (Sat May 16 23:58:44 UTC 2009)
# 
# flag: The query (lyx). If there is no query, then this value is NO_QUERY.
# 
# user: the user that tweeted (robotickilldozr)
# 
# text: the text of the tweet (Lyx is cool)

# %%
df.head()

# %%
df.sentiment.value_counts()

# %%
df1 = df.copy()
df1.drop(['id', 'date', 'query_string', 'user'], axis=1, inplace=True)

# %%
df1.sentiment.unique()

# %%
df1[df1.sentiment == 0].head(10)

# %%
df1[df1.sentiment == 4].head(10)

# %% [markdown]
# 0 ~ 800000 : negative
# >800000 : positive

# %%
df1['pre_clean_len'] = [len(t) for t in df1.text]

# %%
from pprint import pprint
data_dict = {
    'sentiment':{
        'type':df1.sentiment.dtype,
        'description':'sentiment class - 0:negative, 1:positive'
    },
    'text':{
        'type':df1.text.dtype,
        'description':'tweet text'
    },
    'pre_clean_len':{
        'type':df1.pre_clean_len.dtype,
        'description':'Length of the tweet before cleaning'
    },
    'dataset_shape':df1.shape
}
pprint(data_dict)

# %%
fig, ax = plt.subplots(figsize=(5, 5))
plt.boxplot(df1.pre_clean_len)
plt.show()

# %%
df1[df1.pre_clean_len > 140].head(10)

# %%
from bs4 import BeautifulSoup
example1 = BeautifulSoup(df1.text[279], 'html.parser')
print (example1.get_text())

# %%
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()

pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):
    soup = BeautifulSoup(text, 'html.parser')
    souped = soup.get_text()
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    return (" ".join(words)).strip()

# %%
nums = [0,400000,800000,1200000,1600000]
print ("Cleaning and parsing the tweets...\n")
clean_tweet_texts = []
for i in range(nums[0], nums[4]):
    if( (i+1)%10000 == 0 ):
        print ("Tweets %d of %d has been processed" % ( i+1, nums[1] ) )                                                                   
    clean_tweet_texts.append(tweet_cleaner(df['text'][i]))

# %%
clean_df = pd.DataFrame(clean_tweet_texts,columns=['text'])
clean_df['target'] = df.sentiment
clean_df.head()

# %%
clean_df.to_csv('clean_tweet.csv',encoding='utf-8')
csv = 'clean_tweet.csv'
df2 = pd.read_csv(csv,index_col=0)
df2.head()

# %%
df2.info()

# %%
df2[df2.isnull().any(axis=1)].head()

# %%
np.sum(df2.isnull().any(axis=1))

# %%
df2.dropna(inplace=True)
df2.reset_index(drop=True,inplace=True)
df2.info()

# %%
duplicates_specific = df2.duplicated(subset=['text'])
print(duplicates_specific)
duplicates_specific.sum()

# %%
df3 = df2.drop_duplicates()
df3.info()

# %%
from wordcloud import WordCloud

neg_tweets = df3[df3.target == 0]
neg_string = []
for t in neg_tweets.text:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')


wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
pos_tweets = df3[df3.target == 4]
pos_string = []
for t in pos_tweets.text:
    pos_string.append(t)
pos_string = pd.Series(pos_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(pos_string) 
plt.figure(figsize=(12,10)) 
plt.imshow(wordcloud, interpolation="bilinear") 
plt.axis("off") 
plt.show()

# %%
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(df3.text)

# %%
neg_doc_matrix = cvec.transform(df3[df3.target == 0].text)
pos_doc_matrix = cvec.transform(df3[df3.target == 4].text)
neg_tf = np.sum(neg_doc_matrix,axis=0)
pos_tf = np.sum(pos_doc_matrix,axis=0)
neg = np.squeeze(np.asarray(neg_tf))
pos = np.squeeze(np.asarray(pos_tf))
term_freq_df = pd.DataFrame([neg,pos],columns=cvec.get_feature_names_out()).transpose()

# %%
term_freq_df.head()

# %%
term_freq_df.columns = ['negative', 'positive']
term_freq_df['total'] = term_freq_df['negative'] + term_freq_df['positive']
term_freq_df.sort_values(by='total', ascending=False).iloc[:10]

# %%
y_pos = np.arange(500)
plt.figure(figsize=(10,8))
s = 1
expected_zipf = [term_freq_df.sort_values(by='total', ascending=False)['total'][0]/(i+1)**s for i in y_pos]
plt.bar(y_pos, term_freq_df.sort_values(by='total', ascending=False)['total'][:500], align='center', alpha=0.5)
plt.plot(y_pos, expected_zipf, color='r', linestyle='--',linewidth=2,alpha=0.5)
plt.ylabel('Frequency')
plt.title('Top 500 tokens in tweets')

# %%
from pylab import *
counts = term_freq_df.total
tokens = term_freq_df.index
ranks = arange(1, len(counts)+1)
indices = argsort(-counts)
frequencies = counts[indices]
plt.figure(figsize=(8,6))
plt.ylim(1,10**6)
plt.xlim(1,10**6)
loglog(ranks, frequencies, marker=".")
plt.plot([1,frequencies[0]],[frequencies[0],1],color='r')
title("Zipf plot for tweets tokens")
xlabel("Frequency rank of token")
ylabel("Absolute frequency of token")
grid(True)
for n in list(logspace(-0.5, log10(len(counts)-2), 25).astype(int)):
    dummy = text(ranks[n], frequencies[n], " " + tokens[indices[n]], 
                 verticalalignment="bottom",
                 horizontalalignment="left")

# %%
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='negative', ascending=False)['negative'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')

# %%
y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df.sort_values(by='positive', ascending=False)['positive'][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 positive tokens')
plt.title('Top 50 tokens in positive tweets')

# %%
import seaborn as sns
plt.figure(figsize=(8,6))
ax = sns.regplot(x="negative", y="positive",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
plt.ylabel('Positive Frequency')
plt.xlabel('Negative Frequency')
plt.title('Negative Frequency vs Positive Frequency')

# %%
term_freq_df['pos_rate'] = term_freq_df['positive'] * 1./term_freq_df['total']
print(term_freq_df.sort_values(by='pos_rate', ascending=False).iloc[:10])
term_freq_df.head()

# %%
term_freq_df['pos_freq_pct'] = term_freq_df['positive'] * 1./term_freq_df['positive'].sum()
term_freq_df.sort_values(by='pos_freq_pct', ascending=False).iloc[:10]

# %%
import pandas as pd

# Assuming term_freq_df is already defined
new_term_freq_df = term_freq_df.copy()

# Dropping specified columns while keeping the unnamed column
new_term_freq_df = new_term_freq_df.drop(columns=[
    'pos_rate', 'pos_freq_pct'
])

# Save the new DataFrame to a CSV file
new_term_freq_df.to_csv('term.csv')

# Read the CSV file back into a new DataFrame
term_df = pd.read_csv('term.csv', index_col=0)

# Sort the DataFrame by the 'total' column and get the top 10 entries
top_10_terms = term_df.sort_values(by='total', ascending=False).head(10)

# Display the top 10 terms
print(top_10_terms)


# %%
from scipy.stats import hmean
term_freq_df['pos_hmean'] = term_freq_df.apply(lambda x: (hmean([x['pos_rate'], x['pos_freq_pct']])                                                               if x['pos_rate'] > 0 and x['pos_freq_pct'] > 0 else 0), axis=1)
                                                       
term_freq_df.sort_values(by='pos_hmean', ascending=False).iloc[:10]

# %%
from scipy.stats import norm
def normcdf(x):
    return norm.cdf(x, x.mean(), x.std())
term_freq_df['pos_rate_normcdf'] = normcdf(term_freq_df['pos_rate'])
term_freq_df['pos_freq_pct_normcdf'] = normcdf(term_freq_df['pos_freq_pct'])
term_freq_df['pos_normcdf_hmean'] = hmean([term_freq_df['pos_rate_normcdf'], term_freq_df['pos_freq_pct_normcdf']])
term_freq_df.sort_values(by='pos_normcdf_hmean',ascending=False).iloc[:10]

# %%
term_freq_df['neg_rate'] = term_freq_df['negative'] * 1./term_freq_df['total']
term_freq_df['neg_freq_pct'] = term_freq_df['negative'] * 1./term_freq_df['negative'].sum()
term_freq_df['neg_hmean'] = term_freq_df.apply(lambda x: (hmean([x['neg_rate'], x['neg_freq_pct']])                                                                if x['neg_rate'] > 0 and x['neg_freq_pct'] > 0                                                                else 0), axis=1)
                                                       
term_freq_df['neg_rate_normcdf'] = normcdf(term_freq_df['neg_rate'])
term_freq_df['neg_freq_pct_normcdf'] = normcdf(term_freq_df['neg_freq_pct'])
term_freq_df['neg_normcdf_hmean'] = hmean([term_freq_df['neg_rate_normcdf'], term_freq_df['neg_freq_pct_normcdf']])
term_freq_df.sort_values(by='neg_normcdf_hmean', ascending=False).iloc[:10]

# %%
plt.figure(figsize=(8,6))
ax = sns.regplot(x="neg_hmean", y="pos_hmean",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
plt.ylabel('Positive Rate and Frequency Harmonic Mean')
plt.xlabel('Negative Rate and Frequency Harmonic Mean')
plt.title('neg_hmean vs pos_hmean')

# %%
plt.figure(figsize=(7,5))
ax = sns.regplot(x="neg_normcdf_hmean", y="pos_normcdf_hmean",fit_reg=False, scatter_kws={'alpha':0.5},data=term_freq_df)
plt.ylabel('Positive Rate and Frequency CDF Harmonic Mean')
plt.xlabel('Negative Rate and Frequency CDF Harmonic Mean')
plt.title('neg_normcdf_hmean vs pos_normcdf_hmean')

# %%
from bokeh.plotting import figure
from bokeh.io import output_notebook, show
from bokeh.models import LinearColorMapper
from bokeh.models import HoverTool
output_notebook()
color_mapper = LinearColorMapper(palette='Inferno256', low=min(term_freq_df.pos_normcdf_hmean), high=max(term_freq_df.pos_normcdf_hmean))
p = figure(x_axis_label='neg_normcdf_hmean', y_axis_label='pos_normcdf_hmean')
p.circle('neg_normcdf_hmean','pos_normcdf_hmean',size=5,alpha=0.3,source=term_freq_df,color={'field': 'pos_normcdf_hmean', 'transform': color_mapper})
hover = HoverTool(tooltips=[('token','@index')])
p.add_tools(hover)
show(p)

# %%
from sklearn.model_selection import train_test_split

x = df3.text
y = df3.target

SEED = 2000

x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)


print ("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_train),                                                                     
(len(x_train[y_train == 0]) / (len(x_train)*1.))*100,                                                                           
(len(x_train[y_train == 4]) / (len(x_train)*1.))*100))
print ("Validation set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_validation),                                                                          
(len(x_validation[y_validation == 0]) / (len(x_validation)*1.))*100,                                                                       
(len(x_validation[y_validation == 4]) / (len(x_validation)*1.))*100))
print ("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(x_test),                                                          
(len(x_test[y_test == 0]) / (len(x_test)*1.))*100,                                                           
(len(x_test[y_test == 4]) / (len(x_test)*1.))*100))

# %%
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Assuming x_validation and y_validation are already defined
tbresult = [TextBlob(i).sentiment.polarity for i in x_validation]
tbpred = [0 if n < 0 else 4 for n in tbresult]



# %%
print("Unique true labels:", np.unique(y_validation))
print("Unique predicted labels:", np.unique(tbpred))

# %%
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(classification_report(y_validation, tbpred, zero_division=0))


# %%
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(
    x, y, test_size=0.02, random_state=SEED, stratify=y
)


# %%
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate confusion matrix
conmat = np.array(confusion_matrix(y_validation, tbpred, labels=[0, 4]))
confusion = pd.DataFrame(conmat, index=['positive', 'negative'],
                         columns=['predicted_positive', 'predicted_negative'])

# Print accuracy score
print("Accuracy Score: {0:.2f}%".format(accuracy_score(y_validation, tbpred) * 100))
print("-" * 80)

# Print confusion matrix
print("Confusion Matrix\n")
print(confusion)
print("-" * 80)

# Print classification report
print("Classification Report\n")
print(classification_report(y_validation, tbpred))

# %%
x_train_sample = x_train[:10000] 
y_train_sample = y_train[:10000]
x_validation_sample = x_validation[:2000]
y_validation_sample = y_validation[:2000]

# %%
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def accuracy_summary(pipeline, x_train, y_train, x_test, y_test):
    null_accuracy = max((y_test == 0).sum(), (y_test == 1).sum()) / len(y_test)

    t0 = time()
    y_pred = pipeline.fit(x_train, y_train).predict(x_test)
    train_test_time = time() - t0
    
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Null accuracy: {null_accuracy:.2f}%")
    print(f"Accuracy score: {accuracy:.2f}%")
    print(f"Model is {abs(accuracy - null_accuracy) * 100:.2f}% {'more' if accuracy > null_accuracy else 'less' if accuracy < null_accuracy else 'the same'} accurate than null accuracy")
    print(f"Train and test time: {train_test_time:.2f}s")
    print("-" * 80)

    return accuracy, train_test_time

def nfeature_accuracy_checker(vectorizer, n_features, stop_words=None, ngram_range=(1, 1), classifier=LogisticRegression(max_iter=200)):
    results = []
    print(classifier, "\n")

    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        pipeline = Pipeline([('vectorizer', vectorizer), ('scaler', StandardScaler(with_mean=False)), ('classifier', classifier)])
        
        print(f"Validation result for {n} features")
        accuracy, tt_time = accuracy_summary(pipeline, x_train, y_train, x_validation, y_validation)
        results.append((n, accuracy, tt_time))
    
    return results


# %%
from sklearn.feature_extraction import text
a = frozenset(list(term_df.sort_values(by='total', ascending=False).iloc[:10].index))
b = text.ENGLISH_STOP_WORDS
set(a).issubset(set(b))

# %%
stop_words = frozenset(list(term_df.sort_values(by='total', ascending=False).iloc[:10].index))
my_stop_words = list(frozenset(list(term_freq_df.sort_values(by='total', ascending=False).iloc[:10].index)))


# %%
vectorizer = CountVectorizer()
n_features = [20000, 40000, 60000, 80000, 100000]

print("RESULT FOR UNIGRAM WITHOUT STOP WORDS\n")
feature_result_wosw = nfeature_accuracy_checker(vectorizer, n_features, stop_words='english')
print("RESULT FOR UNIGRAM WITH STOP WORDS\n")
feature_result_ug = nfeature_accuracy_checker(vectorizer, n_features)
print("RESULT FOR UNIGRAM WITHOUT CUSTOM STOP WORDS (Top 10 frequent words)\n")
feature_result_wocsw = nfeature_accuracy_checker(vectorizer, n_features, stop_words=my_stop_words)

# %%
nfeatures_plot_ug = pd.DataFrame(feature_result_ug, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ug_wocsw = pd.DataFrame(feature_result_wocsw, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ug_wosw = pd.DataFrame(feature_result_wosw, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, marker='o', label='With Stop Words')
plt.plot(nfeatures_plot_ug_wocsw.nfeatures, nfeatures_plot_ug_wocsw.validation_accuracy, marker='x', label='Without Custom Stop Words')
plt.plot(nfeatures_plot_ug_wosw.nfeatures, nfeatures_plot_ug_wosw.validation_accuracy, marker='s', label='Without Stop Words')

plt.title("Unigram: Validation Accuracy with vs Without Stop Words")
plt.xlabel("Number of Features")
plt.ylabel("Validation Set Accuracy")
plt.xticks(nfeatures_plot_ug.nfeatures)
plt.legend()
plt.grid(True)  
plt.tight_layout()  
plt.show()


# %%
vectorizer = CountVectorizer()
n_features = [40000, 60000, 80000, 100000]


print("RESULT FOR BIGRAM WITH STOP WORDS\n")
feature_result_bg = nfeature_accuracy_checker(vectorizer, n_features, ngram_range=(1, 2))

print("RESULT FOR TRIGRAM WITH STOP WORDS\n")
feature_result_tg = nfeature_accuracy_checker(vectorizer, n_features, ngram_range=(1, 3))


# %%
# Create DataFrames for plotting
nfeatures_plot_tg = pd.DataFrame(feature_result_tg, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_bg = pd.DataFrame(feature_result_bg, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ug = pd.DataFrame(feature_result_ug, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(nfeatures_plot_tg.nfeatures, nfeatures_plot_tg.validation_accuracy, label='Trigram', marker='o', color='royalblue')
plt.plot(nfeatures_plot_bg.nfeatures, nfeatures_plot_bg.validation_accuracy, label='Bigram', marker='o', color='orangered')
plt.plot(nfeatures_plot_ug.nfeatures, nfeatures_plot_ug.validation_accuracy, label='Unigram', marker='o', color='gold')

plt.title("Test Results: Accuracy")
plt.xlabel("Number of Features")
plt.ylabel("Validation Set Accuracy")
plt.xticks(nfeatures_plot_tg.nfeatures) 
plt.xlim(40000, nfeatures_plot_tg.nfeatures.max())
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

# %%
def train_test_and_evaluate(pipeline, x_train, y_train, x_test, y_test):
    # Calculate null accuracy
    null_accuracy = max(len(x_test[y_test == 0]), len(x_test[y_test == 1])) / len(x_test)
    
    # Fit the model and predict
    sentiment_fit = pipeline.fit(x_train, y_train)
    y_pred = sentiment_fit.predict(x_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix
    conmat = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion = pd.DataFrame(conmat, index=['negative', 'positive'],
                              columns=['predicted_negative', 'predicted_positive'])
    
    # Print results
    print(f"Null accuracy: {null_accuracy * 100:.2f}%")
    print(f"Accuracy score: {accuracy * 100:.2f}%")
    
    if accuracy > null_accuracy:
        print(f"Model is {((accuracy - null_accuracy) * 100):.2f}% more accurate than null accuracy")
    elif accuracy == null_accuracy:
        print("Model has the same accuracy as the null accuracy")
    else:
        print(f"Model is {((null_accuracy - accuracy) * 100):.2f}% less accurate than null accuracy")
    
    print("-" * 80)
    print("Confusion Matrix\n")
    print(confusion)
    print("-" * 80)
    print("Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


# %%
%%time
tg_cvec = CountVectorizer(max_features=80000,ngram_range=(1, 3))
tg_pipeline = Pipeline([
        ('vectorizer', tg_cvec),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
train_test_and_evaluate(tg_pipeline, x_train, y_train, x_validation, y_validation)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
# Perform feature extraction using TfidfVectorizer
tvec = TfidfVectorizer()
n_features = [40000, 60000, 80000, 100000]


# Get results for different n-grams
feature_result_ugt = nfeature_accuracy_checker(vectorizer=tvec, n_features = n_features)
feature_result_bgt = nfeature_accuracy_checker(vectorizer=tvec,n_features = n_features, ngram_range=(1, 2))
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tvec,n_features=n_features, ngram_range=(1, 3))

# Create DataFrames for plotting
nfeatures_plot_tgt = pd.DataFrame(feature_result_tgt, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_bgt = pd.DataFrame(feature_result_bgt, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])
nfeatures_plot_ugt = pd.DataFrame(feature_result_ugt, columns=['nfeatures', 'validation_accuracy', 'train_test_time'])

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(nfeatures_plot_tgt.nfeatures, nfeatures_plot_tgt.validation_accuracy, label='Trigram TF-IDF Vectorizer', color='royalblue')
plt.plot(nfeatures_plot_bgt.nfeatures, nfeatures_plot_bgt.validation_accuracy, label='Bigram TF-IDF Vectorizer', color='orangered')
plt.plot(nfeatures_plot_ugt.nfeatures, nfeatures_plot_ugt.validation_accuracy, label='Unigram TF-IDF Vectorizer', color='gold')
plt.title("N-gram (1~3) Test Result: Accuracy")
plt.xlabel("Number of Features")
plt.ylabel("Validation Set Accuracy")
plt.xticks(nfeatures_plot_tgt.nfeatures)
plt.xlim(40000, nfeatures_plot_tgt.nfeatures.max()) 
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()


