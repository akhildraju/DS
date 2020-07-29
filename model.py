from warnings import filterwarnings
filterwarnings("ignore")

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import nltk
import json
import urllib
import re
import pandas as pd


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

#If you get stopwords error pleasew uncomment the following two lines.
# nltk.download('stopwords')
# nltk.download('wordnet')

def get_compund_score(text):
        score = analyzer.polarity_scores(text)
        str(text)
        return score['compound']
    
def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

def get_score_by_comment_id(id):
  x=[]
  html = urllib.request.urlopen(
      'https://hacker-news.firebaseio.com/v0/item/' + str(id) + '.json')
  x.append(json.loads(html.read()))
  df = pd.DataFrame.from_dict(x)
  df_comments = df[df['type'] == 'comment']
  df_comments['clean_text']= df_comments['text'].map(lambda s:preprocess(s)) 
  df_comments['clean_vader_score'] =   df_comments['clean_text'].apply(
      get_compund_score)
  
  return (df_comments['clean_vader_score'][0])


def get_score_for_entries(entries):
  data = []
  max_entries = 25
  count = 0
  for id in entries:
    html = urllib.request.urlopen('https://hacker-news.firebaseio.com/v0/item/' + str(id) + '.json')
    data.append(json.loads(html.read()))
    count += 1
    if count > max_entries:
        break

  data = [i for i in data if i is not None]

  df = pd.DataFrame.from_dict(data)
  df_comments = df[df['type'] == 'comment']

  df_comments['clean_text']=df_comments['text'].map(lambda s:preprocess(s)) 
  df_comments['clean_vader_score'] = df_comments['clean_text'].apply(get_compund_score)

  return df_comments['clean_vader_score'].sum() #we can use mean()


def get_cummulative_score_for_user(username):

  data = []
  html = urllib.request.urlopen('https://hacker-news.firebaseio.com/v0/user/' + str(username) + '.json?print=pretty')
  data.append(json.loads(html.read()))
  df2 = pd.DataFrame.from_dict(data)
  entries =  (df2['submitted'][0])

  score = get_score_for_entries(entries)
  return score



def main():
    id = 23970146
    print(f"Score for the comment id {id} is: ",  get_score_by_comment_id(id))

    test_message = "This is really bad. Never going there again. Absolute worst"
    print (f"Score for the comment '{test_message}' is : ", get_compund_score(test_message))

    test_message = "This is really good. Loved the place. Employees were super kind. Can't wait to go back again."
    print (f"Score for the comment '{test_message}' is : ", get_compund_score(test_message))


    user = 'gmfawcett'
    score = get_cummulative_score_for_user(user)
    print(f"Cummulative score for user {user}", score )


if __name__ == "__main__":
    main()
