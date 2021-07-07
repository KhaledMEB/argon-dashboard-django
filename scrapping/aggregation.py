import pickle
from tensorflow import keras
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from gensim.utils import simple_preprocess
import spacy
import re
import json
from html.parser import HTMLParser
from io import StringIO
import re
from tensorflow.python.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from .logics import StorageManager

class Aggregator:
    def __init__(self):
        pass

    def aggregate_topics_sentiments(self, data_file_path):
        print('Loading the sentiment analysis model ....')
        try:
            gru_model = keras.models.load_model('./scrapping/outputs/gru_model_binary.h5')
        except FileNotFoundError:
            storageManager = StorageManager()
            storageManager.downloadModel()
            gru_model = keras.models.load_model('./scrapping/outputs/gru_model_binary.h5')

        # load the lda Target Model
        lda_model = pickle.load(open("./scrapping/outputs/ldamodel.pkl", "rb"))
        print('Loading the sentiment analysis model .... OK')
        # load the corpus and the words dictionnary

        with open("./scrapping/outputs/corpus.txt", "rb") as fp:
            corpus = pickle.load(fp)

        with open('./scrapping/outputs/id2word.pkl', 'rb') as fp:
            id2word = pickle.load(fp)

        # save the list of keywords
        topics_key_words = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=10)
        with open('./scrapping/outputs/key_words.pkl', 'wb') as fp:
                pickle.dump(topics_key_words, fp, pickle.HIGHEST_PROTOCOL)

        # distribution of topics for each document

        tm_results = lda_model[corpus]

        # We can get the most dominant topic of each document as below:
        corpus_topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in tm_results]

        # get most probable words for the given topicis

        num_keywords = 10
        topics = [[(term,
                    round(wt, 3)) for term,
                wt in lda_model.show_topic(n, topn=num_keywords)] for n in range(0, lda_model.num_topics)]

        # set column width
        pd.set_option('display.max_colwidth', None)
        topics_df = pd.DataFrame([', '.join([term for term, wt in topic]) for topic in topics],
                                columns = ['Terme par Sujet'],
                                index=['Sujet '+str(t) for t in range(1, lda_model.num_topics+1)])

        data_free_punct = pd.read_pickle("./scrapping/outputs/data_free_punct.pkl")

        # Dominant Topic for each Tweet

        corpus_topic_df = pd.DataFrame()

        # get the Titles from the original dataframe
        corpus_topic_df['Tweet_id'] = data_free_punct.index
        corpus_topic_df['Topic'] = [item[0]+1 for item in corpus_topics]
        #corpus_topic_df['Contribution %'] = [round(item[1]*100, 2) for item in corpus_topics]
        corpus_topic_df['Keywords'] = [topics_df.iloc[t[0]]['Terme par Sujet'] for t in corpus_topics]

        topics_dist = corpus_topic_df.groupby('Topic').agg(
                                  Number_of_Documents = ('Tweet_id', np.size),
                                  Topic_Contribution = ('Tweet_id', np.size)).reset_index()

        topics_dist['Topic_Contribution'] = topics_dist['Topic_Contribution'].\
                                                apply(lambda row: round((row*100) / len(corpus), 2))
        
        data_folder = 'scrapping/tweets/'
        df = pd.read_json(data_folder + data_file_path, lines=True)

        df = df[df['Lang'] == 'fr']
        df = df.sort_values("Content") 
        
        # dropping ALL duplicte values 
        df = df.drop_duplicates(subset ="Content", keep = 'first')

        with open('./scrapping/inputs/keywords.txt', encoding='utf-8') as f:
            keywords = f.read().splitlines()

        df = df[df.Content.str.contains('|'.join(keywords), case=False)]

    # supprimer les tweets inutiles (publicité, concours ..)

        print('Classifying documents sentiments....')
        with open('./scrapping/inputs/ads_words.txt', encoding='utf-8') as f:
            ads_words = f.read().splitlines()

        df = df[~df.Content.str.contains('|'.join(ads_words), case=False)]

        tweets = df['Content'].values.tolist()
            
        tweets = normalize_texts(tweets)

        MAX_FEATURES = 12000
        tokenizer = Tokenizer(num_words=MAX_FEATURES)
        tokenizer.fit_on_texts(tweets)
        tweets = tokenizer.texts_to_sequences(tweets)

        vocab_size = len(tokenizer.word_index) + 1

        maxlen = 100

        tweets = pad_sequences(tweets, padding='post', maxlen=maxlen)

        preds = gru_model.predict(tweets)

        sentiments = preds > 0.3

        # SAVE THE GLOBAL SENTIMENTS
        unique, counts = np.unique(sentiments, return_counts=True)
        df_senti = pd.DataFrame(index = unique, data=counts, columns=['Total'])
        df_senti['Pourcentage %'] = round((df_senti.Total / len(sentiments)) * 100)
        df_senti.to_pickle('./scrapping/outputs/sentiment_global.pkl')
        print('Classifying documents sentiments....OK')

        print('Aggregating topics & sentiments....')
        df_agg = corpus_topic_df
        df_agg['sentiment'] = sentiments

        df_satisfaction = df_agg[df_agg['sentiment'] == True]

        df_satisfaction = df_satisfaction[['Topic', 'Tweet_id']].groupby('Topic').agg('count').reset_index()
        df_satisfaction.columns = [['Topic','Client_Satisfaction']]
        topics_dist['Client_Satisfaction'] = df_satisfaction['Client_Satisfaction']
        topics_dist['Client_Satisfaction'] = round(topics_dist['Client_Satisfaction'] / topics_dist['Number_of_Documents'], 2) * 100

        # SAVE THE TOPICS FILE
        #json_records = topics_dist.reset_index().to_json(orient ='records')
        topics_dist.to_pickle('./scrapping/outputs/topics_dist.pkl')
        print('Aggregating topics & sentiments....OK')

def normalize_texts(texts):
    NON_ALPHANUM = re.compile(r'[\W]')
    NON_ASCII = re.compile(r'[^a-z0-1\s]')
    normalized_texts = []
    for text in texts:
        lower = text.lower()
        no_punctuation = NON_ALPHANUM.sub(r' ', lower)
        no_non_ascii = NON_ASCII.sub(r'', no_punctuation)
        normalized_texts.append(no_non_ascii)
    return normalized_texts