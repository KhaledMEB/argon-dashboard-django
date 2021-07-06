import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import re
import json
from html.parser import HTMLParser
from io import StringIO

import demoji
import pickle
from gensim.models import LdaMulticore, CoherenceModel

class TopicModeler:
    def __init__(self):
        demoji.download_codes()
        pass
    
    def get_topics(self, data_file_path):
        data_folder = 'scrapping/tweets/'
        df = pd.read_json(data_folder + data_file_path, lines=True)
        # filtrerer les données
        # remove duplicates 
        # 840 tweets had been droped

        df = df[df['Lang'] == 'fr']
        df = df.sort_values("Content") 
        
        # dropping ALL duplicte values 
        df = df.drop_duplicates(subset ="Content", keep = 'first')

        # selectionner que les tweets qui répondent au requetes de l'utilisateur

        data = df['Content']

        # modifier ici ########
        with open('./scrapping/inputs/keywords.txt', encoding='utf-8') as f:
            keywords = f.read().splitlines()

        data = data[data.str.contains('|'.join(keywords), case=False)]
        #data = data[data.str.contains('batterie', case=False)]

        # supprimer les tweets inutiles (publicité, concours ..)

        with open('./scrapping/inputs/ads_words.txt', encoding='utf-8') as f:
            ads_words = f.read().splitlines()

        data = data[~data.str.contains('|'.join(ads_words), case=False)]

        # nettoyer les données
        data_free_html = data.apply(html_free_text)
        data_free_emoji = data_free_html.apply(emoji_free_text)
        data_free_url = data_free_emoji.apply(url_free_text)
        data_free_abrivot = data_free_url.apply(abrivot_free_text)
        data_free_punct = data_free_url.apply(punct_free_text)

        # save the preprocessed data
        data_free_punct.to_pickle('./scrapping/outputs/data_free_punct.pkl')

        # normalize the data
        # Build the bigram and trigrams

        data = list(data_free_punct)

        bigram = gensim.models.Phrases(data, min_count=5, threshold=100)
        trigram = gensim.models.Phrases(bigram[data], threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)

        # define a preprocessing function

        # only need tagger, no need for parser and named entity recognizer, for faster implementation
        nlp = spacy.load('fr_core_news_lg', disable=['parser', 'ner'])

        # get stopwords
        # ask the user to specify the brand name to be added to the stopwords
        # or implement it manually
        with open('./scrapping/inputs/fr_stopwords.txt', encoding='utf-8') as f:
            fr_stopwords = f.read().splitlines()
        stop_words = nlp.Defaults.stop_words.union(fr_stopwords)

        data_ready = process_words(data, stop_words, nlp, bigram_mod, trigram_mod)

        # create the corpus and the dictionnary
        id2word = corpora.Dictionary(data_ready)
        corpus = [id2word.doc2bow(text) for text in data_ready]
        dict_corpus = {}

        for i in range(len(corpus)):
            for idx, freq in corpus[i]:
                if id2word[idx] in dict_corpus:
                    dict_corpus[id2word[idx]] += freq
                else:
                    dict_corpus[id2word[idx]] = freq

        dict_df = pd.DataFrame.from_dict(dict_corpus, orient='index', columns=['freq'])

        if(len(dict_df) > 30):
            threshold = dict_df.sort_values('freq', ascending=False).iloc[9].values[0]
        else:
            threshold = dict_df.sort_values('freq', ascending=False).iloc[3].values[0]
        extension = dict_df[dict_df.freq>threshold].index.tolist()

        extension = [word for word in extension if word not in keywords]

        # add high frequency words to stop words list
        stop_words.update(extension)
        # rerun the process_words function
        data_ready = process_words(data, stop_words, nlp, bigram_mod, trigram_mod)
        # recreate Dictionary
        id2word = corpora.Dictionary(data_ready)

        id2word.filter_extremes(no_below=20, no_above=0.9)
        corpus = [id2word.doc2bow(text) for text in data_ready]

        # save the dic
        with open('./scrapping/outputs/id2word.pkl', 'wb') as fp:
                pickle.dump(id2word, fp, pickle.HIGHEST_PROTOCOL)
                
        # save the corpus
        with open("./scrapping/outputs/corpus.txt", "wb") as fp:
            pickle.dump(corpus, fp)

        # building the model

        num_topics = list(range(2, 30, 2)[1:])
        num_keywords = 10

        LDA_models = {}
        LDA_topics = {}
        for i in num_topics:
            LDA_models[i] = LdaMulticore(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=i,
                                        chunksize=2000,
                                        passes=25,
                                        iterations=70,
                                        decay=0.5,
                                        random_state=100
                                        )

            shown_topics = LDA_models[i].show_topics(num_topics=i, 
                                                    num_words=num_keywords,
                                                    formatted=False)
            LDA_topics[i] = [[word[0] for word in topic[1]] for topic in shown_topics]
        
        # Use the above to derive the mean stability across topics by considering the next topic:

        LDA_stability = {}
        for i in range(0, len(num_topics)-1):
            jaccard_sims = []
            for t1, topic1 in enumerate(LDA_topics[num_topics[i]]): # pylint: disable=unused-variable
                sims = []
                for t2, topic2 in enumerate(LDA_topics[num_topics[i+1]]): # pylint: disable=unused-variable
                    sims.append(jaccard_similarity(topic1, topic2))    
                
                jaccard_sims.append(sims)    
            
            LDA_stability[num_topics[i]] = jaccard_sims
                        
        mean_stabilities = [np.array(LDA_stability[i]).mean() for i in num_topics[:-1]]

        # calculate the coherence value with the built in gensim

        coherences = [CoherenceModel(model=LDA_models[i], texts=data_ready,
                                    dictionary=id2word, coherence='c_v', topn=num_keywords).get_coherence() for i in num_topics[:-1]]

        # From here derive the ideal number of topics roughly through the difference between the coherence and stability per number of topics:


        coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(len(num_topics) - 1)[:-1]]
        #coh_sta_diffs = [coherences[i] - mean_stabilities[i] for i in range(num_keywords)[:-1]] # limit topic numbers to the number of keywords
        coh_sta_max = max(coh_sta_diffs)
        coh_sta_max_idxs = [i for i, j in enumerate(coh_sta_diffs) if j == coh_sta_max]
        ideal_topic_num_index = coh_sta_max_idxs[0] # choose less topics in case there's more than one max
        ideal_topic_num = num_topics[ideal_topic_num_index]

        # save the model 

        ldamodel = LDA_models[ideal_topic_num]
        pickle.dump(ldamodel, open("./scrapping/outputs/ldamodel.pkl", "wb"))


class MLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs= True
            self.text = StringIO()
        def handle_data(self, d):
            self.text.write(d)
        def get_data(self):
            return self.text.getvalue()
        
def html_free_text(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def emoji_free_text(text):
    return demoji.replace(text, '').strip()

def url_free_text(text):
    text = re.sub(r'(?:\@|https?\://)\S+', '', text)
    return text
    
with open('./scrapping/inputs/abrivot_fr.json', encoding='utf-8') as f:
        abrivot = json.load(f)   
        
def abrivot_free_text(text):
    words = text.lower().split()
    text_out = [abrivot[word] if word in abrivot else word for word in words]
    return ' '.join(text_out)

def punct_free_text(text):
    text_out = simple_preprocess(text, deacc=True, min_len=3)
    return text_out

# final preprocesser
def process_words(texts, stop_words, nlp, bigram_mod, trigram_mod, allowed_tags=['NOUN']):
    
    """Convert a document into a list of lowercase tokens, build bigrams-trigrams, implement lemmatization"""
    
    # remove stopwords, short tokens and letter accents 
    #texts = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts]
    texts = [[word for word in doc if word not in stop_words] for doc in texts]

    
    # bi-gram and tri-gram implementation
    texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
    
    texts_out = []
    
    # implement lemmatization and filter out unwanted part of speech tags
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_tags])
    
    # remove stopwords and short tokens again after lemmatization
    texts_out = [[word for word in simple_preprocess(str(doc), deacc=True, min_len=3) if word not in stop_words] for doc in texts_out]    
    
    return texts_out

def jaccard_similarity(topic_1, topic_2):

    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))
                    
    return float(len(intersection))/float(len(union))