import os
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import snscrape.modules.twitter as sntwitter
import pandas as pd
import json
from datetime import date

class DataCollecter:
    def __init__(self):
        pass
    
    def collect_tweet(self, search, since, lang):
        tweets_list = []
        params = search + ' lang:' + lang + ' since:' +since

        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(params).get_items()):
            tweets_list.append([tweet.id, tweet.content, tweet.lang])

        tweets_df = pd.DataFrame(tweets_list, columns=['Id', 'Content', 'Lang'])
        
        local_file_name = search + '-' + lang + '-' + str(date.today()) + '.json'
        local_directory = './scrapping/tweets/'
        tweets_df.to_json(local_directory + local_file_name, orient='records', lines=True)
        
        return local_file_name