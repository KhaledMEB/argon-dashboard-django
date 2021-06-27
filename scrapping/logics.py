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


class StorageManager:
    def __init__(self):
        self.connect_str = os.environ['AZURE_CONNECT_STRING']
        
    def uploadData(self, local_file_name):
        blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)
        container_name = 'data-container'
        
        # Create a blob client using the local file name as the name for the blob
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
        # Upload the created file
        local_directory = './scrapping/tweets/'
        with open(local_directory + local_file_name, "rb") as data:
            blob_client.upload_blob(data)