import json

f = open("Sarcasm_Headlines_Dataset.json")
datastore = json.load(f)

sentences = []
labels = []
urls = []

for item in datastore:
    sentences.append(item['headline'])
    sentences.append(item['is_sarcastic'])
    sentences.append(item['article_link'])
