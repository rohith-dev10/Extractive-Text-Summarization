import pandas as pd
import nltk
import re
import numpy as np
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

data =  pd.read_csv('./SEM-6ML/project/data.csv')
sentence=data["News"][0]
words = nltk.word_tokenize(sentence)

pos_tags = nltk.pos_tag(words)

proper_nouns = [word for word, pos_tag in pos_tags if pos_tag == 'NNP']
stats = [item for item, pos_tag in pos_tags if pos_tag in ['CD']]
num_proper_nouns = sum (1 for word, pos_tag in pos_tags if pos_tag == 'NNP')
num_stats_numbers = sum(1 for _, pos_tag in pos_tags if pos_tag in ['CD'])

print("list of proper nouns:", proper_nouns)
print("list of stats:", stats)
print("Number of proper nouns:", num_proper_nouns)
print("Number of stats:", num_stats_numbers)


pattern = r'(?<=[.?!])(?:\s*(?=[^0-9.]|[0-9]\.[0-9]))'
for j,para in enumerate(sentence.split("\n\n")):
    for k,sente in enumerate([sentence.rstrip('.?!') for sentence in re.split(pattern,para)]):
        
        titleWords=0
        thematicWords=0
        propnounWords=0
        statsWords=0
        
        for word in sente.split(" "):
            # if word in titleTokens:
            #     titleWords+=1
            # if word in mostCommon:
            #     thematicWords+=1
            # print(word)
            if word in proper_nouns:
                propnounWords+=1
                print(word)
            if word in stats:
                statsWords+=1
                print("stat: ",word)
        titleWords/=(len(sente)-titleWords)
        thematicWords/=(len(sente)-thematicWords)
        propnounWords/=len(sente)
        statsWords/=len(sente)

