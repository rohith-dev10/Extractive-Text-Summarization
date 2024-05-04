import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
nltk.download('stopwords')
import re
# from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
english_stopwords.extend([",",".","$","%","'s","``","''"])


data = pd.read_csv('./SEM-6ML/project/data.csv')

title=data["Title"]
articles = data['News']
summaries = data['Summary']

'''
1. para follows title       ## 0/1
4. 1st sente                ## 0/1
2. para loc                 ## |(cosine(para#.)/#.of.paras in the current news)*pi|
3. sente loc                ## |(cosine(sente#.)/#.of.sentes in the current paragraph)*pi|
5. sent len                 ## #.of.words                 nltk
6. thematic words           ## #.of.mostFrequentWords/#.of.contentWords     rake
# 7. title words              ## #.of.words in title/#.of.contentWords
8. proper noun ratio        ## #.of.propernouns/senteLen  nltk
9. stat ratio               ## #.of.stats/senteLen        nltk

10. included in summary     ## 0/1
'''
'''
X=[
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9],
    [f1,f2,...,f9] 
]
Y=[1,0,0,1,1,0,0,1]
'''
X=[]
y=[]

for i in range(len(title)):
    titleTokens=[t for t in title[i].split(" ") if t.lower() not in english_stopwords]
    pattern = r'(?<=[.?!])(?:\s*(?=[^0-9.]|[0-9]\.[0-9]))'
    summarySente = re.split(pattern, summaries[i])
    # summarySente=summaries[i].split(". ")
    summarySente = [sentence.rstrip('.?!') for sentence in summarySente]

    allWords=nltk.tokenize.word_tokenize(articles[i])
    allWordExceptStopDist = nltk.FreqDist(w.lower() for w in allWords if w.lower() not in english_stopwords and w.isalnum())   
    mostCommon= [k for k,c in allWordExceptStopDist.most_common(10)]
    # print(mostCommon)
    
    pos_tags = nltk.pos_tag(allWords)
    proper_nouns = [word for word, pos_tag in pos_tags if pos_tag == 'NNP']
    stats = [item for item, pos_tag in pos_tags if pos_tag in ['CD']]

    articleFeatureVects=[]
    yVect=[] 
    for j,para in enumerate(articles[i].split("\n\n")):
        for k,sente in enumerate([sentence.rstrip('.?!') for sentence in re.split(pattern,para)]):
            if len(sente)==0:
                continue
            senteFeatureVect=[]
            senteFeatureVect.append(1 if j==0 else 0)
            senteFeatureVect.append(1 if k==0 else 0)
            senteFeatureVect.append(np.absolute(np.pi*np.cos(j)/len(articles[i].split("\n\n"))))
            senteFeatureVect.append(np.absolute(np.pi*np.cos(k)/len(para.split(". "))))
            senteFeatureVect.append(len(sente.split(" ")))



            titleWords=0
            thematicWords=0
            propnounWords=0
            statsWords=0
            for word in sente.split(" "):
                # if word in titleTokens:
                #     titleWords+=1
                if word in mostCommon:
                    thematicWords+=1
                if word in proper_nouns:
                    propnounWords+=1
                if word in stats:
                    statsWords+=1
            # titleWords/=(len(sente)-titleWords)
            thematicWords=100*thematicWords/(len(sente)-thematicWords+1 if len(sente)-thematicWords==0 else len(sente)-thematicWords)
            propnounWords=propnounWords/len(sente)*200
            statsWords=statsWords/len(sente)*300

            senteFeatureVect.append(thematicWords)
            # senteFeatureVect.append(titleWords)
            senteFeatureVect.append(propnounWords)
            senteFeatureVect.append(statsWords)

            yVect.append(1 if sente in summarySente else 0)
            articleFeatureVects.append(senteFeatureVect)
            # print(sente)
        # break
    
    X.extend(articleFeatureVects)
    y.extend(yVect)
    # break
dataset=pd.DataFrame(X,columns=["FisrtPara","FirstSente","ParaLoc","SenteLoc","SenteLen","ThematicWords","ProperNouns","StatRatio"])
# dataset=pd.DataFrame(,columns=["FisrtPara","FirstSente","ParaLoc","SenteLoc","SenteLen","ThematicWords","ProperNouns","StatRatio","Included"])
dataset["Included"]=y
print(dataset.head())
dataset.to_csv("./SEM-6ML/project/featuresDataset.csv")

# print(X)
# print(y)