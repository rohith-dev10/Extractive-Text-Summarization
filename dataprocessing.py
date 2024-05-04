import os
import pandas as pd
import re
# import numpy as np

filepath="M:/downloads/BBC News Summary"
newspath=os.path.join(filepath,"News Articles")
summariespath=os.path.join(filepath,"Summaries")

data=[]
for category in os.listdir(newspath):
    newscategorypath=os.path.join(newspath,category)
    summarycategorypath=os.path.join(summariespath,category)
    for newstxtfile in os.listdir(newscategorypath):
        newstxtpath=os.path.join(newscategorypath,newstxtfile)
        summarytxtpath=os.path.join(summarycategorypath,newstxtfile)
        with open(newstxtpath) as newsfile:
            summaryfile=open(summarytxtpath)
            # print(newstxtpath,"hijk1.",newsfile.readline(),".2----he---",newsfile.read().strip())
            title=newsfile.readline().strip()
            news=newsfile.read().strip()
            summary=summaryfile.read()
            data.append({"Title":title,"News":news,"Summary":summary})
df=pd.DataFrame(data=data)

df.to_csv("./SEM-6ML/project/data.csv")