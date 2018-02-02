# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 11:08:23 2018

@author: Lenny
"""

#CookBook

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import functools

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
df.loc[df.AAA>=5,'BBB']=-1;df
df.loc[df.AAA>=5,['BBB','CCC']]=555;df
df.loc[df.AAA<5,['BBB','CCC']]=2000;df
df_mask=DataFrame({'AAA':[True]*4,'BBB':[False]*4,'CCC':[True,False]*2})
df.where(df_mask,-1000)
df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
df['logic']=np.where(df['AAA']>5,'high','low');df

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
dflow=df[df.AAA<=5];dflow
dfhigh=df[df.AAA>5];dfhigh

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
newseries=df.loc[(df['BBB']<25)&(df['CCC']>=-40),'AAA'];newseries
newseries=df.loc[(df['BBB']<25)|(df['CCC']>=-40),'AAA'];newseries
df.loc[(df['BBB']>25)|(df['CCC']>=75),'AAA']=0.1;df

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
aValue=43.0
df.loc[(df.CCC-aValue).abs().argsort()]

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
Crit1=df.AAA<=5.5
Crit2=df.BBB==10.0
Crit3=df.CCC>-40
AllCrit=Crit1&Crit2&Crit3
df[AllCrit]
CritList=[Crit1,Crit2,Crit3]
AllCrit=functools.reduce(lambda x,y:x&y,CritList)
df[AllCrit]

df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
df[(df.AAA<=6)&(df.index.isin([0,2,4]))]
data={'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]}
df=DataFrame(data,index=['foo','bar','boo','kar']);df
df.loc['bar':'kar']#Label
df.iloc[0:3]
df=DataFrame(data=data,index=[1,2,3,4]);df
df.iloc[1:3]#Position-oriented
df.loc[1:3]#Label-oriented
df=DataFrame({'AAA':[4,5,6,7],'BBB':[10,20,30,40],'CCC':[100,50,-30,-50]});df
df[-((df.AAA<=6)&(df.index.isin([0,2,4])))]

rng=pd.date_range('1/1/2013',periods=100,freq='d')
data=np.random.randn(100,4)
cols=['A','B','C','D']
df1,df2,df3=DataFrame(data,rng,cols),DataFrame(data,rng,cols),DataFrame(data,rng,cols)
pf=pd.Panel({'df1':df1,'df2':df2,'df3':df3});pf
pf.loc[:,:,'F']=DataFrame(data,rng,cols);pf

df=DataFrame({'AAA':[1,2,1,3],'BBB':[1,1,2,2],'CCC':[2,1,3,1]});df
source_cols=df.columns
new_cols=[str(x)+"_cat" for x in source_cols]
categories={1:'Alpha',2:'Beta',3:'Charlie'}
df[new_cols]=df[source_cols].applymap(categories.get);df

df=DataFrame({'AAA':[1,1,1,2,2,2,3,3],'BBB':[2,1,3,4,5,1,2,3]});df
df.loc[df.groupby('AAA')['BBB'].idxmin()]
df.sort_values(by='BBB').groupby('AAA',as_index=False).first()

df=DataFrame({'row':[0,1,2],'One_X':[1.1,1.1,1.1],'One_Y':[1.2,1.2,1.2],'Two_X':[1.11,1.11,1.11],'Two_Y':[1.22,1.22,1.22]});df
df=df.set_index('row');df
df.columns=pd.MultiIndex.from_tuples([tuple(c.split('_')) for c in df.columns]);df
df=df.stack(0).reset_index(1);df
df.columns=['Sample','All_X','All_Y'];df
