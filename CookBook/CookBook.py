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
import itertools

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

cols=pd.MultiIndex.from_tuples([(x,y) for x in['A','B','C'] for y in ['O','I']])
df=DataFrame(np.random.randn(2,6),index=['n','m'],columns=cols);df
df=df.div(df['C'],level=1);df

coords=[('AA','one'),('AA','six'),('BB','one'),('BB','two'),('BB','six')]
index=pd.MultiIndex.from_tuples(coords)
df=DataFrame([11,22,33,44,55],index,['MyData']);df
df.xs('BB',level=0,axis=0)
df.xs('six',level=1,axis=0)

index=list(itertools.product(['Ada','Quinn','Violet'],['Comp','Math','Sci']))
header=list(itertools.product(['Exams','Labs'],['I','II']))
indx=pd.MultiIndex.from_tuples(index,names=['Student','Course'])
cols=pd.MultiIndex.from_tuples(header)
data=[[70+x+y+(x*y)%3 for x in range(4)] for y in range(9)]
df=DataFrame(data,indx,cols);df
All=slice(None);All
df.loc['Violet']
df.loc[(All,'Math'),All]
df.loc[(slice('Ada','Quinn'),'Math'),All]
df.loc[(All,'Math'),('Exams')]
df.loc[(All,'Math'),(All,'II')]
df.sort_values(by=('Labs','II'),ascending=False)

df=DataFrame(np.random.randn(6,1),index=pd.date_range('2013-08-01',periods=6,freq='b'),columns=list('A'))
df.loc[df.index[3],'A']=np.nan;df
df.reindex(df.index[::-1]).ffill()

df=DataFrame({'animal':'cat dog cat fish dog cat cat'.split(),'size':list('SSMMMLL'),'weight':[8,10,11,1,20,12,12],'adult':[False]*5+[True]*2});df
df.groupby('animal').apply(lambda subf:subf['size'][subf['weight'].idxmax()])
gb=df.groupby(['animal'])
gb.get_group('cat')
def GrowUp(x):
    avg_weight=sum(x[x['size']=='S'].weight*1.5)
    avg_weight+=sum(x[x['size']=='M'].weight*1.25)
    avg_weight+=sum(x[x['size']=='L'].weight)
    avg_weight/=len(x)
    return Series(['L',avg_weight,True],index=['size','weight','adult'])
expected_df=gb.apply(GrowUp)
expected_df

S=Series([i/100.0 for i in range(1,11)])
def CumRet(x,y):
    return x*(1+y)
def Red(x):
    return functools.reduce(CumRet,x,1.0)
S.expanding().apply(Red)

df=DataFrame({'A':[1,1,2,2],'B':[1,-1,1,2]})
gb=df.groupby('A')
def replace(g):
    mask=g<0
    g.loc[mask]=g[-mask].mean()
    return g
gb.transform(replace)

df=DataFrame({'code':['foo','bar','baz']*2,'data':[0.16,-0.21,0.33,0.45,-0.59,0.62],'flag':[False,True]*3})
code_groups=df.groupby('code')
agg_n_sort_order=code_groups[['data']].transform(sum).sort_values(by='data')
sorted_df=df.loc[agg_n_sort_order.index]
sorted_df
rng=pd.date_range(start='2014-10-07',periods=10,freq='2min')
ts=Series(data=list(range(10)),index=rng)
def MyCust(x):
    if len(x)>2:
        return x[1]*1.234
    return pd.NaT
mhc={'Mean':np.mean,'Max':np.max,'Custom':MyCust}
ts.resample('5min').apply(mhc)

df=DataFrame({'Color':'Red Red Red Blue'.split(),'Value':[100,150,50,50]});df
df['Counts']=df.groupby(['Color']).transform(len);df

df=DataFrame({'line_race':[10,10,8,10,10,8],'beyer':[99,102,103,103,88,100]},index=['Last Gunfighter','Last Gunfighter','Last Gunfighter','Paynter','Paynter','Paynter']);df
df['beyer_shifted']=df.groupby(level=0)['beyer'].shift(1);df

df=DataFrame({'host':['other','other','that','this','this'],'service':['mail','web','mail','mail','web'],'no':[1,2,1,2,1]}).set_index(['host','service']);df
mask=df.groupby(level=0).agg('idxmax');mask
df_count=df.loc[mask['no']].reset_index();df_count

df=DataFrame([0,1,0,1,1,1,0,1,1],columns=['A']);df
df.A.groupby((df.A!=df.A.shift()).cumsum()).groups
df.A.groupby((df.A!=df.A.shift()).cumsum()).cumsum()

df=DataFrame(data={'Case':['A']*3+['B']+['A']*2+['B']+['A']*2,'Data':np.random.randn(9)})
dfs=list(zip(*df.groupby((1*(df['Case']=='B')).cumsum().rolling(window=3,min_periods=1).median())))[-1]
dfs[0]

df=DataFrame(data={'Province':['ON','QC','BC','AL','AL','MN','ON'],'City':['Toronto','Montreal','Vancouver','Calgary','Edmonton','Winnipeg','Windsor'],'Sales':[13,6,16,8,4,3,1]})
table=pd.pivot_table(df,values=['Sales'],index=['Province'],columns=['City'],aggfunc=np.sum,margins=True)
table.stack('City')

grades=[48,99,75,80,42,80,72,68,36,78]
df=DataFrame({'ID':['x%d'% r for r in range(10)],'Gender':['F','M']*4+['M']*2,'ExamYear':['2007']*3+['2008']*4+['2009']*3,'Class':['algebra','stats','bio','algebra','algebra','stats','stats','algebra','bio','bio'],'Participated':['yes']*4+['no']+['yes']*5,'Passed':['yes' if x>50 else 'no' for x in grades],'Employed':[True]*3+[False]*4+[True]*2+[False],'Grade':grades})
df.groupby('ExamYear').agg({'Participated':lambda x:x.value_counts()['yes'],'Passed':lambda x:sum(x=='yes'),'Employed':lambda x:sum(x),'Grade':lambda x:sum(x)/len(x)})

df=DataFrame({'value':np.random.randn(36)},index=pd.date_range('2011-01-01',freq='m',periods=36))
pd.pivot_table(df,index=df.index.month,columns=df.index.year,values='value',aggfunc='sum')

df=DataFrame(data={'A':[[2,4,8,16],[100,200],[10,20,30]],'B':[['a','b','c'],['jj','kk'],['ccc']]},index=['I','II','III'])
def SeriesFromSubList(aList):
    return Series(aList)
df_orgz=pd.concat(dict([(ind,row.apply(SeriesFromSubList)) for ind,row in df.iterrows()]))
df=DataFrame(data=np.random.randn(2000,2)/10000,index=pd.date_range('2001-01-01',periods=2000),columns=['A','B']);df
def gm(aDF,Const):
    v=((((aDF['A']+aDF['B'])+1).cumprod())-1)*Const
    return (aDF.index[0],v.iloc[-1])
S=Series(dict([gm(df.iloc[i:min(i+51,len(df)-1)],5) for i in range(len(df)-50)]));S

rng=pd.date_range(start='2014-01-01',periods=100)
df=DataFrame({'Open':np.random.randn(len(rng)),'Close':np.random.randn(len(rng)),'Volume':np.random.randint(100,200,len(rng))},index=rng);df
def vwap(bars):
    return ((bars.Close*bars.Volume).sum()/bars.Volume.sum())
window=5
s=pd.concat([(Series(vwap(df.iloc[i:i+window]),index=[df.index[i+window]])) for i in range(len(df)-window)]);s.round(2)

dates=pd.date_range('2000-01-01',periods=5);dates
dates.to_period(freq='m').to_timestamp()

rng=pd.date_range('2000-01-01',periods=6)
df1=DataFrame(np.random.randn(6,3),index=rng,columns=['A','B','C'])
df2=df1.copy()
df=df1.append(df2,ignore_index=True);df

df=DataFrame(data={'Area':['A']*5+['C']*2,'Bins':[110]*2+[160]*3+[40]*2,'Test_0':[0,1,0,1,2,0,1],'Data':np.random.randn(7)});df
df['Test_1']=df['Test_0']-1
pd.merge(df,df,left_on=['Bins','Area','Test_0'],right_on=['Bins','Area','Test_1'],suffixes=('_L','_R'))

df=DataFrame({'stratifying_var':np.random.uniform(0,100,20),'price':np.random.normal(100,5,20)})
df['quartiles']=pd.qcut(df['stratifying_var'],4,labels=['0-25%','25-50%','50-75%','75-100%'])
df.boxplot(column='price',by='quartiles')