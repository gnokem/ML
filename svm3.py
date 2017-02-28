import os, sys;
from sklearn import svm;
import numpy as np, pandas as pd;
import csv, random;

seed=13
np.random.seed(seed);
random.seed(seed);

def isNum(s):
 digits=[str(i) for i in range(0,10)]+['.'];
 for i in s: 
  if i not in digits: return False;
 if i.count('.')>1: return False;
 return True; 

def getCol(x,n): return [i[n] for i in x];

def strit(m): return [str(i) for i in m];

def head(x,n=10):
 for i in range(n): print x[i];
 return;


recs=[];
with open("lenc.csv","r") as csvf:
 rdr=csv.reader(csvf);
 for row in rdr: recs+=[row];

csvf.close();

crecs=[];
for i in recs[1:]: # take off the header
 t=[j if not isNum(j) else float(j) for j in i];
 crecs+=[t];

recs=[];
with open("lenc_test.csv","r") as csvf:
 rdr=csv.reader(csvf);
 for row in rdr: recs+=[row];

csvf.close();

trecs=[];
for i in recs[1:]: # take off the header
 t=[j if not isNum(j) else float(j) for j in i];
 trecs+=[t];


tr_set=crecs[:-5000];
tr_set_x=[i[:-1] for i in crecs[:-5000]];
tr_set_y=[i[-1] for i in crecs[:-5000]];

te_set=crecs[-5000:];
te_set_x=[i[:-1] for i in crecs[-5000:]];
te_set_y=[i[-1] for i in crecs[-5000:]];

oos_set_x=[i[:-1] for i in trecs];

def sample_and_split(x,n):
 r=random.sample(x,n);
 r1=[i[:-1] for i in r];
 r2=[i[-1] for i in r];
 return r1,r2;

def rmssq(a,b): 
 v=[];
 for p in zip(a,b): v+=[(p[0]-p[1])**2];
 v=sum(v)/float(len(v));
 v=v**0.5;
 return v;

print "records read ...";
sys.stdout.flush();

clf_list=[];
errs=[];
num_models=1
for m in range(num_models): 
 k=random.random();
 typ="linear";
 clf_list+=[svm.SVR(kernel="linear")];
 x,y=sample_and_split(tr_set,60000);
 clf_list[-1].fit(x,y);
 print m,typ,;
 sys.stdout.flush();
 p=[];
 for i in te_set_x: 
  p+=[clf_list[-1].predict([i])[0]];
 perf=rmssq(p,te_set_y);
 print "model rmssq is: %5.3f" %perf;
 errs+=[perf];
 t=[];
 for j in oos_set_x: t+=[clf_list[-1].predict([j])[0]];
 f=open("svm_R4_60_"+str(m)+".csv","w");
 for j in t: f.write(str(j)+"\n");
 f.close();

print "average rmsq:",sum(errs)/float(num_models);

