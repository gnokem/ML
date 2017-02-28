import os, sys, random, csv;
from sklearn.neural_network import MLPRegressor as MLPR;

seed=13
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

def sample_and_split(x,n):
 #r=random.sample(x,n);
 r=[random.choice(x) for i in range(n)];
 r1=[i[:-1] for i in r];
 r2=[i[-1] for i in r];
 return r1,r2;

print "records read ...";
sys.stdout.flush();

models=[];
def build_and_run_ann(T,fn,n):
 global models;
 sz=200000;
 x,y=sample_and_split(tr_set,sz);
 ann=MLPR(hidden_layer_sizes=T,activation=fn,warm_start=True,early_stopping=True);
 ann.fit(x,y);
 p=ann.predict(te_set_x);
 p=list(p);
 e=rmssq(p,te_set_y);
 print "for ann ",T," rms-error=",e;
 sys.stdout.flush();
 k=ann.predict(oos_set_x);
 k=list(k);
 f=open("sklearn_ann_"+fn+"_"+str(n)+".csv","w");
 for i in k: f.write(str(round(i,3))+"\n");
 f.close();
 models+=[ann];
 return;

build_and_run_ann((100,66,44,30,20,12,8,),"relu",36);
build_and_run_ann((100,44,30,20,12,8,),"relu",37);
build_and_run_ann((100,66,30,20,12,8,),"relu",38);
build_and_run_ann((100,66,44,20,12,8,),"relu",39);
build_and_run_ann((100,66,44,30,12,8,),"relu",40);
build_and_run_ann((100,66,44,30,20,8,),"relu",41);
build_and_run_ann((100,66,44,30,20,12,),"relu",42);
build_and_run_ann((100,80,66,44,30,20,12,8,),"relu",43);
build_and_run_ann((100,66,20,10,3,10,),"relu",44);
build_and_run_ann((66,44,30,20,12,8,),"relu",45);
build_and_run_ann((66,66,44,44,30,30,),"relu",46);
build_and_run_ann((100,100,44,44,30,30,),"relu",47);
build_and_run_ann((66,66,44,44,30,30,),"relu",48);
build_and_run_ann((66,66,44,44,13,13,),"relu",49);
build_and_run_ann((67,66,43,41,17,13,),"relu",50);
build_and_run_ann((79,73,67,61,59,53,),"relu",51);
build_and_run_ann((47,43,41,37,31,29,),"relu",52);
build_and_run_ann((29,23,19,17,13,7,),"relu",53);

