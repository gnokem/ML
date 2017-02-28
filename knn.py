import numpy, csv, random;
from sklearn.neighbors import KNeighborsRegressor as knn;

seed=13
numpy.random.seed(seed);


def isNum(s):
 digits=[str(i) for i in range(0,10)]+['.'];
 for i in s: 
  if i not in digits: return False;
 if i.count('.')>1: return False;
 return True; 

def getCol(x,n): return [i[n] for i in x];

def strit(m): return [str(i) for i in m];

def rmssq(y1,y2): 
 r=[];
 for z in zip(y1,y2): r+=[(z[0]-z[1])**2];
 r=sum(r)/float(len(y1));
 return r**0.5;
 

recs=[];
with open("lenc.csv","r") as csvf:
 rdr=csv.reader(csvf);
 for row in rdr: recs+=[row];

csvf.close();

crecs=[];
for i in recs[1:]: # take off the header
 t=[j if not isNum(j) else float(j) for j in i];
 crecs+=[t];

def sample_and_split(x,n):
 r=random.sample(x,n);
 r1=[i[:-1] for i in r];
 r2=[i[-1] for i in r];
 return r1,r2;

#Xtr=[i[:-1] for i in crecs[:-15000]];
#ytr=[i[-1] for i in crecs[:-15000]];
neigh=knn(n_neighbors=3,weights="distance");
xs,ys=sample_and_split(crecs[:-15000],50000);
#neigh.fit(Xtr,ytr);
neigh.fit(xs,ys);

Xvl=[i[:-1] for i in crecs[-15000:]];
yvl=[i[-1] for i in crecs[-15000:]];

preds=[];
for i in Xvl: preds+=[neigh.predict([i])];

print "rms error=",rmssq(preds,yvl);

