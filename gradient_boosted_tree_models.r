setwd("fakepath:\\allstate");
tr_dat<-read.csv("lenc.csv",stringsAsFactors=FALSE,header=TRUE);
ts_dat<-read.csv("lenc_test.csv",stringsAsFactors=FALSE,header=TRUE);

nc<-dim(tr_dat)[2];
nr<-dim(tr_dat)[1];

te_sz<-15000;
tr_set<-tr_dat[1:(nr-te_sz),1:(nc-1)];
tr_y<-tr_dat[1:(nr-te_sz),nc];
t_set<-tail(tr_dat,5000);
te_set<-t_set[,1:(nc-1)];
te_y<-t_set[,nc];

nc_oos<-dim(ts_dat)[2];
nr_oos<-dim(ts_dat)[1];
oos_set<-ts_dat[,1:130];

rssqe<-function(predictions,te_y) { (sum((as.matrix(predictions)-te_y)^2)/length(te_y))^0.5;}
writepred<-function(pred,mname)
 { write.csv(pred,file=paste0(mname,".csv"),row.names=FALSE);}

library(h2o);
set.seed(12345);

h2o.init();
tr_set<-tr_dat[1:(nr-te_sz),1:nc]; # after running setup.r
sampl_set<-tr_set;

tr_set_std<-tr_set;
te_set_std<-te_set;
oos_set_std<-oos_set;

dset.hex<-as.h2o(tr_set);
tset.hex<-as.h2o(te_set);
oos.hex<-as.h2o(oos_set);

ann_pred<-function(model,modelstr)
{ predictions<-h2o.predict(model,tset.hex);
  x<-cbind(as.matrix(predictions),te_y);
  str<-sprintf("ann model %s rms_error: %5.3f",modelstr,rssqe(predictions,te_y));
  print(str);
  predictions<-h2o.predict(model,oos.hex);
  predictions<-as.matrix(predictions);
  writepred(predictions,modelstr);
}

build_ann<-function(nsampl=75000,nhid=66,smetric="AUTO",nfld=10,modlstr)
{ #tr_set2<-tr_dat[sample(nrow(sampl_set),nsampl,replace=TRUE),];
  tr_set2<-tr_dat[sample(nrow(sampl_set),nsampl),];
  dset.hex<-as.h2o(tr_set2);
  ann_Rx<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=nhid,stopping_metric=smetric,nfolds=nfld,seed=12345,standardize=TRUE);
  ann_pred(ann_Rx,modlstr);
}

x<-sample(40:80,20);
x2<-sample(7:13,20,replace=TRUE);
for (i in 1:2)
 { modlstr<-paste0("ann_R4_",x[i],"_",x2[i]);
   ssz<-sample(120000:140000,1);
   print(sprintf("modlstr: %s, nhid: %3d, %3d sampl: %5d",modlstr,x[i],x2[i],ssz));
   build_ann(nsampl=ssz,nhid=c(x[i],x2[i]),smetric="AUTO",nfld=10,modlstr);
 }

library(xgboost);
set.seed(12345);

xgb_pred<-function(model,modelstr)
{ predictions<-predict(model,as.matrix(te_set));
  x<-cbind(as.matrix(predictions),te_y);
  str<-sprintf("*** gbm model %s rms_error: %5.3f",modelstr,rssqe(predictions,te_y));
  print(str);
  predictions<-predict(model,as.matrix(oos_set));
  predictions<-as.matrix(predictions);
  writepred(predictions,modelstr);
}

build_xgb<-function(nsampl=75000,modlstr,nrnd)
 { tr_set2<-tr_dat[sample(nrow(tr_dat),nsampl),];
   tr_x2<-tr_set2[,1:(nc-1)];
   tr_y2<-tr_set2[,nc];
   xgbm<-xgboost(data=as.matrix(tr_x2),label=tr_y2,nrounds=i,eta=0.01,verbose=0);
   #xgbm<-xgboost(data=as.matrix(tr_x2),label=tr_y2,nrounds=i,verbose=0);
   xgb_pred(xgbm,modstr);
 }

for (i in seq(2808,2900))
 { for (j in seq(1,3))
   {
    rstr<-round(runif(1,100000,999999));
    modstr<-paste0("xgbm#_R4_",i,"_",rstr);
    print(modstr);
    build_xgb(120000,modstr,i);
   }
 }

### try_20161127
for (i in seq(2773,2900))
 {  rstr<-round(runif(1,100000,999999));
    modstr<-paste0("xgbm#_R5_",i,"_",rstr);
    print(modstr);
    build_xgb(173318,modstr,i);
 }
