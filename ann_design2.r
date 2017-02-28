# package h2o does work! need to see if we can structure network differently to make
# it even better!

library(h2o);

set.seed(12345);

rssqe<-function(predictions,te_y) { (sum((as.matrix(predictions)-te_y)^2)/length(te_y))^0.5;}

h2o.init();
tr_set<-tr_dat[1:(nr-te_sz),1:nc]; # after running setup.r
sampl_set<-tr_set;

tr_set_std<-tr_set;
te_set_std<-te_set;
oos_set_std<-oos_set;

dset.hex<-as.h2o(tr_set);
tset.hex<-as.h2o(te_set);
oos.hex<-as.h2o(oos_set);

for (i in 117:130)
{ tr_set_std[,i]<-(tr_set[,i]-mean(tr_set[,i]))/sd(tr_set[,i]);
  te_set_std[,i]<-(te_set[,i]-mean(te_set[,i]))/sd(te_set[,i]);
  oos_set_std[,i]<-(oos_set[,i]-mean(oos_set[,i]))/sd(oos_set[,i]);
}

dset_std.hex<-as.h2o(tr_set_std);
tset_std.hex<-as.h2o(te_set_std);
oos_std.hex<-as.h2o(oos_set_std);

writepred<-function(pred,mname)
 { write.csv(pred,file=paste(mname,".csv",collapse=NULL),row.names=FALSE);}

ann_pred<-function(model,modelstr)
{ predictions<-h2o.predict(model,tset.hex);
  x<-cbind(as.matrix(predictions),te_y);
  str<-sprintf("ann model %s rms_error: %5.3f",modelstr,rssqe(predictions,te_y));
  print(str);
  predictions<-h2o.predict(model,oos.hex);
  predictions<-as.matrix(predictions);
  writepred(predictions,modelstr);
}


ann1<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,seed=12345);
std_ann1<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,seed=12345);
ann2<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
std_ann2<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann3<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,10),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
std_ann3<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,10),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann4<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,20,10),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
std_ann4<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,20,10),stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann5<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,20,10,3,10),activation="Tanh",stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
std_ann5<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,20,10,3,10),activation="Tanh",stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann6<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,20,10,3,10,20),activation="Tanh",stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
std_ann6<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,20,10,3,10,20),activation="Tanh",stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann7<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,20,10,3,10,20),activation="TanhWithDropout",input_dropout_ratio=0.2,hidden_dropout_ratios=c(0.2,0.2,0.2,0.2,0.2,0.2,0.2),stopping_metric="MSE",nfolds=10,seed=12345,epochs=10,standardize=TRUE);
std_ann7<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,20,10,3,10,20),activation="TanhWithDropout",input_dropout_ratio=0.2,hidden_dropout_ratios=c(0.2,0.2,0.2,0.2,0.2,0.2,0.2),stopping_metric="MSE",nfolds=10,seed=12345,epochs=10,standardize=TRUE);
ann8<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=c(60,40,20,10,3,10),activation="Tanh",stopping_metric="AUTO",nfolds=10,seed=12345,standardize=TRUE);
std_ann8<-h2o.deeplearning(x=1:130,y=131,training_frame=dset_std.hex,hidden=c(60,40,20,10,3,10),activation="Tanh",stopping_metric="AUTO",nfolds=10,seed=12345,standardize=TRUE);

ann_pred(ann1,"ann1");
ann_pred(std_ann1,"std_ann1");

ann_pred(ann2,"ann2");
ann_pred(std_ann2,"std_ann2");

ann_pred(ann3,"ann3");
ann_pred(std_ann3,"std_ann3");

ann_pred(ann4,"ann4");
ann_pred(std_ann4,"std_ann4");

ann_pred(ann5,"ann5");
ann_pred(std_ann5,"std_ann5");

ann_pred(ann6,"ann6");
ann_pred(std_ann6,"std_ann6");

ann_pred(ann7,"ann7");
ann_pred(std_ann7,"std_ann7");

ann_pred(ann8,"ann8");
ann_pred(std_ann8,"std_ann7");


ann_R1<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=143,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R1,"ann_R1");

ann_R2<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=88,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R2,"ann_R2");

ann_R3<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=66,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R3,"ann_R3");

ann_R4<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=50,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R4,"ann_R4");

"ann model ann_R1 rms_error: 1930.488"
"ann model ann_R2 rms_error: 1954.363"
"ann model ann_R3 rms_error: 1921.604"
"ann model ann_R4 rms_error: 1946.394"

ann_R5<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=66,activation="TanhWithDropout",input_dropout_ratio=0.2,hidden_dropout_ratios=0.2,stopping_metric="MSE",nfolds=10,seed=12345,epochs=10,standardize=TRUE);
ann_pred(ann_R5,"ann_R5");

"ann model ann_R5 rms_error: 2028.656"

ann_R6<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=66,input_dropout_ratio=0.2,stopping_metric="AUTO",nfolds=10,seed=12345,epochs=10,standardize=TRUE);
ann_pred(ann_R6,"ann_R6");
ann_R7<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=66,input_dropout_ratio=0.4,stopping_metric="MSE",nfolds=10,seed=12345,epochs=10,standardize=TRUE);
ann_pred(ann_R7,"ann_R7");

ann_R8<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=66,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R8,"ann_R8");

ann_R9<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=30,stopping_metric="MSE",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_R9,"ann_R9");

ann_RA<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=30,stopping_metric="AUTO",nfolds=10,seed=12345,standardize=TRUE);
ann_pred(ann_RA,"ann_RA");


build_ann<-function(nsampl=75000,nhid=66,smetric="AUTO",nfld=10,modlstr)
{ #tr_set2<-tr_dat[sample(nrow(sampl_set),nsampl,replace=TRUE),];
  tr_set2<-tr_dat[sample(nrow(sampl_set),nsampl),];
  dset.hex<-as.h2o(tr_set2);
  ann_Rx<-h2o.deeplearning(x=1:130,y=131,training_frame=dset.hex,hidden=nhid,stopping_metric=smetric,nfolds=nfld,seed=12345,standardize=TRUE);
  ann_pred(ann_Rx,modlstr);
}

x<-sample(40:80,20);
for (i in x) 
 { modlstr<-paste0("ann_R3_",i);
   ssz<-sample(90000:110000,1);
   print(sprintf("modlstr: %s, nhid: %5.3f sampl: %5.3f",modlstr,i,ssz));
   build_ann(nsampl=ssz,nhid=i,smetric="AUTO",nfld=10,modlstr);
 }

