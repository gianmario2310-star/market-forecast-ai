"""Point-in-time daily market research engine. No trading recommendation."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss

MARKETS = {
 'S&P 500':'^GSPC','Nasdaq 100':'^NDX','Euro Stoxx 50':'^STOXX50E',
 'FTSE MIB':'FTSEMIB.MI','DAX':'^GDAXI','Mercati emergenti (ETF USD)':'EEM',
 'Treasury USA (ETF USD)':'IEF','Obbligazioni Europa (ETF EUR)':'IEAG.AS',
 'Oro (USD)':'GC=F','Petrolio WTI (USD)':'CL=F','VIX':'^VIX',
 'EUR/USD':'EURUSD=X','Treasury USA 10Y (yield)':'^TNX'}
USD_ASSETS={'S&P 500','Nasdaq 100','Mercati emergenti (ETF USD)','Treasury USA (ETF USD)','Oro (USD)','Petrolio WTI (USD)'}

def indicators(s):
 s=s.astype(float).dropna().sort_index(); r=s.pct_change(fill_method=None)
 x=pd.DataFrame(index=s.index)
 for n in (1,3,5,10,20,60): x[f'ret_{n}']=s.pct_change(n,fill_method=None)
 for n in (5,10,20,60):
  x[f'vol_{n}']=r.rolling(n).std()
  x[f'ma_{n}']=s/s.rolling(n).mean()-1
 x['dd_60']=s/s.rolling(60).max()-1
 x['vol_ratio']=x['vol_5']/x['vol_60'].replace(0,np.nan)
 return x.replace([np.inf,-np.inf],np.nan)

def _models():
 return {
  'Logistica':lambda:make_pipeline(StandardScaler(),LogisticRegression(C=.15,max_iter=1000)),
  'Random Forest':lambda:RandomForestClassifier(n_estimators=120,max_depth=3,min_samples_leaf=45,max_features=.65,random_state=42,n_jobs=1),
  'Gradient Boosting':lambda:HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=5,l2_regularization=15,learning_rate=.035,min_samples_leaf=45,random_state=42)
 }

def _block_bootstrap_ci(diff, block=14, n=300, seed=42):
 """CI for improvement in Brier: baseline loss minus model loss."""
 a=np.asarray(diff,dtype=float); rng=np.random.default_rng(seed)
 if len(a)<30:return (float('nan'),float('nan'))
 k=int(np.ceil(len(a)/block)); vals=[]
 for _ in range(n):
  starts=rng.integers(0,max(1,len(a)-block+1),size=k)
  ix=np.concatenate([np.arange(t,min(t+block,len(a))) for t in starts])[:len(a)]
  vals.append(a[ix].mean())
 return tuple(np.quantile(vals,[.025,.975]))

def evaluate(s,h=7,threshold=0.0,minimum=650,step=21):
 """Expanding walk-forward; test block labels are unknown until h sessions later.
 A conservative h-session embargo is applied before each test block.
 Candidate selection uses earlier validation only; final holdout is never used to choose.
 """
 if h not in (1,3,7):raise ValueError('Orizzonte non supportato')
 x=indicators(s)
 future=s.shift(-h)/s-1
 z=x.join(future.rename('target')).dropna()
 if len(z)<minimum+180:raise ValueError('Storico insufficiente: servono almeno 830 osservazioni utilizzabili.')
 cols=list(x.columns); X=z[cols]; y=(z.target>threshold).astype(int)
 # Last 25% is held out for final evaluation. All selection uses earlier folds.
 boundary=max(minimum,int(len(z)*.75)); boundary=min(boundary,len(z)-90)
 def walk(begin,end,names):
  out=[]
  for j in range(begin,end,step):
   cutoff=j-h # training labels ending at or after test start are excluded
   if cutoff<300:continue
   train_y=y.iloc[:cutoff]
   if train_y.nunique()<2:continue
   test_end=min(j+step,end)
   baseline=float(train_y.mean())
   fitted={}
   for name in names:
    try:
     m=_models()[name]();m.fit(X.iloc[:cutoff],train_y)
     fitted[name]=np.clip(m.predict_proba(X.iloc[j:test_end])[:,1],.001,.999)
    except Exception:pass
   for t in range(test_end-j):
    row={'date':z.index[j+t],'actual':int(y.iloc[j+t]),'return':float(z.target.iloc[j+t]),'benchmark':baseline}
    for name,arr in fitted.items():row[name]=float(arr[t])
    out.append(row)
  return pd.DataFrame(out).set_index('date') if out else pd.DataFrame()
 validation=walk(minimum,boundary,list(_models()))
 if validation.empty:raise ValueError('Nessuna finestra di validazione')
 scores={n:float(brier_score_loss(validation.actual,validation[n])) for n in _models() if n in validation and validation[n].notna().all()}
 base_validation=float(brier_score_loss(validation.actual,validation.benchmark))
 best=min(scores,key=scores.get) if scores else None
 # Selection is made ONLY on validation, with a minimum margin; otherwise use baseline.
 selected=best if best and scores[best]<base_validation-.001 else 'Benchmark'
 holdout=walk(boundary,len(z),[selected] if selected!='Benchmark' else [])
 if holdout.empty:raise ValueError('Nessuna finestra di test')
 holdout['prob']=holdout[selected] if selected!='Benchmark' and selected in holdout else holdout.benchmark
 holdout['loss_model']=(holdout.prob-holdout.actual)**2
 holdout['loss_benchmark']=(holdout.benchmark-holdout.actual)**2
 gain=holdout.loss_benchmark-holdout.loss_model
 ci=_block_bootstrap_ci(gain,block=max(14,2*h))
 # A strictly positive lower CI is necessary, not sufficient, to assert evidence of skill.
 evidence=bool(selected!='Benchmark' and np.isfinite(ci[0]) and ci[0]>0)
 # Latest model trains only on fully realized labels; no use of pending outcomes.
 latest=indicators(s).dropna().iloc[[-1]]
 if selected=='Benchmark': p=float(y.mean())
 else:
  m=_models()[selected]();m.fit(X,y);p=float(m.predict_proba(latest)[0,1])
 # Conditional empirical return estimate, not a calibrated forecast interval.
 hist=z.target
 mu=float(hist.mean()); lo,hi=np.quantile(hist,[.05,.95])
 # Tail probability defined relative to empirical distribution and horizon.
 tail=float((hist<-.03).mean())
 return {'selected':selected,'prob':p,'validation_scores':scores,'validation_benchmark':base_validation,
  'holdout':holdout,'brier':float(holdout.loss_model.mean()),'base_brier':float(holdout.loss_benchmark.mean()),
  'logloss':float(log_loss(holdout.actual,holdout.prob,labels=[0,1])),
  'accuracy':float(((holdout.prob>=.5)==holdout.actual).mean()),
  'base_accuracy':float(((holdout.benchmark>=.5)==holdout.actual).mean()),
  'ci':ci,'evidence':evidence,'expected_return_historical':mu,
  'historical_range':(float(lo),float(hi)),'historical_tail':tail,
  'last_date':str(s.index[-1].date()),'n_test':len(holdout)}
