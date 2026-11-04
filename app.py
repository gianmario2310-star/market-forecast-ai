"""Market Forecast AI — mobile-first Streamlit research dashboard."""
from datetime import datetime, timezone
from urllib.parse import quote
import json
import sqlite3
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Market Forecast AI", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1200px}
[data-testid="stMetric"]{border:1px solid #64748b44;border-radius:12px;padding:12px}
[data-testid="stMetricValue"]{font-size:1.45rem}
@media(max-width:640px){.block-container{padding-left:.75rem;padding-right:.75rem}h1{font-size:1.75rem!important}h2{font-size:1.3rem!important}}
</style>""", unsafe_allow_html=True)

TICKERS={"S&P 500":"^GSPC","Nasdaq 100":"^NDX","Euro Stoxx 50":"^STOXX50E","FTSE MIB":"FTSEMIB.MI","DAX":"^GDAXI","EUR/USD":"EURUSD=X","Oro (USD)":"GC=F","VIX":"^VIX","Treasury USA 10Y (yield)":"^TNX"}
DB=Path(__file__).resolve().parent / 'forecasts.sqlite3'

@st.cache_data(ttl=3600,show_spinner=False)
def load(symbol):
    df=yf.download(symbol,period='10y',interval='1d',auto_adjust=True,progress=False,threads=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    if df.empty or 'Close' not in df: raise ValueError('Il fornitore prezzi non ha restituito dati. Riprova più tardi.')
    s=df['Close'].dropna().astype(float)
    idx=pd.to_datetime(s.index)
    s.index=idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
    s=s[~s.index.duplicated(keep='last')].sort_index()
    if len(s)<650: raise ValueError('Storico insufficiente per addestramento e test.')
    return s

def features(s):
    r=s.pct_change()
    x=pd.DataFrame(index=s.index)
    for n in [1,3,5,10,20]: x[f'ret{n}']=s.pct_change(n)
    for n in [5,10,20]: x[f'vol{n}']=r.rolling(n).std()
    x['ma20']=s/s.rolling(20).mean()-1
    x['ma60']=s/s.rolling(60).mean()-1
    x['dd60']=s/s.rolling(60).max()-1
    return x.replace([np.inf,-np.inf],np.nan)

def fit(x,y):
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=700,C=0.3))
    model.fit(x,y)
    return model

@st.cache_data(ttl=3600,show_spinner=False)
def forecast(s,h,threshold):
    x=features(s)
    future=s.shift(-h)/s-1
    complete=x.notna().all(axis=1)
    known=complete & future.notna()
    xv=x.loc[known]; y=(future.loc[known]>threshold).astype(int)
    if len(xv)<580 or y.nunique()<2: raise ValueError('Campione insufficiente o con una sola classe.')
    start=max(500,int(len(xv)*0.60))
    step=max(5,int((len(xv)-start)/75))
    rows=[]
    # Each training label has finished strictly before the test observation.
    for j in range(start,len(xv),step):
        cutoff=j-h
        yt=y.iloc[:cutoff]
        if yt.nunique()<2: continue
        model=fit(xv.iloc[:cutoff],yt)
        end=min(j+step,len(xv))
        probabilities=model.predict_proba(xv.iloc[j:end])[:,1]
        baseline=float(yt.mean())
        for k,p in enumerate(probabilities):
            rows.append((xv.index[j+k],float(p),int(y.iloc[j+k]),baseline,float(future.loc[xv.index[j+k]])))
    if not rows: raise ValueError('Nessuna finestra valida per il test.')
    val=pd.DataFrame(rows,columns=['date','prob','actual','baseline','return']).set_index('date')
    # Most recent known outcome is h sessions before the most recent price.
    latest=x.loc[complete].iloc[[-1]]
    model=fit(xv,y)
    p=float(model.predict_proba(latest)[0,1])
    hist=future.loc[known]
    quantiles=tuple(float(v) for v in np.quantile(hist,[.1,.5,.9]))
    return p,val,quantiles,latest.index[-1]

def store_prediction(market,h,threshold,date,p,price):
    # Local SQLite is ephemeral on some cloud hosts: not a durable shared database.
    with sqlite3.connect(DB,timeout=5) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS predictions (market TEXT, horizon INTEGER, threshold REAL, asof TEXT, probability REAL, price REAL, created_utc TEXT, PRIMARY KEY(market,horizon,threshold,asof))')
        conn.execute('INSERT OR IGNORE INTO predictions VALUES (?,?,?,?,?,?,?)',(market,h,threshold,str(date.date()),p,price,datetime.now(timezone.utc).isoformat()))

def saved_predictions(market):
    if not DB.exists(): return pd.DataFrame()
    with sqlite3.connect(DB,timeout=5) as conn:
        return pd.read_sql_query('SELECT asof, horizon, threshold, probability, price, created_utc FROM predictions WHERE market=? ORDER BY asof DESC LIMIT 150',conn,params=(market,))

@st.cache_data(ttl=1800,show_spinner=False)
def news_feed(query):
    url='https://news.google.com/rss/search?q='+quote(query)+'&hl=it&gl=IT&ceid=IT:it'
    resp=requests.get(url,timeout=9,headers={'User-Agent':'Mozilla/5.0 MarketForecastResearch/1.0'})
    resp.raise_for_status()
    root=ET.fromstring(resp.content)
    out=[]
    for item in root.findall('./channel/item')[:10]:
        out.append({'title':item.findtext('title',''), 'link':item.findtext('link',''), 'published':item.findtext('pubDate','')})
    return out

st.title('📈 Market Forecast AI')
st.caption('Dashboard mobile · dati aggiornabili · probabilità sperimentali, non segnali di acquisto/vendita')
with st.expander('⚙️ Mercato e parametri',expanded=True):
    market=st.selectbox('Mercato',list(TICKERS))
    c1,c2=st.columns(2)
    with c1: h=st.selectbox('Orizzonte (sedute)',[1,3,7],index=2)
    with c2: threshold=st.selectbox('Rialzo minimo', [0.0,0.005,0.01,0.02],format_func=lambda x:f'{x:.1%}')
    if st.button('🔄 Aggiorna i dati',use_container_width=True):
        load.clear();forecast.clear();news_feed.clear();st.rerun()

try:
    s=load(TICKERS[market]);p,val,quantiles,asof=forecast(s,h,threshold)
    brier=brier_score_loss(val.actual,val.prob)
    base_brier=brier_score_loss(val.actual,val.baseline)
    acc=accuracy_score(val.actual,val.prob>=.5)
    base_acc=accuracy_score(val.actual,val.baseline>=.5)
    st.caption(f'Ultima quotazione giornaliera disponibile: {s.index[-1].date()} · {len(val)} osservazioni di test · provider Yahoo Finance, possibile ritardo')
    tab1,tab2,tab3,tab4=st.tabs(['📊 Previsione','🧪 Verifica','🌍 Euro','📰 Notizie'])
    with tab1:
        a,b=st.columns(2)
        a.metric('Probabilità di rialzo',f'{p:.1%}',help=f'Probabilità stimata di rendimento > {threshold:.1%} in {h} sedute')
        b.metric('Ultima chiusura',f'{s.iloc[-1]:,.2f}',f'{s.pct_change().iloc[-1]:+.2%}')
        st.progress(min(1,max(0,p)),text=f'Probabilità modello: {p:.1%}')
        if brier>=base_brier:
            st.warning('⚠️ Il modello non supera il benchmark probabilistico nel test storico: previsione poco affidabile.')
        else:
            st.info('Il modello supera il benchmark nel test storico, ma questo non garantisce risultati futuri.')
        fig=go.Figure(go.Scatter(x=s.index[-504:],y=s.iloc[-504:],name='Chiusura',line=dict(width=2)))
        fig.update_layout(height=320,margin=dict(l=4,r=4,t=12,b=5),xaxis_title=None,yaxis_title=None)
        st.plotly_chart(fig,use_container_width=True)
        st.write(f'**Rendimenti storici a {h} sedute:** 10° percentile {quantiles[0]:+.2%} · mediana {quantiles[1]:+.2%} · 90° percentile {quantiles[2]:+.2%}.')
        st.caption('Questi percentili descrivono il passato: non sono limiti di rischio garantiti né intervalli previsionali calibrati.')
        try:
            store_prediction(market,h,threshold,asof,p,float(s.iloc[-1]))
            st.caption('Previsione archiviata localmente per la data del segnale. Su hosting gratuito l’archivio potrebbe essere cancellato al riavvio.')
        except (OSError,sqlite3.Error) as exc:
            st.caption(f'Archivio locale non disponibile: {exc}')
    with tab2:
        a,b=st.columns(2)
        a.metric('Brier modello',f'{brier:.3f}')
        b.metric('Brier benchmark',f'{base_brier:.3f}')
        c,d=st.columns(2)
        c.metric('Direzione corretta',f'{acc:.1%}')
        d.metric('Benchmark direzione',f'{base_acc:.1%}')
        st.caption('Brier: più basso è meglio. Il benchmark usa solo la frequenza dei rialzi nota prima di ciascuna previsione.')
        view=val[['prob','actual','baseline']].rolling(45,min_periods=20).mean()
        fig2=go.Figure()
        for col,label in [('prob','Probabilità prevista'),('actual','Frequenza osservata'),('baseline','Benchmark')]:
            fig2.add_trace(go.Scatter(x=view.index,y=view[col],name=label))
        fig2.update_layout(height=320,yaxis_tickformat='.0%',margin=dict(l=4,r=4,t=12,b=5),legend=dict(orientation='h',y=-.2))
        st.plotly_chart(fig2,use_container_width=True)
        st.download_button('⬇️ Esporta test CSV',val.to_csv().encode(),'backtest.csv','text/csv',use_container_width=True)
        try:
            archive=saved_predictions(market)
            if not archive.empty:
                with st.expander('Previsioni salvate su questo server'):
                    st.dataframe(archive,hide_index=True,use_container_width=True)
                    st.download_button('Esporta archivio',archive.to_csv(index=False).encode(),'previsioni.csv','text/csv')
        except (OSError,sqlite3.Error): pass
        st.caption('Test walk-forward con etichette purgate. Osservazioni sovrapposte e scelta dei parametri possono alterare le metriche; non è una prova di redditività operativa.')
    with tab3:
        if market in ('S&P 500','Nasdaq 100','Oro (USD)'):
            fx=load('EURUSD=X')
            common=pd.concat([s.rename('asset'),fx.rename('eurusd')],axis=1).dropna()
            if len(common)>2:
                usd=common.asset.iloc[-1]/common.asset.iloc[-2]-1
                eur=(common.asset.iloc[-1]/common.eurusd.iloc[-1])/(common.asset.iloc[-2]/common.eurusd.iloc[-2])-1
                a,b=st.columns(2)
                a.metric('Asset in USD',f'{usd:+.2%}')
                b.metric('Equivalente in EUR',f'{eur:+.2%}')
                st.caption(f'Variazione fra le ultime due date comuni ({common.index[-2].date()} → {common.index[-1].date()}); stima prima di costi e imposte. Per oro future, non equivale al rendimento di uno specifico ETF.')
        else:
            st.info('Il confronto diretto USD/EUR è disponibile per S&P 500, Nasdaq 100 e oro denominato in USD.')
    with tab4:
        st.write('Titoli recenti da Google News RSS: **non sono automaticamente interpretati dal modello quantitativo**.')
        query={'S&P 500':'S&P 500 Federal Reserve inflation','Nasdaq 100':'Nasdaq technology earnings','Euro Stoxx 50':'Euro Stoxx BCE economia','FTSE MIB':'FTSE MIB Borsa italiana','DAX':'DAX Germania economia','EUR/USD':'EUR USD BCE Federal Reserve','Oro (USD)':'gold price central banks','VIX':'VIX market volatility','Treasury USA 10Y (yield)':'US Treasury 10 year yield'}[market]
        try:
            items=news_feed(query)
            if not items: st.info('Nessun titolo restituito.')
            for item in items:
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(item['published'])
        except (requests.RequestException,ET.ParseError) as exc:
            st.warning(f'Feed notizie temporaneamente non disponibile: {exc}')
        st.markdown('[Calendario economico](https://tradingeconomics.com/calendar) · [BCE](https://www.ecb.europa.eu/press/html/index.en.html) · [Federal Reserve](https://www.federalreserve.gov/newsevents.htm)')
except Exception as exc:
    st.error(f'Impossibile elaborare il mercato selezionato: {exc}')

st.divider()
st.caption('⚠️ Ricerca sperimentale: nessuna garanzia di accuratezza, guadagno o protezione da perdite. Le notizie sono separate dalle previsioni statistiche. I mercati chiusi non aggiornano le quotazioni.')
