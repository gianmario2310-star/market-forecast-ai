import json
from datetime import datetime,timezone
from urllib.parse import quote
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from engine import MARKETS, USD_ASSETS, evaluate

st.set_page_config(page_title='Market Forecast AI v3',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
st.markdown('<style>.block-container{padding-top:1rem;max-width:1100px}@media(max-width:640px){.block-container{padding-left:.7rem;padding-right:.7rem}h1{font-size:1.65rem!important}}</style>',unsafe_allow_html=True)
@st.cache_data(ttl=3600,show_spinner=False)
def prices(ticker):
 d=yf.download(ticker,period='10y',interval='1d',auto_adjust=True,progress=False,threads=False)
 if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
 if d.empty or 'Close' not in d:raise ValueError('Provider prezzi non disponibile')
 s=d.Close.dropna().astype(float);s.index=pd.to_datetime(s.index).tz_localize(None) if s.index.tz is None else pd.to_datetime(s.index).tz_convert(None)
 return s[~s.index.duplicated(keep='last')].sort_index()
@st.cache_data(ttl=3600,show_spinner=False)
def compute(s,h,t):return evaluate(s,h,t)
@st.cache_data(ttl=1800,show_spinner=False)
def news(q):
 u='https://news.google.com/rss/search?q='+quote(q)+'&hl=it&gl=IT&ceid=IT:it'
 r=requests.get(u,timeout=10,headers={'User-Agent':'Mozilla/5.0'});r.raise_for_status()
 return [{'title':v.findtext('title',''),'link':v.findtext('link',''),'date':v.findtext('pubDate','')} for v in ET.fromstring(r.content).findall('./channel/item')[:12]]
st.title('📈 Market Forecast AI · v3')
st.caption('Ricerca quantitativa mobile · selezione su validazione precedente · test finale separato')
with st.expander('⚙️ Mercato e parametri',expanded=True):
 market=st.selectbox('Mercato',list(MARKETS))
 a,b=st.columns(2)
 with a:h=st.selectbox('Orizzonte (sedute)',[1,3,7],index=2)
 with b:t=st.selectbox('Rialzo minimo',[0.,.005,.01,.02],format_func=lambda x:f'{x:.1%}')
 if st.button('🔄 Aggiorna dati',use_container_width=True):prices.clear();compute.clear();news.clear();st.rerun()
try:
 with st.spinner('Calcolo e verifica fuori campione…'):
  s=prices(MARKETS[market]);result=compute(s,h,t)
 st.caption(f'Ultima quotazione: {result["last_date"]} · test finale: {result["n_test"]} osservazioni · dati Yahoo Finance, possibili ritardi')
 tabs=st.tabs(['📊 Previsione','🧪 Verifica','🌍 Euro','📰 Notizie','📁 Esporta'])
 with tabs[0]:
  c,d=st.columns(2);c.metric('Probabilità rialzo',f'{result["prob"]:.1%}');d.metric('Probabilità non rialzo',f'{1-result["prob"]:.1%}')
  st.write('**Modello selezionato sulla validazione:**',result['selected'])
  if result['evidence']:st.info('Miglioramento del Brier nel test finale: intervallo bootstrap sopra zero. Evidenza preliminare, non garanzia futura.')
  else:st.warning('NESSUN VANTAGGIO PREDITTIVO DIMOSTRATO sul test finale. Non utilizzare la probabilità come segnale operativo.')
  st.metric('Rendimento medio storico sul periodo',f'{result["expected_return_historical"]:+.2%}')
  st.write(f'Intervallo storico descrittivo 5°–95° percentile: **{result["historical_range"][0]:+.2%} / {result["historical_range"][1]:+.2%}**')
  st.write(f'Frequenza storica di perdite oltre il 3% in {h} sedute: **{result["historical_tail"]:.1%}**')
  st.caption('Rendimento e intervallo NON sono previsioni condizionali calibrate. La probabilità di rialzo è sperimentale.')
  f=go.Figure(go.Scatter(x=s.index[-504:],y=s.iloc[-504:],name='Chiusura'));f.update_layout(height=310,margin=dict(l=5,r=5,t=8,b=5));st.plotly_chart(f,use_container_width=True)
 with tabs[1]:
  c,d=st.columns(2);c.metric('Brier modello',f'{result["brier"]:.4f}');d.metric('Brier benchmark',f'{result["base_brier"]:.4f}')
  c,d=st.columns(2);c.metric('Direzione corretta',f'{result["accuracy"]:.1%}');d.metric('Benchmark direzione',f'{result["base_accuracy"]:.1%}')
  st.metric('Log loss',f'{result["logloss"]:.4f}')
  st.write(f'Intervallo bootstrap 95% sul miglioramento Brier: **{result["ci"][0]:+.4f} / {result["ci"][1]:+.4f}**')
  st.caption('Il test finale non è usato per selezionare il modello. Osservazioni sovrapposte e ricerche ripetute riducono la forza statistica dei risultati.')
  df=result['holdout'];rolling=df[['prob','actual','benchmark']].rolling(45,min_periods=20).mean()
  f=go.Figure()
  for col in rolling:f.add_trace(go.Scatter(x=rolling.index,y=rolling[col],name=col))
  f.update_layout(height=310,yaxis_tickformat='.0%',margin=dict(l=5,r=5,t=8,b=5));st.plotly_chart(f,use_container_width=True)
  st.write('Brier in validazione (NON test finale):',result['validation_scores']);st.write('Benchmark validazione:',round(result['validation_benchmark'],4))
 with tabs[2]:
  if market in USD_ASSETS:
   fx=prices('EURUSD=X');aligned=pd.concat([s.rename('asset'),fx.rename('fx')],axis=1).dropna()
   if len(aligned)>2:
    retusd=aligned.asset.iloc[-1]/aligned.asset.iloc[-2]-1
    reteuro=(aligned.asset.iloc[-1]/aligned.fx.iloc[-1])/(aligned.asset.iloc[-2]/aligned.fx.iloc[-2])-1
    c,d=st.columns(2);c.metric('Ultima variazione USD',f'{retusd:+.2%}');d.metric('Equivalente EUR',f'{reteuro:+.2%}')
   st.caption('Conversione indicativa: non rappresenta il rendimento netto di un ETF o fondo specifico.')
  else:st.info('Conversione disponibile per gli strumenti denominati in USD.')
 with tabs[3]:
  st.caption('Notizie aggiornate: NON sono input del modello, perché manca un archivio storico con timestamp verificabili.')
  try:
   for item in news(market+' economia mercati'):
    st.markdown(f'[{item["title"]}]({item["link"]})');st.caption(item['date'])
  except (requests.RequestException,ET.ParseError) as e:st.warning(f'Feed non disponibile: {e}')
  st.markdown('[Calendario macro](https://tradingeconomics.com/calendar) · [BCE](https://www.ecb.europa.eu/press/html/index.en.html) · [Fed](https://www.federalreserve.gov/newsevents.htm)')
 with tabs[4]:
  st.download_button('⬇️ Esporta test CSV',result['holdout'].to_csv().encode(),f'test_{h}.csv','text/csv',use_container_width=True)
  summary={k:v for k,v in result.items() if k not in ('holdout',)}
  st.download_button('⬇️ Esporta previsione JSON',json.dumps({'market':market,'horizon':h,'threshold':t,'generated_utc':datetime.now(timezone.utc).isoformat(),**summary},ensure_ascii=False,indent=2).encode(),f'forecast_{h}.json','application/json',use_container_width=True)
except Exception as e:st.error(f'Impossibile completare il calcolo: {e}')
st.divider();st.caption('⚠️ Sistema di ricerca, non consulenza finanziaria. Nessuna accuratezza o redditività garantita. Nessuna integrazione automatica di notizie o dati macro nel modello finché non è disponibile uno storico point-in-time affidabile.')
