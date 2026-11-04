"""Archive an honest daily report; GitHub Actions timing is approximate, UTC-based."""
from pathlib import Path
from datetime import datetime,timezone
import json
import yfinance as yf
import pandas as pd
from engine import MARKETS,evaluate

now=datetime.now(timezone.utc)
report={'generated_utc':now.isoformat(),'disclaimer':'Ricerca sperimentale. Non consulenza finanziaria. Nessuna previsione garantita.','markets':{}}
for market,ticker in MARKETS.items():
 try:
  d=yf.download(ticker,period='10y',interval='1d',auto_adjust=True,progress=False,threads=False)
  if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
  s=d['Close'].dropna();s.index=pd.to_datetime(s.index).tz_localize(None) if s.index.tz is None else pd.to_datetime(s.index).tz_convert(None)
  horizons={}
  for h in (1,3,7):
   try:
    r=evaluate(s,h)
    horizons[str(h)]={'probability_up':r['prob'],'selected':r['selected'],'brier':r['brier'],'benchmark_brier':r['base_brier'],'evidence':r['evidence'],'brier_gain_ci':r['ci'],'last_quote':r['last_date']}
   except Exception as e:horizons[str(h)]={'error':str(e)}
  report['markets'][market]=horizons
 except Exception as e:report['markets'][market]={'error':str(e)}
p=Path('reports');p.mkdir(exist_ok=True)
(p/(now.strftime('%Y-%m-%d')+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print('Archived',len(report['markets']),'markets')
