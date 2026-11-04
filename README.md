# Market Forecast AI v3 — ricerca quantitativa mobile

**Versione sperimentale, non sistema di trading validato.** Sostituisce la regressione logistica singola con confronto tra tre modelli, selezione su validazione precedente e holdout finale separato. Se non emerge vantaggio, usa il benchmark.

## Deploy Streamlit
Caricare `app.py`, `engine.py`, `daily_report.py`, `requirements.txt`, `.github/workflows/daily.yml` e la cartella `tests` nella radice del repository GitHub già esistente. Mantenere il branch `main` e l'entrypoint `app.py`. Il deploy Streamlit si aggiorna al commit.

## Aggiornamenti automatici
La workflow GitHub Actions è programmata per le 06:15 UTC nei giorni feriali, **non** per le 08:00 locali tutto l'anno (06:15 UTC = 07:15 in inverno, 08:15 in estate). GitHub Actions non garantisce l'ora esatta. Archivia report JSON nel repository pubblico `reports/` se Actions ha permessi di scrittura. Non inserire informazioni personali o credenziali nei report. L'automazione ChatGPT delle 08:00 è separata e non legge questi report automaticamente.

## Limiti essenziali
- I prezzi Yahoo Finance possono avere ritardi, dati mancanti o cambiamenti retroattivi.
- Le notizie RSS **non entrano** nel modello: servirebbe uno storico affidabile con orari di pubblicazione e dati vintage macroeconomici.
- Il rendimento medio, gli intervalli 5°–95° percentile e la probabilità di perdita oltre 3% sono statistiche **storiche non condizionali**, non previsioni calibrate.
- La selezione dei modelli usa la validazione prima dell'holdout; consultare ripetutamente il test finale per progettare modifiche ne riduce l'indipendenza.
- Gli orizzonti sovrapposti richiedono cautela nell'interpretare gli intervalli bootstrap.
- La dashboard non include costi, slippage, tasse, modelli di rendimento condizionale, valutazioni fondamentali o database persistente privato.
- Gli strumenti VIX e yield Treasury non rappresentano investimenti direttamente acquistabili. Alcuni ETF sono quotati in valute diverse e hanno storici più brevi.
- Nessuna email automatica viene inviata da questa workflow.

## Test
`pip install -r requirements.txt pytest` e `pytest -q`.

## Privacy
Il repository attuale è pubblico. Evitare di caricare credenziali, API key o dati personali.
