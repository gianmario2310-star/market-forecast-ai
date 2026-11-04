# Market Forecast AI — dashboard mobile (v2)

App Streamlit responsive per smartphone Android/iPhone e computer. Nessuna chiave API richiesta per i dati di base; disponibilità e ritardi dipendono da Yahoo Finance e Google News RSS.

## Pubblicazione sul telefono (consigliata)

1. Crea un repository **privato** GitHub e carica i file di questa cartella (inclusa `.streamlit/config.toml`). Non caricare `forecasts.sqlite3`.
2. Apri https://share.streamlit.io/ e accedi con GitHub.
3. Clicca **Create app**, seleziona il repository, il branch `main` e come entrypoint `app.py`, poi **Deploy**.
4. Attendi l'installazione automatica delle dipendenze e apri l'URL HTTPS risultante sul telefono. Puoi aggiungerlo alla schermata Home dal browser.
5. **Attenzione privacy:** un repository privato non rende necessariamente privata l'app pubblicata. Controlla le impostazioni di accesso del servizio hosting prima di condividere dati sensibili.

La pubblicazione richiede l'accesso al tuo account GitHub e Streamlit Cloud: non è stata eseguita da ChatGPT. Gli URL definitivi non sono ancora disponibili.

## Avvio su PC

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Funzioni

- Mercati S&P 500, Nasdaq 100, Euro Stoxx 50, FTSE MIB, DAX, EUR/USD, oro, VIX, rendimento Treasury USA 10Y.
- Previsioni probabilistiche sperimentali a 1/3/7 **sedute**, non giorni di calendario.
- Backtest walk-forward con etichette purgate e benchmark causale; Brier score e accuratezza.
- Grafici mobili, export CSV, impatto EUR/USD per strumenti in USD.
- RSS di titoli economici recenti (non ancora input del modello quantitativo).
- Archivio locale SQLite: persistente solo dove il disco è persistente; su Streamlit Community Cloud può essere cancellato ai riavvii e non equivale a un database affidabile.

## Limiti tecnici e scientifici

La previsione usa un modello di regressione logistica su indicatori di prezzo. Non è un modello causale né un sistema di trading validato. Il backtest è una valutazione storica parziale e non include costi, slippage, intervalli di confidenza, ottimizzazione senza leakage o un test esterno indipendente. Le osservazioni con orizzonti sovrapposti non sono indipendenti. I percentili dei rendimenti sono descrittivi, non intervalli calibrati. L'app aggiorna i dati quando viene aperta/ricaricata e dopo la scadenza della cache (1 ora), **non esegue aggiornamenti autonomi quando è chiusa**.

Il rapporto ChatGPT delle 8:00 è un'automazione **separata**: non alimenta questa dashboard. Per un sistema completamente integrato occorrono hosting, API affidabili, database remoto persistente, job schedulati e credenziali configurate dall'utente.
