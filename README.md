# Swing Scanner V1

Darmowy prototyp skanera swingowego.

## Założenia
- Weekly / Daily / 4H / 1H
- bez EMA
- cena + market structure jako podstawa
- SMA 50/200 tylko jako filtr
- Relative Strength
- wolumen
- ATR
- breakout / pullback
- ranking setupów

## Uruchomienie lokalne
```bash
pip install yfinance pandas
python scanner.py
```

## GitHub Pages
Wrzuć repozytorium na GitHub, włącz Settings -> Pages -> Deploy from branch -> main / root.

Workflow `.github/workflows/scan.yml` aktualizuje `data.json` automatycznie co 4 godziny w dni robocze.

> Uwaga: darmowe dane Yahoo Finance są dobre do prototypu i researchu, ale nie traktujemy ich jako profesjonalnego feedu egzekucyjnego.
