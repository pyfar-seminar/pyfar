# TODOs: Fractional Octave Smoothing Modul

## Grundstruktur für FractionalOctaveSmoothing Objekt:
* [x] Methoden definieren (init, calc_integration_limits, calc_weights, apply, 
    smooth, ...)
* [x] Attribute definieren (Fensterbreite, Abtastfrequenz, Frequenzen an der geglättet wird, Integrationsgrenzen(?), Gewichte)
* [x] Datentypen für Gewichte und Integrationsgrenzen festlegen
* [x] Methoden zum Smoothing anhand Signal und HRTF erstellen
* [x] Get-& Set-Methoden definieren
* [x] Möglichkeit Fensterbreite zu ändern einbauen
* [x] apply() Methode: Signal als Argument, return Signal
* [x] Neuberechnung oder löschen der Gewichte, wenn Frequenzstützstellenanzahl oder Fensterbreite geändert wird -> Innerhalb der Setter von n_bins und smoothing_width
* [x] Esetze `polar2cartesian`
* [ ] ~~Limits Matrix mit scipy.sparse bauen (prüfen)~~ (Überflüssig mti neuer Loop Methode)
* [ ] ``Phase types: minimum, linear``` hinzufügen
* [x] Gewichtung über loop für jedes Frequenzbin

    ### Padding Alternativen:
    * [x] Spektrum mit konstantem Wert padden (Mittelwert über letztes Fenster nutzen)
    * [ ] Vergleich Zero, Mean und Const Padding in Notebook darstellen
    * [ ] Evtl. Mean Padding Möglichkeit in Loop Methode einbauen
    ### Ctors:
    * [x] Fractional Smoothing Objekt ohne Signal initalisieren und zurückgeben, analog zu Filter Funktionen
    * [ ] Konstruktor mit Signal (ohne n_bins), ohne Signal (mit n_bins)
    * [ ] Möglichkeit neue Daten zu übergeben einbauen (Nicht mehr nötig mit neuer loop Version)



## Signalverarbeitung
* [x] Paddingmethode für Smoothing am Rand des Spektrums überlegen
* [x] Festlegen, wann Integrationsgrenzen und Gewichte erstmalig berechnet werden, wann neue Berechnung nötig ist: Erstberechung in Konstruktor, Neuberechnung in `apply` nachdem in Setter Methoden eine Flag gesetzt wurde


## Testing:
* [x] Berechnung der Integrationsgrenzen 
* [ ] Integrationsgrenzen mit analytischer Lösung vergleichen (für einen Frequenzstützstelle)
* [x] Berechnung der Gewichte
* [x] Gewichte mit analytischer Lösung vergleichen (für einen Frequenzstützstelle)
* [x] Berechnung des geglätteten Signals
* [x] apply() Test fixen (Signal als input und return)
* [ ] Padding Type testen
    ### Test for `apply()` method:
    * [x] Test, ob Gewichte nur auf Magnitude oder auch auf Phase angewendet werden
    * [x] Test check phase handling: Original vs Zero
    * [ ] Smoothing eines mehrkanaliges Eingangssignal(HRTF) testen
    * [ ] Test, ob input signal verändert wird durch smoothing Objekt
    * [ ] Test to check if the meta-data of the output signal matches that of the input signal
    * [ ] Test, ob weights matrix neu berechnet wird (Überflüssig mti neuer Loop Methode)
    ### Error Exceptions:
    * [x] Ctors und Error Exceptions updaten
    * [x] Test Errors in `apply()` und Signal in time domain
    * [ ] Test Errors in calc_integration limits (Überflüssig mti neuer Loop Methode)

    ### Get- & Set-Methoden testen:
    * [x] Test n_bins setter
    * [x] Test smoothing_width setter
    * [x] Test phase type setter
    * [ ] ~~Test Error in data_padder~~ (Überflüssig mti neuer Loop Methode)
---
* [ ] Alle Methoden getestet: ```pytest --cov=. --cov-report=html```

## Doku:
* [x] Objekt dokumentieren
* [x] Formelreferenzen hinzufügen
* [x] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben

    ### Inhalt Jupyter Notebook:
    * Unterscheid Padding Methoden
    * Unterschied smoothing per Matrix vs smoothing per Loop
