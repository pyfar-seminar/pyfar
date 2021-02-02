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
* [ ] Limits Matrix mit scipy.sparse bauen (prüfen)
* [ ] ``Phase types: minimum, linear``` hinzufügen
    ### Ctors:
    * [ ] Konstruktor mit Signal (ohne n_bins), ohne Signal (mit n_bins)
    * [ ] Fractional Smoothing Objekt ohne Signal initalisieren und zurückgeben, analog zu Filter Funktionen
    * [ ] Möglichkeit neue Daten zu übergeben einbauen



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
    ### Test for `apply()` method:
    * [ ] Smoothing eines mehrkanaliges Eingangssignal(HRTF) testen
    * [ ] Test, ob Gewichte nur auf Magnitude oder auch auf Phase angewendet werden
    * [ ] Test check phase handling: Original vs Zero
    * [ ] Test, ob input signal verändert wird durch smoothing Objekt
    * [ ] Test to check if the meta-data of the output signal matches that of the input signal
    * [ ] Test, ob weights matrix neu berechnet wird
    ### Error Exceptions:
    * [ ] Ctors und Error Exceptions updaten
    * [ ] Test Errors in calc_integration limits
    * [ ] Test Errors in `apply()` und Signal in time domain

    ### Get- & Set-Methoden testen:
    * [ ] Test n_bins setter
    * [ ] Test smoothing_width setter
    * [ ] Test phase type setter
    * [ ] Test Error in data_padder
---
* [ ] Alle Methoden getestet: ```pytest --cov=. --cov-report=html```

## Doku:
* [x] Objekt dokumentieren
* [x] Formelreferenzen hinzufügen
* [x] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben
