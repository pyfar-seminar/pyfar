# TODOs: Fractional Octave Smoothing Modul

## Grundstruktur für FractionalOctaveSmoothing Objekt:
* [x] Methoden definieren (init, calc_integration_limits, calc_weights, apply, 
    smooth, ...)
* [x] Attribute definieren (Fensterbreite, Abtastfrequenz, Frequenzen an der geglättet wird, Integrationsgrenzen(?), Gewichte)
* [x] Datentypen für Gewichte und Integrationsgrenzen festlegen
* [x] Methoden zum Smoothing anhand Signal und HRTF erstellen
* [x] Get-& Set-Methoden definieren
* [ ] Möglichkeit neue Daten zu übergeben einbauen
* [x] Möglichkeit Fensterbreite zu ändern einbauen
* [ ] Limits Matrix mit scipy.sparse bauen (prüfen)
* [x] apply() Methode: Signal als Argument, return Signal
* [ ] Konstruktor mit Signal (ohne n_bins), ohne Signal (mit n_bins)
* [ ] Fractional Smoothing Objekt ohne Signal initalisieren und zurückgeben, analog zu Filter Funktionen
* [x] Neuberechnung oder löschen der Gewichte, wenn Frequenzstützstellenanzahl oder Fensterbreite geändert wird -> Innerhalb der Setter von n_bins und smoothing_width
* [ ] Limits Matrix mit scipy.sparse bauen

## Signalverarbeitung
* [x] Paddingmethode für Smoothing am Rand des Spektrums überlegen
* [ ] Festlegen, wann Integrationsgrenzen und Gewichte erstmalig berechnet 
    werden, wann neue Berechnung nötig ist


## Testing:
* [x] Ctors und Error Exceptions testen
* [x] Berechnung der Integrationsgrenzen 
* [ ] Integrationsgrenzen mit analytischer Lösung vergleichen (für einen Frequenzstützstelle)
* [x] Berechnung der Gewichte
* [x] Gewichte mit analytischer Lösung vergleichen (für einen Frequenzstützstelle)
* [x] Berechnung des geglätteten Signals
* [ ] apply() fixen
* [ ] Get- & Set-Methoden testen
* [ ] Smoothing eines mehrkanaliges Eingangssignal(HRTF) testen
* [ ] Alle Methoden getestet?

## Doku:
* [x] Objekt dokumentieren
* [x] Formelreferenzen hinzufügen
* [x] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben
