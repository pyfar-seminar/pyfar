# TODOs: Fractional Octave Smoothing Modul

## Grundstruktur für FractionalOctaveSmoothing Objekt:
* [x] Methoden definieren (init, calc_integration_limits, calc_weights, apply, 
    smooth, ...)
* [x] Attribute definieren (Fensterbreite, Abtastfrequenz, Frequenzen an der geglättet wird, Integrationsgrenzen(?), Gewichte)
* [x] Datentypen für Gewichte und Integrationsgrenzen festlegen
* [x] Methoden zum Smoothing anhand Signal und HRTF erstellen
* [ ] Get-& Set-Methoden definieren
* [ ] Möglichkeit neue Daten zu übergeben einbauen
* [ ] Möglichkeit Fensterbreite zu ändern einbauen
* [ ] Limits Matrix mit scipy.sparse bauen (prüfen) 

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
* [ ] Get- & Set-Methoden testen
* [ ] Smoothing eines mehrkanaliges Eingangssignal(HRTF) testen
* [ ] Alle Methoden getestet?

## Doku:
* [x] Objekt dokumentieren
* [x] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben
