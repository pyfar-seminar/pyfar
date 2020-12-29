# TODOs: Fractional Octave Smoothing Modul

## Grundstruktur für FractionalOctaveSmoothing Objekt:
* [x] Methoden definieren (init, calc_integration_limits, calc_weights, apply, 
    smooth, ...)
* [x] Attribute definieren (Fensterbreite, Abtastfrequenz, Frequenzen an der geglättet wird, Integrationsgrenzen(?), Gewichte)
* [x] Datentypen für Gewichte und Integrationsgrenzen festlegen
* [x] Methoden zum Smoothing anhand Signal und HRTF erstellen
* [ ] Get-& Set-Methoden definieren

## Signalverarbeitung
* [x] Paddingmethode für Smoothing am Rand des Spektrums überlegen
* [ ] Festlegen, wann Integrationsgrenzen und Gewichte erstmalig berechnet 
    werden, wann neue Berechnung nötig ist


## Testing:
* [x] Ctors und Error Exceptions testen
* [x] Berechnung der Integrationsgrenzen 
* [ ] Berechnung der Gewichte mit analytischer Lösung vergeleichen (Dirac?)
* [ ] Get- & Set-Methoden testen
* [ ] Smoothing eines mehrkanaliges Eingangssignal testen
* [ ] Smoothing einer mehrkanaliger HRTF testen
* [ ] Alle Methoden getestet?

## Doku:
* [ ] Objekt dokumentieren
* [ ] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben
