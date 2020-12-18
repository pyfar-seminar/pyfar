# TODOs: Fractional Octave Smoothing Modul

## Grundstruktur für FractionalOctaveSmoothing Objekt:
* [ ] Methoden definieren (init, calc_integration_limits, calc_weights, apply, 
    smooth, ...)
* [ ] Attribute definieren (Fensterbreite, Abtastfrequenz, Frequenzen an der 
    geglättet wird, Integrationsgrenzen(?), Gewichte)
* [ ] Get-& Set-Methoden definieren
* [ ] Datentypen für Gewichte und Integrationsgrenzen festlegen
* [ ] 

## Signalverarbeitung
* [ ] Paddingmethode für Smoothing am Rand des Spektrums überlegen
* [ ] Festlegen, wann Integrationsgrenzen und Gewichte erstmalig berechnet 
    werden, wann neue Berechnung nötig ist


## Testing:
* [ ] Error Codes testen
* [ ] Berechnung der Integrationsgrenzen mit analytischer Lösung vergleichen
* [ ] Berechnung der Gewichte mt analytischer Lösung vergeleichen (Dirac?)
* [ ] Get- & Set-Methoden testen
* [ ] Verschiedene Konstruktormethoden testen 
* [ ] Smoothing eines mehrkanaliges Eingangssignal testen
* [ ] Alle Methoden getestet?

## Doku:
* [ ] Objekt dokumentieren
* [ ] Methoden dokumentieren
* [ ] Jupyter Notebook Tutorial schreiben
