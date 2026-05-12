# Drzewa decyzyjne dla danych z brakującymi wartościami

## Cel projektu
Celem projektu była implementacja drzewa decyzyjnego inspirowanego ID3/CART
oraz analiza metod radzenia sobie z brakującymi danymi:
- metoda domyślna
- metoda trywialna
- surrogate splits
- imputacja kNN

## Implementacja
Drzewo:
- entropia + information gain
- podziały binarne
- obsługa atrybutów ciągłych i dyskretnych

Imputacja:
- autorska implementacja kNN
- dystans mieszany

## Metodologia eksperymentów
Datasety:
- Titanic
- Adult
- Car Sales

Procedura:
- podział 80/20
- 25 uruchomień
- metryki: accuracy, F1, confusion matrix

## Wnioski
Metody odporne na braki danych poprawiają stabilność klasyfikacji.
Najlepsze wyniki uzyskano dla imputacji kNN.