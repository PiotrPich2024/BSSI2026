# Uzasadnienie parametrów

---

## Batch size

Wybraliśmy 128 aby efektywnie  wykorzystać GPU

## Learning rate = 0.001

standardowa wartość dla optymalizatora Adam.

## Adam optimizer

dobrze radzi sobie z problemami wizji komputerowej i szybko konwerguje.

## 10 epok

wystarczające do uzyskania stabilnych wyników na CIFAR-10.

## Dropout = 0.5

redukuje overfitting poprzez losowe wyłączanie neuronów.

## 2 warstwy konwolucyjne

wystarczające do ekstrakcji podstawowych cech obrazów CIFAR-10

## WNIOSKI

Model osiągnął dokładność 81.98% na zbiorze treningowym oraz 73.83% na zbiorze testowym.
Różnica między wynikami wynosi około 8%, co wskazuje na niewielki poziom overfittingu.
Otrzymany wynik jest typowy dla prostych sieci konwolucyjnych trenowanych na zbiorze CIFAR-10.