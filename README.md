# Disneyland Reviews Analysis and Conclusion Generation

Bu proje, Disneyland yorumlarını analiz eden ve sonuçlar oluşturan bir makine öğrenimi ve doğal dil işleme (NLP) tabanlı bir sistemdir. Sistem, kullanıcı yorumlarını özetler, sınıflandırır ve kümeleme ile gruplar. Ayrıca, bir Flask API aracılığıyla sonuçları sunar.

---

## Özellikler

- **Metin Ön İşleme**: Yorumlardaki gereksiz karakterler, sayılar ve durak kelimeler temizlenir.
- **Sınıflandırma**: Yorumların puanlarını tahmin etmek için bir Random Forest sınıflandırıcı kullanır.
- **Kümeleme**: Yorumları içeriklerine göre gruplamak için MiniBatchKMeans algoritması.
- **Özetleme**: Kullanıcı yorumlarını özetlemek için BART tabanlı bir özetleme modeli.
- **API**: Flask tabanlı bir API ile sistem erişilebilir kılınır.


