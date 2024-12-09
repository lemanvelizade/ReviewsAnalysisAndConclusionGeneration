import re
import nltk
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# NLTK indirmeleri
nltk.download('stopwords')
nltk.download('punkt')

# Flask uygulaması
app = Flask(__name__)

# Yorum ön işleme fonksiyonu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    text = " ".join(word for word in tokens if word not in stop_words)
    return text

# Veri Seti Yükleme ve İşleme
import kagglehub
# Download latest version
path = kagglehub.dataset_download("arushchillar/disneyland-reviews")
data = pd.read_csv(path, encoding='latin1')

data = data[["Review_Text", "Rating"]].dropna()
data["Cleaned_Review"] = data["Review_Text"].apply(preprocess_text)

# TF-IDF ile Özellik Çıkarımı
tfidf_vectorizer = TfidfVectorizer(max_features=500)  # Daha küçük özellik sayısı
X = tfidf_vectorizer.fit_transform(data["Cleaned_Review"]).toarray().astype('float64')  # float64 türüne dönüştürülmesi
y = data["Rating"]

# Modelleri Kaydetme
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")

# Sınıflandırıcı Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
joblib.dump(classifier, "classifier_model.pkl")

# Sınıflandırma Performansı
y_pred = classifier.predict(X_test)
print("Sınıflandırma Performansı:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# Kümeleme (MiniBatchKMeans kullanıldı)
# Veri kümesinin büyük boyutlarda olması nedeniyle MiniBatchKMeans tercih edildi
data_sample = data.sample(frac=0.1, random_state=42)  # Örnekleme (Veri kümesinin %10'u)
X_sample = tfidf_vectorizer.transform(data_sample["Cleaned_Review"]).toarray().astype('float64')  # float64 türüne dönüştürülmesi
kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_sample)
data_sample["Cluster"] = clusters
joblib.dump(kmeans, "kmeans_model.pkl")

# Özetleme (LLM ile)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# API: Yorum İşleme
@app.route("/process_review", methods=["POST"])
def process_review():
    try:
        data = request.json
        review = data.get("review", "")

        if not review:
            raise ValueError("Review text is missing")

        # Yorum ön işleme
        cleaned_review = preprocess_text(review)
        vectorized_review = tfidf_vectorizer.transform([cleaned_review])

        # Kümeleme tahmini
        cluster = int(kmeans.predict(vectorized_review)[0])  # Convert to native Python int

        # Determine max_length based on review length
        max_length = min(len(review.split()), 50)  # max_length should not exceed the length of the review
        min_length = max(10, max_length - 1)  # Ensure min_length is less than max_length

        # Özetleme
        summary = summarizer(review, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

        # Sonucu döndür
        return jsonify({"cluster": cluster, "summary": summary})

    except Exception as e:
        app.logger.error(f"Error processing review: {str(e)}")  # Log the error to Flask logs
        return jsonify({"error": f"Error processing review: {str(e)}"}), 500



# Flask API çalıştırma
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False, threaded=True)
