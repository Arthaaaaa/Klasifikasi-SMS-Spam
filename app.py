import os
import random

from flask import Flask, request
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score


app = Flask(__name__)

# =========================================================
# 1) LOAD DATA (Rubrik Poin 1: Load Data)
# =========================================================
# encoding latin-1 sering diperlukan untuk dataset SMS spam
data = pd.read_csv("spam.csv", encoding="latin-1")

# Ambil kolom yang dibutuhkan saja (v1 label, v2 text)
data = data[["v1", "v2"]].copy()
data.columns = ["label", "text"]

# =========================================================
# 2) ENCODING LABEL (Rubrik Poin 1: Encoding wajib)
#    Dosen minta LabelEncoder -> kita pakai LabelEncoder
# =========================================================
le = LabelEncoder()
# ham/spam -> angka (biasanya ham=0, spam=1; bisa kebalik tergantung alfabet)
data["label_encoded"] = le.fit_transform(data["label"])

# Biar jelas, kita catat mapping-nya
# le.classes_ contoh: ['ham', 'spam']
label_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}

# =========================================================
# 3) SPLIT DATA 80/20 (Rubrik Poin 1: Split Data)
# =========================================================
X_text = data["text"]
y = data["label_encoded"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 4) FEATURE EXTRACTION (CountVectorizer)
#    Untuk teks: kata -> angka (frekuensi kata)
#    Catatan: Scaling TIDAK dipakai karena ini MultinomialNB (teks)
# =========================================================
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train_text)   # fit hanya di train (anti data leakage)
X_test = vectorizer.transform(X_test_text)

# =========================================================
# 5) TRAIN MODEL NAIVE BAYES (Rubrik Poin 2)
# =========================================================
model = MultinomialNB()
model.fit(X_train, y_train)

# =========================================================
# 6) EVALUASI (Rubrik Poin 3: Confusion Matrix + Accuracy)
# =========================================================
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# =========================================================
# 7) ANALISIS PROBABILITAS (Rubrik Poin 3: predict_proba)
#    Ambil 1 sampel yang salah prediksi kalau ada, kalau tidak ambil random.
# =========================================================
mis_idx = [i for i, (a, b) in enumerate(zip(y_test, y_pred)) if a != b]

if len(mis_idx) > 0:
    chosen_i = mis_idx[0]
    chosen_type = "SAMPel SALAH PREDIKSI (dari test set)"
else:
    chosen_i = random.randint(0, len(y_test) - 1)
    chosen_type = "SAMPel ACAK (tidak ada yang salah prediksi di test set)"

sample_text = X_test_text.iloc[chosen_i]
sample_true = int(y_test.iloc[chosen_i])
sample_pred = int(y_pred[chosen_i])

sample_vec = vectorizer.transform([sample_text])
proba = model.predict_proba(sample_vec)[0]  # [p(class0), p(class1)]

# Karena label encoding bisa ham=0 spam=1 atau kebalik, kita tampilkan label aslinya:
# Ambil nama kelas untuk index 0 dan 1:
class0_label = le.inverse_transform([0])[0]
class1_label = le.inverse_transform([1])[0]

# =========================================================
# 8) SIMULASI DATA BARU (Rubrik Poin 4: 2 dummy kontras)
#    Dummy A: spammy banget
#    Dummy B: chat normal
# =========================================================
dummy_A = "CONGRATULATIONS! You WIN FREE cash prize, claim now, urgent, call now!"
dummy_B = "Hai, nanti pulang sekolah jadi makan bakso bareng ya?"

dummy_A_vec = vectorizer.transform([dummy_A])
dummy_B_vec = vectorizer.transform([dummy_B])

dummy_A_pred = int(model.predict(dummy_A_vec)[0])
dummy_B_pred = int(model.predict(dummy_B_vec)[0])

dummy_A_proba = model.predict_proba(dummy_A_vec)[0]
dummy_B_proba = model.predict_proba(dummy_B_vec)[0]

# =========================================================
# 9) BUAT LAPORAN STRING (biar gampang tampil & copy ke laporan)
# =========================================================
def decode_label(encoded_int: int) -> str:
    return le.inverse_transform([encoded_int])[0]

REPORT_TEXT = f"""
=== LAPORAN EVALUASI (TEST SET 20%) ===
Label mapping (LabelEncoder): {label_mapping}

Confusion Matrix (baris=actual, kolom=pred):
{cm}

Accuracy: {acc:.4f}

=== ANALISIS PROBABILITAS ===
Jenis sampel: {chosen_type}
Teks sampel: {sample_text}

Actual (encoded): {sample_true} -> {decode_label(sample_true)}
Pred   (encoded): {sample_pred} -> {decode_label(sample_pred)}

predict_proba:
P({class0_label}) = {proba[0]*100:.2f}%
P({class1_label}) = {proba[1]*100:.2f}%

Catatan interpretasi:
- Kalau mendekati 50%-50% berarti model ragu-ragu.
- Kalau mendekati 90%-99% berarti model yakin.

=== SIMULASI 2 DATA DUMMY KONTRAS ===
Dummy A (spammy): {dummy_A}
Prediksi: {decode_label(dummy_A_pred)}
Proba -> P({class0_label})={dummy_A_proba[0]*100:.2f}%, P({class1_label})={dummy_A_proba[1]*100:.2f}%

Dummy B (normal): {dummy_B}
Prediksi: {decode_label(dummy_B_pred)}
Proba -> P({class0_label})={dummy_B_proba[0]*100:.2f}%, P({class1_label})={dummy_B_proba[1]*100:.2f}%

Kesimpulan fitur yang kuat (versi teks):
- Kata-kata seperti "FREE", "WIN", "PRIZE", "URGENT", "CALL NOW" biasanya mendorong prediksi ke spam.
- Kata percakapan normal cenderung mendorong prediksi ke ham.
"""

print(REPORT_TEXT)


# =========================================================
# 10) ROUTE WEB
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    proba_text = ""
    user_message = ""

    if request.method == "POST":
        user_message = request.form["message"]

        msg_vec = vectorizer.transform([user_message])
        pred = int(model.predict(msg_vec)[0])
        pred_label = decode_label(pred)

        prob = model.predict_proba(msg_vec)[0]
        proba_text = f"P({class0_label})={prob[0]*100:.2f}% | P({class1_label})={prob[1]*100:.2f}%"

        if pred_label == "spam":
            result = "🚨 SPAM"
        else:
            result = "✅ HAM (Bukan Spam)"

    return f"""
    <!DOCTYPE html>
    <html lang="id">
      <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TA-12 Klasifikasi SMS Spam (Naive Bayes)</title>
        <style>
          * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
          }}
          
          body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
          }}
          
          .container {{
            max-width: 1200px;
            margin: 0 auto;
          }}
          
          .header {{
            text-align: center;
            color: white;
            margin-bottom: 40px;
            padding: 30px 20px;
          }}
          
          .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
          }}
          
          .header p {{
            font-size: 1.1rem;
            opacity: 0.9;
          }}
          
          .card {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
          }}
          
          .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
          }}
          
          .card h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8rem;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
          }}
          
          .card h3 {{
            color: #764ba2;
            margin-bottom: 15px;
            font-size: 1.4rem;
          }}
          
          .form-group {{
            margin-bottom: 20px;
          }}
          
          textarea {{
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s ease;
          }}
          
          textarea:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
          }}
          
          .btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
          }}
          
          .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
          }}
          
          .btn:active {{
            transform: translateY(0);
          }}
          
          .result-box {{
            margin-top: 25px;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            animation: fadeIn 0.5s ease;
          }}
          
          .result-spam {{
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            color: white;
          }}
          
          .result-ham {{
            background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
            color: white;
          }}
          
          .prob-text {{
            margin-top: 15px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 1rem;
            color: #495057;
          }}
          
          .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
          }}
          
          .metric-card {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border-left: 4px solid #667eea;
          }}
          
          .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
          }}
          
          .metric-label {{
            color: #6c757d;
            font-size: 0.9rem;
          }}
          
          .confusion-matrix {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
            overflow-x: auto;
          }}
          
          .confusion-matrix pre {{
            font-family: 'Courier New', monospace;
            font-size: 1rem;
            color: #495057;
            margin: 0;
          }}
          
          .info-box {{
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            border-radius: 5px;
            margin-top: 15px;
          }}
          
          .sample-text {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #764ba2;
            margin: 10px 0;
            white-space: pre-wrap;
            word-wrap: break-word;
          }}
          
          .dummy-example {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
          }}
          
          .dummy-example h4 {{
            color: #667eea;
            margin-bottom: 10px;
          }}
          
          .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin: 0 5px;
          }}
          
          .badge-spam {{
            background: #ff6b6b;
            color: white;
          }}
          
          .badge-ham {{
            background: #51cf66;
            color: white;
          }}
          
          .badge-info {{
            background: #667eea;
            color: white;
          }}
          
          @keyframes fadeIn {{
            from {{
              opacity: 0;
              transform: translateY(-10px);
            }}
            to {{
              opacity: 1;
              transform: translateY(0);
            }}
          }}
          
          .section-divider {{
            height: 2px;
            background: linear-gradient(90deg, transparent, #667eea, transparent);
            margin: 30px 0;
          }}
          
          @media (max-width: 768px) {{
            .header h1 {{
              font-size: 1.8rem;
            }}
            
            .card {{
              padding: 20px;
            }}
            
            .metrics-grid {{
              grid-template-columns: 1fr;
            }}
          }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="header">
            <h1>🛡️ Klasifikasi SMS Spam</h1>
            <p>Multinomial Naive Bayes Classifier</p>
          </div>

          <div class="card">
            <h2>📝 Uji Pesan Baru</h2>
            <form method="post">
              <div class="form-group">
                <textarea name="message" rows="5" placeholder="Masukkan teks SMS di sini..." required>{user_message}</textarea>
              </div>
              <button type="submit" class="btn">🔍 Cek Pesan</button>
            </form>
            
            {f'<div class="result-box {"result-spam" if "SPAM" in result else "result-ham"}">{result}</div>' if result else ''}
            {f'<div class="prob-text"><strong>📊 Probabilitas:</strong> {proba_text}</div>' if proba_text else ''}
          </div>

          <div class="card">
            <h2>📈 Evaluasi Model</h2>
            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-value">{acc:.2%}</div>
                <div class="metric-label">Accuracy (Test Set 20%)</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{cm[0][0] + cm[1][1]}</div>
                <div class="metric-label">True Predictions</div>
              </div>
              <div class="metric-card">
                <div class="metric-value">{cm[0][1] + cm[1][0]}</div>
                <div class="metric-label">False Predictions</div>
              </div>
            </div>
            
            <h3 style="margin-top: 25px;">Confusion Matrix</h3>
            <p style="color: #6c757d; margin-bottom: 10px;">Baris = Actual, Kolom = Predicted</p>
            <div class="confusion-matrix">
              <pre>{cm}</pre>
            </div>
            
            <div class="info-box">
              <strong>💡 Catatan:</strong> Karena dataset ini berupa teks, kita menggunakan <strong>MultinomialNB</strong>.
              Feature scaling (StandardScaler) <strong>tidak diperlukan</strong> dan biasanya dipakai untuk <strong>GaussianNB</strong> pada data numerik.
            </div>
            
            <div style="margin-top: 20px;">
              <strong>🏷️ Label Mapping (LabelEncoder):</strong>
              <div style="margin-top: 10px;">
                {', '.join([f'<span class="badge badge-info">{k} = {v}</span>' for k, v in label_mapping.items()])}
              </div>
            </div>
          </div>

          <div class="card">
            <h2>🔬 Analisis Probabilitas</h2>
            <p><strong>Jenis Sampel:</strong> <span class="badge badge-info">{chosen_type}</span></p>
            
            <h3 style="margin-top: 20px;">Teks Sampel</h3>
            <div class="sample-text">{sample_text}</div>
            
            <div style="margin-top: 20px; display: flex; gap: 15px; flex-wrap: wrap;">
              <div>
                <strong>Actual:</strong> 
                <span class="badge {'badge-spam' if decode_label(sample_true) == 'spam' else 'badge-ham'}">{decode_label(sample_true)}</span>
              </div>
              <div>
                <strong>Predicted:</strong> 
                <span class="badge {'badge-spam' if decode_label(sample_pred) == 'spam' else 'badge-ham'}">{decode_label(sample_pred)}</span>
              </div>
            </div>
            
            <div class="prob-text" style="margin-top: 15px;">
              <strong>📊 Predict Probabilities:</strong><br>
              P({class0_label}) = <strong>{proba[0]*100:.2f}%</strong> | 
              P({class1_label}) = <strong>{proba[1]*100:.2f}%</strong>
            </div>
          </div>

          <div class="card">
            <h2>🎯 Simulasi Data Dummy Kontras</h2>
            
            <div class="dummy-example">
              <h4>📨 Dummy A (Spammy)</h4>
              <p style="margin-bottom: 10px; font-style: italic;">"{dummy_A}"</p>
              <div style="margin-top: 10px;">
                <strong>Prediksi:</strong> 
                <span class="badge {'badge-spam' if decode_label(dummy_A_pred) == 'spam' else 'badge-ham'}">{decode_label(dummy_A_pred)}</span>
                <br>
                <strong>Probabilitas:</strong> P({class0_label}) = {dummy_A_proba[0]*100:.2f}% | P({class1_label}) = {dummy_A_proba[1]*100:.2f}%
              </div>
            </div>
            
            <div class="dummy-example">
              <h4>💬 Dummy B (Normal)</h4>
              <p style="margin-bottom: 10px; font-style: italic;">"{dummy_B}"</p>
              <div style="margin-top: 10px;">
                <strong>Prediksi:</strong> 
                <span class="badge {'badge-spam' if decode_label(dummy_B_pred) == 'spam' else 'badge-ham'}">{decode_label(dummy_B_pred)}</span>
                <br>
                <strong>Probabilitas:</strong> P({class0_label}) = {dummy_B_proba[0]*100:.2f}% | P({class1_label}) = {dummy_B_proba[1]*100:.2f}%
              </div>
            </div>
          </div>
        </div>
      </body>
    </html>
    """


# =========================================================
# 11) RUN (Railway-friendly)
# =========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
