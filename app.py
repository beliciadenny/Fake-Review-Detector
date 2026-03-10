import streamlit as st
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack, csr_matrix
import pandas as pd

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# ─────────────────────────────────────────
# Styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
        width: 100%;
        border: none;
    }
    .stButton>button:hover { background-color: #34495e; }
    .result-fake {
        background-color: #fdecea;
        border-left: 6px solid #e74c3c;
        padding: 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        color: #c0392b;
    }
    .result-real {
        background-color: #eafaf1;
        border-left: 6px solid #2ecc71;
        padding: 16px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: bold;
        color: #1e8449;
    }
    .signal-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────
STOPWORDS = set([
    'i','me','my','we','our','you','your','he','she','it','they','them',
    'this','that','these','those','is','are','was','were','be','been',
    'being','have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','a','an','the','and','but','or',
    'for','with','at','by','from','in','of','on','to','up','as','if','so',
    'about','into','through','during','before','after','than','then','also',
    'not','no','nor','very','just','here','there'
])

SUPERLATIVES = {
    'best','worst','greatest','perfect','amazing','terrible','incredible',
    'awful','fantastic','horrible','exceptional','outstanding'
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in text.split() if w not in STOPWORDS)

def repetition_score(text):
    words = text.lower().split()
    if not words:
        return 0
    return round(1 - len(set(words)) / len(words), 4)

def superlative_density(text):
    words = text.lower().split()
    if not words:
        return 0
    return sum(1 for w in words if w in SUPERLATIVES) / len(words)

def caps_word_ratio(text):
    words = text.split()
    if not words:
        return 0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


# ─────────────────────────────────────────
# Train Model (cached so it runs only once)
# ─────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    rng = np.random.default_rng(42)

    fake_phrases = [
        "Absolutely amazing experience!", "Best place EVER!!! Highly recommend!!!",
        "Five stars, perfect in every way.", "DO NOT GO HERE. Worst service ever.",
        "I love this place so much!!!", "Outstanding!!! Will definitely return!!!",
        "This is the greatest restaurant in the city!", "Terrible terrible terrible.",
        "Perfect perfect perfect! 10/10!", "Never coming back. Horrible staff.",
        "Best food I have ever tasted in my entire life!",
        "Management is incredible and super friendly!",
        "AVOID AT ALL COSTS. They ruined my evening.",
        "Exceptional service, exceptional food, exceptional ambiance!",
    ]
    real_phrases = [
        "The pasta was decent, though a bit overpriced for the portion size.",
        "Service was slow on a Friday night, but the staff were friendly.",
        "I've been coming here for three years. Quality has slightly declined lately.",
        "Good spot for a quick lunch. Parking can be tricky on weekdays.",
        "The ambiance is nice but the menu hasn't changed in two years.",
        "Had the salmon — it was cooked well but the sauce was too salty.",
        "Decent coffee, nothing special. Would choose it over a chain nearby.",
        "Mixed experience. The appetizers were great but the main course disappointed.",
        "Came for a birthday dinner. Staff acknowledged it which was a nice touch.",
        "It's okay. Not my first choice but fine if you're in the area.",
        "The burger was juicy and fresh. Fries were a bit soggy though.",
        "Prices went up but the quality didn't follow. Disappointing.",
        "Nice place for a date. A bit loud when full but manageable.",
        "They got my order wrong but fixed it quickly without hassle.",
    ]

    records = []
    for _ in range(3000):
        is_fake = rng.random() < 0.40
        if is_fake:
            base = rng.choice(fake_phrases)
            text = " ".join([base] * int(rng.integers(1, 4)))
            rating = int(rng.choice([1, 5], p=[0.3, 0.7]))
            reviewer_count = int(rng.integers(1, 5))
            days_since = int(rng.integers(0, 3))
        else:
            base = rng.choice(real_phrases)
            extras = rng.choice(real_phrases, size=int(rng.integers(0, 3)), replace=False)
            text = " ".join([base] + list(extras))
            rating = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.25, 0.35, 0.25]))
            reviewer_count = int(rng.integers(5, 200))
            days_since = int(rng.integers(3, 365))

        records.append({
            'review_text': text, 'rating': rating,
            'reviewer_review_count': reviewer_count,
            'days_since_last_review': days_since,
            'exclamation_count': text.count('!'),
            'caps_ratio': round(sum(1 for c in text if c.isupper()) / max(len(text), 1), 4),
            'review_length': len(text.split()),
            'extreme_rating': 1 if rating in [1, 5] else 0,
            'new_reviewer': 1 if reviewer_count < 5 else 0,
            'burst_posting': 1 if days_since <= 1 else 0,
            'repetition_score': repetition_score(text),
            'superlative_density': superlative_density(text),
            'caps_word_ratio': caps_word_ratio(text),
            'label': int(is_fake)
        })

    df = pd.DataFrame(records)
    behavior_features = [
        'rating', 'reviewer_review_count', 'days_since_last_review',
        'exclamation_count', 'caps_ratio', 'review_length',
        'extreme_rating', 'new_reviewer', 'burst_posting',
        'repetition_score', 'superlative_density', 'caps_word_ratio'
    ]

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    df['cleaned'] = df['review_text'].apply(clean_text)
    X_text = tfidf.fit_transform(df['cleaned'])
    X_beh = csr_matrix(df[behavior_features].values.astype(float))
    X = hstack([X_text, X_beh])
    y = df['label'].values

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, tfidf, behavior_features


model, tfidf, behavior_features = train_model()


# ─────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────
def predict(review_text, rating, reviewer_count, days_since):
    cleaned = clean_text(review_text)
    X_text = tfidf.transform([cleaned])

    excl = review_text.count('!')
    caps_r = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
    length = len(review_text.split())
    extreme_r = 1 if rating in [1, 5] else 0
    new_rev = 1 if reviewer_count < 5 else 0
    burst = 1 if days_since <= 1 else 0
    rep = repetition_score(review_text)
    sup = superlative_density(review_text)
    cap_w = caps_word_ratio(review_text)

    beh = [rating, reviewer_count, days_since, excl, caps_r,
           length, extreme_r, new_rev, burst, rep, sup, cap_w]
    X_beh = csr_matrix([beh])
    X = hstack([X_text, X_beh])

    prob = model.predict_proba(X)[0][1]
    return prob, {
        'Star Rating': f"{rating} ⭐ {'(Extreme ⚠️)' if extreme_r else '(Normal ✅)'}",
        'Reviewer Total Reviews': f"{reviewer_count} {'(New Account ⚠️)' if new_rev else '(Established ✅)'}",
        'Days Since Last Review': f"{days_since} {'(Burst Posting ⚠️)' if burst else '(Normal ✅)'}",
        'Exclamation Marks': f"{excl} {'⚠️' if excl > 3 else '✅'}",
        'Repetition Score': f"{rep:.2f} {'(High ⚠️)' if rep > 0.4 else '(Normal ✅)'}",
        'Superlative Density': f"{sup:.2f} {'(High ⚠️)' if sup > 0.1 else '(Normal ✅)'}",
    }


# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
st.title("🕵️ Fake Review Detector")
st.markdown("Detects fake business reviews using **NLP + behavioral signal analysis**.")
st.markdown("---")

# Input
review = st.text_area("📝 Paste a review here:", height=120,
                       placeholder="e.g. Best place EVER!!! Amazing amazing amazing!!!")

col1, col2, col3 = st.columns(3)
with col1:
    rating = st.selectbox("⭐ Star Rating", [1, 2, 3, 4, 5], index=4)
with col2:
    reviewer_count = st.slider("👤 Reviewer's Total Reviews", 1, 200, 10)
with col3:
    days_since = st.slider("📅 Days Since Last Review", 0, 365, 30)

st.markdown("")
predict_btn = st.button("🔍 Detect Review")

# Output
if predict_btn:
    if not review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        prob, signals = predict(review, rating, reviewer_count, days_since)
        is_fake = prob > 0.5

        st.markdown("---")
        st.markdown("### 🎯 Result")

        if is_fake:
            st.markdown(f'<div class="result-fake">🚨 FAKE REVIEW &nbsp;|&nbsp; Confidence: {prob:.1%}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-real">✅ REAL REVIEW &nbsp;|&nbsp; Confidence: {1-prob:.1%}</div>',
                        unsafe_allow_html=True)

        # Confidence bar
        st.markdown("")
        st.markdown("**Fake Probability Score:**")
        st.progress(float(prob))
        st.caption(f"{prob:.1%} chance this review is fake")

        # Signals
        st.markdown("---")
        st.markdown("### 📊 Signal Breakdown")
        for signal, value in signals.items():
            st.markdown(f'<div class="signal-box"><b>{signal}:</b> {value}</div>',
                        unsafe_allow_html=True)

        # Try another
        st.markdown("---")
        st.info("💡 Try changing the inputs above and click Detect again to see how signals affect the result.")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with Python · Scikit-learn · Streamlit &nbsp;|&nbsp; "
    "Fake Review Detector Project</small></center>",
    unsafe_allow_html=True
)
