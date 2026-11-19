import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# --- Load Models ---
@st.cache_resource 
def load_models():
    news_model = joblib.load('fake_news_model.pkl')
    news_vec = joblib.load('tfidf_vectorizer.pkl')
    ins_model = joblib.load('insurance_rf_model.pkl')
    ins_scaler = joblib.load('insurance_scaler.pkl')
    return news_model, news_vec, ins_model, ins_scaler

news_model, news_vec, ins_model, ins_scaler = load_models()

# --- Streamlit UI ---
st.title("ML Portfolio: Classification & Regression")
st.write("This app demonstrates two models: Fake News Detection and Insurance Cost Prediction.")

# create two tabs
tab1, tab2 = st.tabs(["📰 Fake News Detection", "🏥 Insurance Cost Prediction"])

# tab 1: classification
with tab1:
    st.header("Fake News Detection")
    st.write("Enter a news article below and the NLP model will classify it.")
    
    text_input = st.text_area("Paste article text here:", height=150)
    
    if st.button("Classify Article"):
        if text_input:
            cleaned = clean_text(text_input)
            vectorized = news_vec.transform([cleaned])
            prediction = news_model.predict(vectorized)[0]
            probability = news_model.predict_proba(vectorized)[0]
            
            if prediction == 1:
                st.success(f"📰 **REAL NEWS** (Confidence: {probability[1]*100:.2f}%)")
            else:
                st.error(f"🚨 **FAKE NEWS** (Confidence: {probability[0]*100:.2f}%)")
        else:
            st.warning("Please enter some text.")

# tab 2: regression
with tab2:
    st.header("Medical Insurance Cost Prediction")
    st.write("Enter customer details below to predict their estimated medical charges.")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 15.0, 55.0, 25.0)
        children = st.number_input("Number of Children", 0, 10, 0)
        sex = st.selectbox("Sex", ["male", "female"])
    with col2:
        smoker = st.selectbox("Smoker", ["no", "yes"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
    if st.button("Predict Charges"):
        sex_encoded = 1 if sex == "male" else 0
        smoker_encoded = 1 if smoker == "yes" else 0
        region_northwest = 1 if region == "northwest" else 0
        region_southeast = 1 if region == "southeast" else 0
        region_southwest = 1 if region == "southwest" else 0
        
        if bmi < 18.5: bmi_cat = 0
        elif bmi < 25: bmi_cat = 1
        elif bmi < 30: bmi_cat = 2
        else: bmi_cat = 3

        input_data = pd.DataFrame([{
            'age': age, 'sex': sex_encoded, 'bmi': bmi, 'children': children,
            'smoker': smoker_encoded, 'region_northwest': region_northwest,
            'region_southeast': region_southeast, 'region_southwest': region_southwest,
            'bmi_category': bmi_cat
        }])

        scaled_data = ins_scaler.transform(input_data)
        prediction = ins_model.predict(scaled_data)[0]

        st.success(f"💰 Predicted Insurance Charge: **${prediction:,.2f}**")