import streamlit as st
import pandas as pd
import joblib

# =========================================================
# LOAD MODEL & SCALER
# =========================================================

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main Background */
.main {
    background-color: #f8fafc;
}

/* Remove dimming effect */
div[data-baseweb="checkbox"] {
    opacity: 1 !important;
}

div[data-baseweb="radio"] {
    opacity: 1 !important;
}

/* Main Title */
h1 {
    text-align: center;
    color: #0f172a;
    font-size: 50px;
    font-weight: bold;
}

/* Section headings */
h3 {
    color: #1e293b;
}

/* Button Styling */
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3.3em;
    width: 100%;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

/* Info Box */
.info-box {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #2563eb;
    margin-bottom: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("🩺 Diabetes Prediction System")

st.markdown("""
<div class='info-box'>

This AI-powered healthcare application predicts whether a person is likely to be:

<ul>
<li><b>Diabetic</b></li>
<li><b>Non-Diabetic</b></li>
</ul>

based on medical and lifestyle indicators.

</div>
""", unsafe_allow_html=True)

st.write("---")

# =========================================================
# CHECKBOX SECTION
# =========================================================

col1, col2 = st.columns(2)

# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    st.subheader("🧬 Medical Conditions")

    HighBP = st.checkbox(
        "High Blood Pressure"
    )

    HighChol = st.checkbox(
        "High Cholesterol"
    )

    CholCheck = st.checkbox(
        "Cholesterol Check Done in Last 5 Years"
    )

    Smoker = st.checkbox(
        "Smoker"
    )

    Stroke = st.checkbox(
        "History of Stroke"
    )

    HeartDiseaseorAttack = st.checkbox(
        "Heart Disease / Heart Attack"
    )

    DiffWalk = st.checkbox(
        "Difficulty Walking"
    )

    AnyHealthcare = st.checkbox(
        "Has Healthcare Coverage"
    )

    NoDocbcCost = st.checkbox(
        "Couldn't Visit Doctor Due To Cost"
    )

# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    st.subheader("🏃 Lifestyle Information")

    PhysActivity = st.checkbox(
        "Physical Activity"
    )

    Fruits = st.checkbox(
        "Consumes Fruits Regularly"
    )

    Veggies = st.checkbox(
        "Consumes Vegetables Regularly"
    )

    HvyAlcoholConsump = st.checkbox(
        "Heavy Alcohol Consumption"
    )

# =========================================================
# SLIDERS
# =========================================================

st.write("---")

st.subheader("📊 Health Metrics")

col3, col4 = st.columns(2)

# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    BMI = st.slider(
        "BMI (Body Mass Index)",
        10.0,
        100.0,
        25.0,
        help="""
        BMI Categories:

        Below 18.5 → Underweight

        18.5 - 24.9 → Normal Weight

        25 - 29.9 → Overweight

        30 and above → Obese
        """
    )

    MentHlth = st.slider(
        "Poor Mental Health Days",
        0,
        30,
        0,
        help="""
        Number of days in the last 30 days
        during which mental health was not good.
        """
    )

    PhysHlth = st.slider(
        "Poor Physical Health Days",
        0,
        30,
        0,
        help="""
        Number of days in the last 30 days
        during which physical health was not good.
        """
    )

    GenHlth = st.slider(
        "General Health",
        1,
        5,
        3,
        help="""
        General Health Rating:

        1 → Excellent

        2 → Very Good

        3 → Good

        4 → Fair

        5 → Poor
        """
    )

# ---------------------------------------------------------
# COLUMN 4
# ---------------------------------------------------------

with col4:

    Age = st.slider(
        "Age Category",
        1,
        13,
        5,
        help="""
        Age Categories:

        1  → 18-24 years

        2  → 25-29 years

        3  → 30-34 years

        4  → 35-39 years

        5  → 40-44 years

        6  → 45-49 years

        7  → 50-54 years

        8  → 55-59 years

        9  → 60-64 years

        10 → 65-69 years

        11 → 70-74 years

        12 → 75-79 years

        13 → 80+ years
        """
    )

    Education = st.slider(
        "Education Level",
        1,
        6,
        3,
        help="""
        Education Levels:

        1 → Never attended school

        2 → Elementary School

        3 → Some High School

        4 → High School Graduate

        5 → Some College / Technical School

        6 → College Graduate
        """
    )

    Income = st.slider(
        "Income Level",
        1,
        8,
        4,
        help="""
        Income Levels:

        1 → Less than $10,000

        2 → $10,000 - $15,000

        3 → $15,000 - $20,000

        4 → $20,000 - $25,000

        5 → $25,000 - $35,000

        6 → $35,000 - $50,000

        7 → $50,000 - $75,000

        8 → More than $75,000
        """
    )

    Sex = st.radio(
        "Gender",
        ["Female", "Male"]
    )

# =========================================================
# CONVERT VALUES
# =========================================================

Sex = 1 if Sex == "Male" else 0

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_df = pd.DataFrame({

    'HighBP': [int(HighBP)],
    'HighChol': [int(HighChol)],
    'CholCheck': [int(CholCheck)],
    'BMI': [BMI],
    'Smoker': [int(Smoker)],
    'Stroke': [int(Stroke)],
    'HeartDiseaseorAttack': [int(HeartDiseaseorAttack)],
    'PhysActivity': [int(PhysActivity)],
    'Fruits': [int(Fruits)],
    'Veggies': [int(Veggies)],
    'HvyAlcoholConsump': [int(HvyAlcoholConsump)],
    'AnyHealthcare': [int(AnyHealthcare)],
    'NoDocbcCost': [int(NoDocbcCost)],
    'GenHlth': [GenHlth],
    'MentHlth': [MentHlth],
    'PhysHlth': [PhysHlth],
    'DiffWalk': [int(DiffWalk)],
    'Sex': [Sex],
    'Age': [Age],
    'Education': [Education],
    'Income': [Income]

})

# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("---")

if st.button("🔍 Predict Diabetes Risk"):

    # Scale user input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Prediction probability
    probability = model.predict_proba(input_scaled)

    st.write("---")

    st.subheader("📌 Prediction Result")

    if prediction[0] == 1:

        confidence = probability[0][1] * 100

        st.error(
            f"""
            ⚠️ HIGH RISK OF DIABETES
            
            Prediction Confidence: {confidence:.2f}%
            """
        )

        st.warning(
            """
            Recommendation:
            
            • Consult a healthcare professional
            
            • Maintain healthy diet
            
            • Increase physical activity
            
            • Monitor blood sugar regularly
            """
        )

    else:

        confidence = probability[0][0] * 100

        st.success(
            f"""
            ✅ LOW RISK OF DIABETES
            
            Prediction Confidence: {confidence:.2f}%
            """
        )

        st.info(
            """
            Recommendation:
            
            • Continue healthy lifestyle
            
            • Maintain regular exercise
            
            • Follow balanced diet
            
            • Attend routine health checkups
            """
        )

# =========================================================
# FOOTER
# =========================================================

st.write("---")

st.caption(
    "Machine Learning Powered Diabetes Prediction System"
)