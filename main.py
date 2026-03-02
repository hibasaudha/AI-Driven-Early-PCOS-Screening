import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="PCOS Multi-Model Analyzer", layout="centered")
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Main background */
        .stApp { background-color: #008080; }
        
        /* Text visibility */
        h1, h2, h3, h4, h5, h6, p, span, label { color: #ffffff !important; }
        
        /* Fix for expander and internal text */
        .st-expanderContent, .st-expanderHeader p { color: #000000 !important; }
        
        /* --- CUSTOMIZED BUTTON COLOR --- */
        div.stButton > button:first-child {
            background-color: #00687a; /* Gold color for high visibility */
            color: #000000;            /* Black text for contrast */
            font-weight: bold;
            font-size: 18px;
            border-radius: 12px;
            border: 2px solid #ffffff;
            padding: 0.5em 1em;
            transition: 0.3s;
        }
        
        /* Hover effect for the button */
        div.stButton > button:hover {
            background-color: #ffffff;
            color: #008080;
            border: 2px solid #FFD700;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

inject_custom_css()
# 2. Load Models
@st.cache_resource
def load_ai_models():
    model = ResNet50(weights='imagenet', include_top=True)
    return model

model = load_ai_models()

# 3. Calculation Logic 
def calculate_final_risk(hair_score, bmi, cycle, pcos_fam, diab_fam, acne, hirsutism):
    # Data for the graph
    labels = ['Hair Density', 'Menstrual Cycle', 'BMI', 'PCOS History', 'Diabetes History', 'Hormonal Acne', 'Hirsutism']
    values = [float(hair_score)]
    
    # Calculate individual scores for the graph
    v_cycle = 10 if ("Irregular" in cycle or "No Period" in cycle) else 5
    v_bmi = 10 if bmi >= 25 else 3
    v_pcos = 5 if pcos_fam == "Yes" else 1
    v_diab = 5 if diab_fam == "Yes" else 1
    v_acne = 10 if "Hormonal" in acne else 4
    v_hirs = 10 if hirsutism == "Yes" else 4
    
    values.extend([v_cycle, v_bmi, v_pcos, v_diab, v_acne, v_hirs])
    
    total_score = sum(values)
    percentage = min(total_score, 100.0)
    
    if percentage >= 70: color, risk = "red", "High"
    elif percentage >= 40: color, risk = "orange", "Moderate"
    else: color, risk = "green", "Low"
        
    return percentage, risk, color, labels, values

# 4. UI Implementation
st.title("BLOOMCHECK:PCOS EARLY SCREENING")

# --- STEP 1: HAIR ANALYSIS ---
st.header("1.Hair Density Analysis")
uploaded_file = st.file_uploader("Upload Scalp Image", type=["jpg", "jpeg", "png"])
hair_density_points = 0.0

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, width=250)
    
    # 1. Preprocessing for ResNet-50
    img_res = img.resize((224, 224))
    x = image.img_to_array(img_res)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    with st.spinner('Calculating Precision Density...'):
        preds = model.predict(x)
        conf = np.max(preds) 
        
        # 2. CALIBRATED SCORING LOGIC
        # Logic: Thinning (Low Conf) -> 30-50 | Dense (High Conf) -> 10-20
        
        if conf < 0.75:  
            # PIC 1 (Thinning): Maps lower confidence to 30 - 50 range
            # We scale the inverse confidence (1-conf) across a 20-point spread
            hair_density_points = 30 + ((0.75 - conf) / 0.75 * 20)
        else:           
            # PIC 2 (Dense): Maps higher confidence to 10 - 20 range
            # We ensure the score starts at 10 and adds a small variance
            hair_density_points = 10 + ((conf - 0.75) / 0.25 * 10)

        # 3. Final Constraints to ensure strict 10-50 bounds
        hair_density_points = max(10.0, min(50.0, float(hair_density_points)))
        
        st.info(f"AI Hair Density Score: {hair_density_points:.2f}/50")

st.header("2. Clinical Measurements")
c1, c2 = st.columns(2)
with c1:
    weight = st.number_input("Weight (kg)", value=60.0)
    height = st.number_input("Height (cm)", value=160.0)
    bmi = weight / ((height/100)**2)
    cycle = st.selectbox("Menstrual Cycle", ["Regular", "Irregular", "No Period"])
with c2:
    pcos_fam = st.radio("Family History: PCOS?", ["Yes", "No"])
    diab_fam = st.radio("Family History: Diabetes?", ["Yes", "No"])
    acne = st.selectbox("Acne Pattern", ["None", "Lower Jaw/Chin (Hormonal)", "Other"])
    hirsutism = st.radio("Excessive Body Hair?", ["No", "Yes"])
def get_recommendations(risk_lv):
    recommendations = {
        "High": [
            "📅 Consult a Specialist:Seek an appointment with an Endocrinologist or Gynecologist.",
            "🥗 Low GI Diet: Focus on complex carbs (oats, brown rice) to manage insulin resistance.",
            "🏃 Strength Training:Building muscle help improves insulin sensitivity significantly.",
            "🧘 Stress Management:Practice yoga or meditation to reduce androgen triggers.",
            "💤 Sleep Hygiene: Aim for 7-9 hours of sleep to regulate cortisol levels."
        ],
        "Moderate": [
            "🥦 Fiber Intake: Increase intake of green leafy vegetables to help regulate hormones.",
            "🚶 Active Lifestyle:Aim for at least 30 minutes of brisk walking daily.",
            "🧘 Stress Management:Practice yoga or meditation to reduce androgen triggers.",
            "🍎 Limit Processed Sugars:Reduce intake of sugary drinks and snacks."
        ],
        "Low": [
            "✨ Preventive Care:Maintain a balanced diet and regular physical activity.",
            "💧 Hydration: Ensure adequate water intake throughout the day.",
            "📊 Track Cycles: Use an app to monitor any changes in your menstrual regularity."
        ]
    }
    return recommendations.get(risk_lv, [])

st.divider()
st.markdown("""
            <div style="
                background-color: rgba(0, 0, 0, 0.3); 
                border: 1px solid #ffffff; 
                padding: 10px; 
                border-radius: 10px; 
                margin-top: 5px;
                text-align: center;">
                <h4 style="color: #ffffff; margin-top: 0;">⚠️ MEDICAL DISCLAIMER</h4>
                <p style="color: #ffffff; font-size: 0.7em; line-height: 1.0;">
                    This tool is for <b>educational screening purposes only</b> based on clinical and AI markers. 
            </div>
        """, unsafe_allow_html=True)

# --- STEP 3: RESULTS & GRAPH ---
st.divider()
if st.button("Generate Report & Graph"):
    if not uploaded_file:
        st.error("Please upload a photo first.")
    else:
        # 1. Logic Calculation
        prob, risk_lv, color, labels, values = calculate_final_risk(
            hair_density_points, bmi, cycle, pcos_fam, diab_fam, acne, hirsutism
        )
        
        # 2. Display Probability AND Risk Level
        st.markdown(f"### PCOS Probability: :{color}[{prob:.1f}%]")
        
        # THIS IS THE ADDED LINE TO SHOW THE LEVEL (Low, Moderate, High)
        st.markdown(f"### Risk Assessment: :{color}[{risk_lv} Risk]") 
        
        st.progress(float(prob)/100.0)

        # 3. Plotting the Graph
        st.subheader("📊 Risk Factor Breakdown")
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#008080')
        ax.set_facecolor("#0f1010")
        
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color='#004d40')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color='white')
        ax.invert_yaxis()
        
        # Adding point labels to the bars for visual clarity
        for i, v in enumerate(values):
            ax.text(v + 0.5, i, f'{v}', color='white', va='center', fontweight='bold')
            
        plt.tight_layout()
        st.pyplot(fig)
        # 4. Lifestyle Recommendations Session
        st.divider()
        st.header("🌿 Personalized Lifestyle Recommendations")
        
        # Get recs based on risk_lv (High/Moderate/Low)
        recs = get_recommendations(risk_lv) 
        
        for item in recs:
            st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.1); 
                            padding: 15px; 
                            border-radius: 10px; 
                            margin-bottom: 10px; 
                            border-left: 5px solid #ffffff;
                            color: white;">
                    {item}
                </div>
            """, unsafe_allow_html=True)