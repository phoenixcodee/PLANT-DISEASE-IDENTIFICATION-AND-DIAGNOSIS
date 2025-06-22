import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


# --- Streamlit app UI ---
st.set_page_config(
    page_title="Plant Disease Diagnosis",
    page_icon="🌿",
    layout="centered"
)

# Custom styled HTML header
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 48px;
        color: #2E8B57;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #6e6e6e;
        margin-bottom: 20px;
    }
    .uploadbox .css-1y0tads {
        border: 2px dashed #2E8B57;
        padding: 20px;
    }
    </style>
    <h1 class='main-title'>🌿 Plant Disease Identification & Diagnosis</h1>
    <div class='subtitle'>Upload a leaf image and let our AI identify potential plant diseases</div>
""", unsafe_allow_html=True)

st.title("🌿 Plant Disease Diagnosis App")
st.write("Upload a leaf image and let the AI diagnose its health status.")


# -------------- Load your model --------------
@st.cache_resource
def load_plant_model():
    model = load_model("plant_disease_model_retrained.3.keras")
    return model

model = load_plant_model()

# -------------- Class names --------------
class_names = [
    "Pepper__bell___Bacterial_spot", "Pepper__bell___healthy",
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight",
    "Tomato__Target_Spot", "Tomato__Tomato_mosaic_virus", "Tomato__Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_healthy",
    "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# -------------- Disease info dict (use your original one here) --------------
disease_info= {
    "Pepper__bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "taxonomy": "Capsicum annuum",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "cause": "Bacterium Xanthomonas campestris",
        "deficiency": "May resemble magnesium deficiency",
        "diagnosis": "Small, water-soaked spots that enlarge and turn brown with yellow halos"
    },
    "Pepper__bell___healthy": {
        "plant": "Bell Pepper",
        "taxonomy": "Capsicum annuum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No visible disease symptoms"
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Diseased",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani",
        "deficiency": "Possible potassium deficiency if severe",
        "diagnosis": "Brown concentric spots on leaves with yellow margins"
    },
    "Potato___healthy": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No symptoms observed"
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "taxonomy": "Solanum tuberosum",
        "status": "Diseased",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans",
        "deficiency": "May resemble calcium deficiency",
        "diagnosis": "Large, irregular brown lesions with white mold underneath leaves"
    },
    "Tomato__Target_Spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Target Spot",
        "cause": "Fungus Corynespora cassiicola",
        "deficiency": "May mimic potassium or magnesium deficiency",
        "diagnosis": "Dark, circular spots with concentric rings, especially on older leaves"
    },
    "Tomato__Tomato_mosaic_virus": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Tomato Mosaic Virus",
        "cause": "Tobamovirus",
        "deficiency": "Can be confused with iron deficiency",
        "diagnosis": "Mottled or mosaic light/dark green leaf patterns with distortion"
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "cause": "Begomovirus transmitted by whiteflies",
        "deficiency": "May resemble nitrogen deficiency",
        "diagnosis": "Upward leaf curling, yellowing, stunted growth"
    },
    "Tomato_Bacterial_spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas spp.",
        "deficiency": "May resemble salt stress or toxicity",
        "diagnosis": "Dark, greasy-looking leaf spots, may merge and kill leaves"
    },
    "Tomato_Early_blight": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Early Blight",
        "cause": "Alternaria solani",
        "deficiency": "May mimic magnesium deficiency",
        "diagnosis": "Dark brown spots with concentric rings, lower leaf drop"
    },
    "Tomato_healthy": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Healthy",
        "disease": "None",
        "cause": "N/A",
        "deficiency": "N/A",
        "diagnosis": "No symptoms; normal leaf and stem appearance"
    },
    "Tomato_Late_blight": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Late Blight",
        "cause": "Phytophthora infestans",
        "deficiency": "May resemble bacterial canker in severe stages",
        "diagnosis": "Large, gray-green water-soaked lesions, white mold under humid conditions"
    },
    "Tomato_Leaf_Mold": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Leaf Mold",
        "cause": "Fungus Passalora fulva (Cladosporium)",
        "deficiency": "Can be confused with aging leaves",
        "diagnosis": "Yellow patches on top of leaves and olive-green mold underneath"
    },
    "Tomato_Septoria_leaf_spot": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Septoria Leaf Spot",
        "cause": "Fungus Septoria lycopersici",
        "deficiency": "May be confused with nutrient burn or chemical damage",
        "diagnosis": "Small, circular spots with dark brown margins and gray centers"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "plant": "Tomato",
        "taxonomy": "Solanum lycopersicum",
        "status": "Diseased",
        "disease": "Spider Mite Infestation",
        "cause": "Tetranychus urticae (Two-Spotted Spider Mite)",
        "deficiency": "May resemble chlorosis or zinc deficiency",
        "diagnosis": "Stippled yellow leaves with fine webbing, typically on undersides"
    }
}



# Upload image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Diagnose"):
        with st.spinner('🔍 Analyzing image...'):
            # Preprocess image
            img_resized = img.resize((128, 128))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction))

            # Get disease info
            raw_class = class_names[predicted_index] if predicted_index < len(class_names) else None
            info = disease_info.get(raw_class, {
                "plant": "Unknown",
                "taxonomy": "Unknown",
                "status": "Unknown",
                "disease": "Unknown",
                "cause": "Unknown",
                "deficiency": "Unknown",
                "diagnosis": "Unknown"
            })

        # Show results
        st.markdown("---")
        st.markdown(f"### 🪴 Plant: **{info['plant']}**  _(Taxonomy: *{info['taxonomy']}*)_")
        st.markdown(f"### 🌿 Status: **{info['status']}**")
        st.markdown(f"### 🦠 Disease: **{info['disease']}**")
        st.markdown(f"### 📌 Cause: {info['cause']}")
        st.markdown(f"### 🥕 Nutrient Deficiency: {info['deficiency']}")
        st.markdown(f"### 🧪 Diagnosis: {info['diagnosis']}")

        st.markdown(f"### 🧠 Confidence: **{confidence * 100:.2f}%**")
        if confidence < 0.7:
            st.warning("⚠️ Low confidence — please consider manual verification.")
            
    # ✅ Moved here to avoid error
    st.success(f"✅ AI confidently identified this as **{info['disease']}** with **{confidence * 100:.2f}%** certainty.")

 # ✅ Generate downloadable report here
    report = f"""
🌿 PLANT DISEASE DIAGNOSIS REPORT

🪴 Plant: {info['plant']}
🔬 Taxonomy: {info['taxonomy']}
🌿 Status: {info['status']}
🦠 Disease: {info['disease']}
📌 Cause: {info['cause']}
🥕 Nutrient Deficiency: {info['deficiency']}
🧪 Diagnosis: {info['diagnosis']}
🧠 Confidence: {confidence * 100:.2f}%
"""

    st.download_button(
        label="📥 Download Diagnosis Report",
        data=report,
        file_name="plant_disease_report.txt",
        mime="text/plain"
    )

else:
    st.info("👆 Please upload a leaf image to start diagnosis.")




# Footer
st.markdown("""
    <style>
    body {
        background-image:url('background.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>

    <hr style="border-top: 1px solid #bbb;">
    <div style='text-align: center; color: gray; font-size: 15px;'>
        Developed with 💚 by <strong>Jaydish Kennedy J</strong><br>
        AI-ML Developer | Plant Health for Smarter Farming
    </div>
""", unsafe_allow_html=True)



