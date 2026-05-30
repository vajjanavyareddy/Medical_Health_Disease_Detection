import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="AI-Powered Pneumonia Detection",
    page_icon="🩺",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div style="
background: linear-gradient(90deg, #1565C0, #42A5F5);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:25px;
">

<h1 style="
color:white;
font-size:50px;
margin:0;
">

🩺 AI-Powered Pneumonia Detection System

</h1>

<p style="
color:white;
font-size:18px;
margin-top:10px;
">

AI-Assisted Chest X-Ray Screening & Clinical Decision Support

</p>

</div>
""", unsafe_allow_html=True)

# ==================================================
# LOAD TFLITE MODEL
# ==================================================
@st.cache_resource
def load_model():

    interpreter = tf.lite.Interpreter(
        model_path="pneumonia_model.tflite"
    )

    interpreter.allocate_tensors()

    return interpreter

interpreter = load_model()

# ==================================================
# IMAGE PREPROCESSING
# ==================================================
def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((224, 224))

    img_array = np.array(image)

    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.title("🩺 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Pneumonia Detection",
        "Model Information",
        "About"
    ]
)

# ==================================================
# MODEL INFORMATION
# ==================================================
if page == "Model Information":

    st.title("Model Information")

    st.markdown("""
### CNN Performance

- Accuracy: 82%
- Pneumonia Recall: 97.7%
- Pneumonia F1 Score: 0.87
- False Negatives: 9

### Model Architecture

- Conv2D
- MaxPooling2D
- Dropout
- Dense Layers
- Sigmoid Output Layer

### Business Impact

This system can assist healthcare professionals
in screening Chest X-Ray images for signs of
pneumonia, enabling faster preliminary diagnosis.
""")

# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "About":

    st.title("About This Project")

    st.markdown("""
### AI-Powered Medical Image Disease Detection

This application uses Deep Learning to detect
pneumonia from Chest X-Ray images.

### Workflow

1. Upload Chest X-Ray
2. Image Preprocessing
3. TFLite Model Inference
4. Prediction Generation
5. Clinical Recommendation

### Disclaimer

This tool is intended for educational and
decision-support purposes only.

Final diagnosis should always be made by
qualified healthcare professionals.
""")

# ==================================================
# MAIN PAGE
# ==================================================
else:

    st.subheader("Upload Chest X-Ray Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded X-Ray",
                use_container_width=True
            )

        with col2:

            with st.spinner("Analyzing X-Ray..."):

                processed_image = preprocess_image(image)

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                interpreter.set_tensor(
                    input_details[0]["index"],
                    processed_image
                )

                interpreter.invoke()

                prediction = interpreter.get_tensor(
                    output_details[0]["index"]
                )[0][0]

            if prediction >= 0.5:

                diagnosis = "PNEUMONIA"
                confidence = prediction * 100
                risk = "HIGH"

            else:

                diagnosis = "NORMAL"
                confidence = (1 - prediction) * 100
                risk = "LOW"

            st.subheader("Diagnosis Report")

            if diagnosis == "PNEUMONIA":

                st.error(
                    f"⚠️ Prediction: {diagnosis}"
                )

                st.warning(
                    "Signs consistent with pneumonia detected. "
                    "Medical evaluation is recommended."
                )

            else:

                st.success(
                    f"✅ Prediction: {diagnosis}"
                )

                st.info(
                    "No significant pneumonia indicators detected."
                )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.metric(
                "Risk Level",
                risk
            )

            st.write(
                "Raw Model Output:",
                round(float(prediction), 4)
            )

        st.markdown("---")

        st.subheader("Model Statistics")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Accuracy", "82%")

        with c2:
            st.metric("Recall", "97.7%")

        with c3:
            st.metric("F1 Score", "0.87")

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")

st.markdown(
    """
    <center>
    AI-Powered Pneumonia Detection System • Deep Learning Project
    </center>
    """,
    unsafe_allow_html=True
)