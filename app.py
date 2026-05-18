
import streamlit as st
import pickle
import numpy as np

with open("asd_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("ASD Screening Tool")
st.write("Answer each question by moving the slider. 1 = rarely, 10 = always.")

social    = st.slider("Responds to social cues",       1, 10, 5)
eye       = st.slider("Makes eye contact",             1, 10, 5)
repetitive= st.slider("Shows repetitive behaviors",   1, 10, 5)
comm      = st.slider("Has communication delays",     1, 10, 5)
sensory   = st.slider("Sensitive to sounds/textures", 1, 10, 5)
routine   = st.slider("Prefers strict routines",      1, 10, 5)
emotion   = st.slider("Recognizes emotions in others",1, 10, 5)
peer      = st.slider("Engages with peers",           1, 10, 5)

if st.button("Run screening"):
    input_data = np.array([[social, eye, repetitive, comm,
                            sensory, routine, emotion, peer]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    if prediction == 1:
        st.warning(f"Screening suggests further evaluation may be helpful.")
    else:
        st.success(f"Screening does not indicate ASD traits.")
    st.write(f"Confidence: {probability:.0%}")
    st.caption("This tool is for educational purposes only, not medical diagnosis.")
