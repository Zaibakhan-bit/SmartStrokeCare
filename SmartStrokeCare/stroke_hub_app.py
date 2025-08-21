import os, time, json, numpy as np, pandas as pd, streamlit as st
from PIL import Image
import webbrowser
import threading
import os

# Automatically open browser
def open_browser():
    webbrowser.open_new("http://localhost:8501")

# Run Streamlit and open browser in a separate thread
if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    os.system("streamlit run streamlit_hub_app.py")


# ---- optional voice ----
USE_VOICE = True
try:
    import pyttsx3; tts = pyttsx3.init(); tts.setProperty('rate', 150)
except Exception: USE_VOICE=False; tts=None
def speak(msg):
    if USE_VOICE and tts:
        try: tts.say(msg); tts.runAndWait()
        except: pass

# ---- optional Arduino serial ----
USE_ARDUINO = True
SERIAL_PORT = "COM3"   # change to your port; e.g. /dev/ttyACM0 on Linux
BAUD = 9600
ser = None
if USE_ARDUINO:
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1); time.sleep(2)
    except Exception:
        USE_ARDUINO=False; ser=None

# ---- vitals model ----
vitals_model = tf.keras.models.load_model("stroke_cnn_model.h5")
scaler = joblib.load("scaler.pkl")

# ---- imaging model ----
img_model = tf.keras.models.load_model("stroke_img_cnn.h5")
IMG_SIZE = (224,224)
class_map = {"no_stroke":0,"stroke":1}
if os.path.exists("class_indices.json"):
    try:
        with open("class_indices.json") as f: class_map = json.load(f)
    except: pass

st.set_page_config(page_title="Smart Stroke Care Hub", layout="wide")
st.title("🧠 Smart Stroke Care – Vitals + MRI/CT")

tab1, tab2 = st.tabs(["Vitals (Arduino/CSV)", "Imaging (MRI/CT)"])

# ---------------- VITALS TAB ----------------
with tab1:
    st.subheader("Vitals-based Prediction")
    c1,c2,c3 = st.columns(3)
    with c1: pulse = st.number_input("Pulse (bpm)", 40, 180, 92)
    with c2: temp  = st.number_input("Temperature (°C)", 35.0, 40.5, 38.1, step=0.1)
    with c3: trem  = st.slider("Tremor (0–2)", 0.0, 2.0, 0.6, 0.01)

    def predict_vitals(p,t,tr):
        x = np.array([[p,t,tr]], dtype=np.float32)
        xs = scaler.transform(x).reshape((1,3,1))
        prob = float(vitals_model.predict(xs, verbose=0)[0][0])
        label = 1 if prob>=0.5 else 0
        return prob, label

    if st.button("Predict (Vitals)"):
        prob, lab = predict_vitals(pulse,temp,trem)
        if lab==1:
            st.error(f"⚠️ High risk (p={prob:.2f})"); speak("Warning. High risk of stroke.")
            if ser: ser.write(b'H')
        else:
            st.success(f"✅ Low risk (p={prob:.2f})"); speak("No stroke risk detected.")
            if ser: ser.write(b'N')

    st.markdown("—")
    st.caption("Batch CSV prediction")
    up = st.file_uploader("Upload CSV with columns: pulse,temp,tremor", type=["csv"], key="csv1")
    if up:
        df = pd.read_csv(up)
        X = df[["pulse","temp","tremor"]].astype(np.float32)
        Xs = scaler.transform(X).reshape((-1,3,1))
        probs = vitals_model.predict(Xs, verbose=0).ravel()
        preds = (probs>=0.5).astype(int)
        df["prob"]=np.round(probs,3); df["pred"]=preds
        st.dataframe(df)

# ---------------- IMAGING TAB ----------------
with tab2:
    st.subheader("MRI/CT Image Classification")
    file = st.file_uploader("Upload MRI/CT image (jpg/png)", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)
        arr = np.array(img.resize(IMG_SIZE))/255.0
        arr = np.expand_dims(arr, axis=0)
        prob = float(img_model.predict(arr, verbose=0)[0][0])
        label = 1 if prob>=0.5 else 0
        if label==1:
            st.error(f"⚠️ Stroke detected (p={prob:.2f})"); speak("Stroke detected from imaging.")
            if ser: ser.write(b'H')
        else:
            st.success(f"✅ No stroke (p={prob:.2f})"); speak("No stroke found in the image.")
            if ser: ser.write(b'N')

st.caption("Demo only – not a medical device.")

