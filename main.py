import streamlit as st
import numpy as np
import lightgbm as lgb

# Load the model
model_save_path = 'model.txt'
st.write("Loading the model...")
bst = lgb.Booster(model_file=model_save_path)
st.success("Model loaded.")

def process_file_into_pieces(file_content, piece_size=1024):
    pieces = []
    while file_content:
        piece = file_content[:piece_size]
        file_content = file_content[piece_size:]
        if len(piece) < piece_size:
            piece += bytes([0] * (piece_size - len(piece)))
        pieces.append(list(piece))
    return pieces

# Streamlit app
st.title("File Type Analyzer")
uploaded_file = st.file_uploader("Upload a file to analyze", type=None)

if uploaded_file is not None:
    file_content = uploaded_file.read()
    file_pieces = process_file_into_pieces(file_content)

    data = np.array(file_pieces, dtype=np.uint8)
    counts = [0, 0, 0, 0, 0]

    progress_bar = st.progress(0)
    status_text = st.empty()

    total_pieces = len(data)
    for i, piece_data in enumerate(data):
        piece_data = piece_data.reshape(1, -1)
        y_pred = bst.predict(piece_data)
        y_pred_label = np.argmax(y_pred, axis=1)[0]
        counts[y_pred_label] += 1

        progress = int((i + 1) / total_pieces * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing... {progress}%")

    prediction_result = {
        "Image": f"{round((counts[0] / total_pieces) * 100, 2)}%",
        "Executable": f"{round((counts[1] / total_pieces) * 100, 2)}%",
        "Document": f"{round((counts[2] / total_pieces) * 100, 2)}%",
        "Audio": f"{round((counts[3] / total_pieces) * 100, 2)}%",
        "Video": f"{round((counts[4] / total_pieces) * 100, 2)}%"
    }

    st.subheader("Prediction Result")
    st.json(prediction_result)
