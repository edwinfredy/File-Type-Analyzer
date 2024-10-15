import numpy as np
import lightgbm as lgb
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

model_save_path = 'model.txt'
print(f'Loading the model...')
bst = lgb.Booster(model_file=model_save_path)
print(f'Model loaded.')

def process_file_into_pieces(file_content, piece_size=1024):
    pieces = []
    while file_content:
        piece = file_content[:piece_size]
        file_content = file_content[piece_size:]
        if len(piece) < piece_size:
            piece += bytes([0] * (piece_size - len(piece)))
        pieces.append(list(piece))
    return pieces

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    file_content = file.read()

    file_pieces = process_file_into_pieces(file_content)

    data = np.array(file_pieces, dtype=np.uint8)
    counts = [0, 0, 0, 0, 0]

    total_pieces = len(data)
    for i, piece_data in enumerate(data):
        piece_data = piece_data.reshape(1, -1)
        y_pred = bst.predict(piece_data)
        y_pred_label = np.argmax(y_pred, axis=1)[0]
        counts[y_pred_label] += 1
        progress = int((i + 1) / total_pieces * 100)
        socketio.emit('progress', {'progress': progress})

    prediction_result = {
        "Image": f"{round((counts[0] / total_pieces) * 100, 2)}%",
        "Executable": f"{round((counts[1] / total_pieces) * 100, 2)}%",
        "Document": f"{round((counts[2] / total_pieces) * 100, 2)}%",
        "Audio": f"{round((counts[3] / total_pieces) * 100, 2)}%",
        "Video": f"{round((counts[4] / total_pieces) * 100, 2)}%"
    }
    print(prediction_result)

    return jsonify(prediction_result)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
