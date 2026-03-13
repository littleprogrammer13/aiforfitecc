from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from engine import NanoTransformer

app = Flask(__name__)

# Carregar Dados
with open('treino.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

chars = sorted(list(set(text_data)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}

# Inicializar e Treinar
model = NanoTransformer(len(chars))
model.treinar(text_data, char_to_int)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_msg = request.json.get("msg", "")
    resposta = model.gerar(user_msg, 30, char_to_int, int_to_char)
    return jsonify({"answer": resposta})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
