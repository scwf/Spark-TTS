import argparse
import base64
import io
import platform

import os
import soundfile as sf
import tempfile
import torch
from flask import Flask, jsonify, request

from cli.SparkTTS import SparkTTS


def create_app(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Determine device
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        device = torch.device(f"mps:{device}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device}")
    else:
        device = torch.device("cpu")

    model = SparkTTS(model_dir, device)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/tts", methods=["POST"])
    def tts():
        data = request.get_json(silent=True) or {}
        text = data.get("text") or request.form.get("text")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        prompt_text = data.get("prompt_text") or request.form.get("prompt_text")

        # Handle uploaded audio as base64 string or multipart file
        prompt_speech_b64 = data.get("prompt_speech") or request.form.get("prompt_speech")
        prompt_file = request.files.get("prompt_speech")
        tmp_path = None
        if prompt_file:
            speech_bytes = prompt_file.read()
        elif prompt_speech_b64:
            try:
                speech_bytes = base64.b64decode(prompt_speech_b64)
            except Exception:
                return jsonify({"error": "Invalid prompt_speech"}), 400
        else:
            speech_bytes = None

        if speech_bytes is not None:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_file.write(speech_bytes)
            tmp_file.flush()
            tmp_path = tmp_file.name
            tmp_file.close()

        gender = data.get("gender") or request.form.get("gender")
        pitch = data.get("pitch") or request.form.get("pitch")
        speed = data.get("speed") or request.form.get("speed")

        with torch.no_grad():
            wav = model.inference(
                text,
                tmp_path,
                prompt_text=prompt_text,
                gender=gender,
                pitch=pitch,
                speed=speed,
            )

        if tmp_path:
            os.unlink(tmp_path)
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, wav, samplerate=16000, format="WAV")
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"audio": audio_b64})

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="SparkTTS Flask API server")
    parser.add_argument("--model_dir", default="pretrained_models/Spark-TTS-0.5B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args.model_dir, args.device)
    app.run(host=args.host, port=args.port)
