import os
import torch
import soundfile as sf
import logging
import platform
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_file

from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

logging.basicConfig(level=logging.INFO)

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device_idx=0):
    logging.info(f"Loading model from: {model_dir}")
    if platform.system() == "Darwin":
        device = torch.device(f"mps:{device_idx}")
        logging.info(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_idx}")
        logging.info(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        logging.info("GPU acceleration not available, using CPU")

    try:
        model = SparkTTS(model_dir, device)
        return model
    except Exception as e:
        logging.error(f"Error loading SparkTTS model: {e}")
        raise

def create_app(model_dir="pretrained_models/Spark-TTS-0.5B"):
    app = Flask(__name__)
    app.model = initialize_model(model_dir=model_dir)

    @app.route('/synthesize', methods=['POST'])
    def synthesize():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 415

        data = request.get_json()
        text = data.get('text')

        if not text:
            return jsonify({"error": "Missing text parameter"}), 400

        prompt_text = data.get('prompt_text')
        prompt_speech_path = data.get('prompt_speech_path')
        gender = data.get('gender')

        pitch_param = data.get('pitch')
        speed_param = data.get('speed')

        pitch = None
        if pitch_param is not None:
            try:
                pitch = LEVELS_MAP_UI[int(pitch_param)]
            except (ValueError, KeyError):
                logging.warning(f"Invalid pitch value: {pitch_param}. Using model default.")

        speed = None
        if speed_param is not None:
            try:
                speed = LEVELS_MAP_UI[int(speed_param)]
            except (ValueError, KeyError):
                logging.warning(f"Invalid speed value: {speed_param}. Using model default.")

        try:
            with torch.no_grad():
                wav_output = app.model.inference(
                    text,
                    prompt_speech_path,
                    prompt_text,
                    gender,
                    pitch,
                    speed,
                )

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, wav_output, samplerate=16000)
            temp_file.close()

            response = send_file(temp_file.name, as_attachment=True, download_name="output.wav", mimetype="audio/wav")

            @response.call_on_close
            def cleanup():
                os.remove(temp_file.name)

            return response

        except Exception as e:
            logging.error(f"Error during synthesis: {e}")
            return jsonify({"error": "Synthesis failed", "details": str(e)}), 500

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
