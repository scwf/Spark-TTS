import base64
from unittest.mock import MagicMock, patch

import numpy as np

from flask_api import create_app


@patch('flask_api.SparkTTS')
def test_health_endpoint(mock_sparktts):
    mock_sparktts.return_value = MagicMock()
    app = create_app()
    client = app.test_client()
    resp = client.get('/health')
    assert resp.status_code == 200
    assert resp.get_json() == {'status': 'ok'}


@patch('flask_api.SparkTTS')
def test_tts_endpoint(mock_sparktts):
    mock_model = MagicMock()
    mock_model.inference.return_value = np.zeros(16000)
    mock_sparktts.return_value = mock_model

    app = create_app()
    client = app.test_client()
    audio_b64 = base64.b64encode(b'00').decode('utf-8')
    resp = client.post('/tts', json={'text': 'hello', 'prompt_speech': audio_b64})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'audio' in data
    audio_bytes = base64.b64decode(data['audio'])
    assert len(audio_bytes) > 0
    with open('tests/generated_audio.wav', 'wb') as f:
        f.write(audio_bytes)
    args, kwargs = mock_model.inference.call_args
    assert args[0] == 'hello'
    assert args[1] is not None
