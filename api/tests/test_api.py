import unittest
import json
import os
# import sys # sys is no longer needed
import requests

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # No longer needed
# from api.app import create_app # No longer needed

# Definition of SparkTTSClient class
class SparkTTSClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url.rstrip('/') # Ensure no trailing slash

    def synthesize(self, text, prompt_text=None, prompt_speech_path=None,
                   gender=None, pitch=None, speed=None, timeout=60):
        url = f"{self.base_url}/synthesize"

        # text is required by the client's synthesize method.
        # If text is None or empty, it's up to the server to validate.
        payload = {"text": text}

        if prompt_text:
            payload["prompt_text"] = prompt_text
        if prompt_speech_path:
            payload["prompt_speech_path"] = prompt_speech_path
        if gender:
            payload["gender"] = gender
        if pitch is not None:
            payload["pitch"] = pitch
        if speed is not None:
            payload["speed"] = speed

        try:
            response = requests.post(url, json=payload, timeout=timeout)
            return response
        except requests.exceptions.Timeout:
            print(f"Request to {url} timed out after {timeout} seconds.")
            # For tests, it's often better to let exceptions propagate or return None
            # and assert on that None to indicate failure.
            return None # Or re-raise custom exception
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during the request to {url}: {e}")
            return None # Or re-raise custom exception

# Refactored TestAPI class for end-to-end tests
class TestAPI(unittest.TestCase):
    def setUp(self):
        # This client will make requests to a live server
        self.client = SparkTTSClient(base_url="http://127.0.0.1:5000")
        # Ensure the Flask server (api/app.py) is running before executing these tests.

    def test_synthesize_simple_text(self):
        response = self.client.synthesize(text="Hello world")

        self.assertIsNotNone(response, "API request failed or timed out")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], "audio/wav")
        self.assertTrue(len(response.content) > 0, "Response content should not be empty")

    def test_synthesize_missing_text(self):
        # To test missing 'text' key, we need to craft a payload without it.
        # The SparkTTSClient.synthesize method currently requires 'text'.
        # So, we use requests.post directly for this specific case.
        payload = {} # Empty payload, 'text' key is missing
        url = f"{self.client.base_url}/synthesize"

        try:
            response = requests.post(url, json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            self.fail(f"Request failed for missing text test: {e}")
            return

        self.assertEqual(response.status_code, 400)
        try:
            json_response = response.json() # Use response.json() for requests library
        except json.JSONDecodeError:
            self.fail("Response for missing text was not valid JSON.")

        self.assertIn("error", json_response)
        # The error message in api/app.py is "Missing text parameter"
        self.assertEqual(json_response["error"], "Missing text parameter")


    def test_synthesize_with_voice_creation_params(self):
        response = self.client.synthesize(
            text="Custom voice test with parameters",
            gender="female",
            pitch=2,
            speed=4
        )

        self.assertIsNotNone(response, "API request failed or timed out")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['Content-Type'], "audio/wav")
        self.assertTrue(len(response.content) > 0, "Response content for custom voice should not be empty")

if __name__ == '__main__':
    # Note: These tests require the Flask server (api/app.py) to be running separately.
    unittest.main()
