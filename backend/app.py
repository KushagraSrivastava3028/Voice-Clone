from flask import Flask, request, send_file
import numpy as np
import soundfile as sf
from io import BytesIO
from inference import Synthesizer
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Initialize synthesizer
synthesizer = Synthesizer(
    tacotron2_path="models/tacotron2_statedict.pt",
    waveglow_path="models/waveglow_256channels.pt"
)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.get_json()
        text = data['text']
        pitch = float(data.get('pitch', 0))
        speed = float(data.get('speed', 1.0))
        
        app.logger.info(f"Synthesizing: {text[:50]}... (pitch: {pitch}, speed: {speed})")
        
        audio = synthesizer.synthesize(text, pitch_shift=pitch, speed_factor=speed)
        
        # Create in-memory WAV file
        buffer = BytesIO()
        sf.write(buffer, audio, 22050, format='WAV')
        buffer.seek(0)
        
        return send_file(buffer, mimetype='audio/wav')
    
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return {"error": str(e)}, 500

@app.route('/health', methods=['GET'])
def health_check():
    return {"status": "healthy", "model_loaded": synthesizer.models_loaded}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
