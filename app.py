from flask import Flask, request, jsonify, send_file
import os
import speech_recognition as sr
from gtts import gTTS
from dotenv import load_dotenv
import google.generativeai as genai  # Import Gemini library
import logging # Added for better logging

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Setup ---
app = Flask(__name__)

# Temporary directory to store audio files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER # Good practice to store in app config

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY not found in environment variables.")
    # Depending on your deployment, you might want to exit or handle this differently
    # raise ValueError("GEMINI_API_KEY is required.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Gemini API configured successfully.")
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        # Handle configuration error appropriately

# --- Model Selection ---
# Using gemini-1.5-flash-latest as a modern, fast model.
# Replace with 'gemini-1.5-pro-latest' for potentially higher quality but maybe slower responses,
# or other available model names.
# MODEL_NAME = 'Gemma 3 1B' # Previous model
MODEL_NAME = 'gemini-1.5-flash-latest'
logging.info(f"Using Gemini model: {MODEL_NAME}")

# --- Flask Route ---
@app.route('/process-audio', methods=['POST'])
def process_audio():
    try:
        # Step 1: Receive and save the uploaded audio file
        if 'file' not in request.files:
            logging.warning("Request received without 'file' part.")
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("Request received with empty filename.")
            return jsonify({"error": "No selected file"}), 400

        # Consider adding file type/extension validation here
        # e.g., if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg')):
        #     return jsonify({"error": "Invalid file type"}), 400

        # Use a more robust way to handle filenames if needed (e.g., secure_filename)
        # For simplicity, keeping the original name logic but using app config
        input_filename = "input.wav" # Force WAV for sr.AudioFile consistency, might need conversion if input isn't WAV
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)

        try:
            file.save(filepath)
            logging.info(f"Step 1: Audio file saved successfully to {filepath}")
        except Exception as e:
            logging.error(f"Error saving uploaded file: {e}")
            return jsonify({"error": "Failed to save uploaded file"}), 500

        # Step 2: Convert audio to text using SpeechRecognition
        recognizer = sr.Recognizer()
        recognized_text = ""
        try:
            # Ensure the file is WAV or convert it first if necessary
            # For simplicity, this code assumes the input *is* WAV or sr can handle it.
            # Libraries like pydub can be used for conversion if needed.
            with sr.AudioFile(filepath) as source:
                # Adjust for ambient noise (optional but good practice)
                # recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logging.info("Recording audio from file...")
                audio_data = recognizer.record(source)
                logging.info("Attempting speech recognition...")
                recognized_text = recognizer.recognize_google(audio_data)  # Uses Google Speech Recognition API
                # Limit input length *after* recognition if needed (though Gemini handles long inputs well)
                # recognized_text = recognized_text[:1000] # Example limit increase
                logging.info(f"Step 2: Recognized text (raw): '{recognized_text}'")
        except sr.UnknownValueError:
            logging.warning("Speech Recognition could not understand audio.")
            # Optionally, you could try another recognizer here (e.g., whisper)
            return jsonify({"error": "Could not understand audio"}), 400
        except sr.RequestError as e:
            logging.error(f"Speech recognition request failed: {e}")
            return jsonify({"error": f"Speech recognition service error: {e}"}), 500
        except FileNotFoundError:
            logging.error(f"Audio file not found at {filepath} for recognition.")
            return jsonify({"error": "Internal server error: audio file missing"}), 500
        except Exception as e:
             logging.error(f"Unexpected error during speech recognition: {e}", exc_info=True)
             return jsonify({"error": "Error processing audio for speech recognition"}), 500

        # Step 3: Generate a response using Gemini
        response_text = ""
        if not recognized_text:
             logging.warning("No text recognized, skipping Gemini.")
             response_text = "I didn't understand what you said. Could you please repeat?"
        else:
            try:
                logging.info(f"Sending text to Gemini: '{recognized_text}'")
                # --- THIS IS THE KEY CHANGE ---
                model = genai.GenerativeModel(MODEL_NAME)
                # -----------------------------
                # Consider adding safety settings or generation config if needed
                # generation_config = genai.types.GenerationConfig(temperature=0.7)
                # safety_settings = [...]
                # response = model.generate_content(recognized_text, generation_config=generation_config, safety_settings=safety_settings)
                response = model.generate_content(recognized_text)

                # Robustly check response and access text
                if response.parts:
                    response_text = "".join(part.text for part in response.parts).strip()
                elif hasattr(response, 'text'): # Fallback for older/different response structures
                     response_text = response.text.strip()
                else:
                     # Handle cases where the response might be blocked or empty
                     logging.warning(f"Gemini response did not contain usable text. Response: {response}")
                     # Check for blocking reasons if available
                     block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason')
                     if block_reason:
                         logging.warning(f"Content blocked by Gemini. Reason: {block_reason}")
                         response_text = "I cannot provide a response to that due to safety guidelines."
                     else:
                         response_text = "I received your message but couldn't generate a response."


                logging.info(f"Step 3: Generated response: '{response_text}'")

            except Exception as e:
                logging.error(f"Gemini API call failed: {e}", exc_info=True)
                return jsonify({"error": f"Gemini API error: {str(e)}"}), 500

        # Step 4: Convert response text to speech using gTTS
        output_filename = "output.mp3"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        try:
            if not response_text:
                 logging.warning("Response text is empty, generating placeholder TTS.")
                 response_text="Sorry, I could not generate a response." # Ensure some text for TTS

            tts = gTTS(text=response_text, lang='en', slow=False) # slow=False for normal speed
            tts.save(output_path)
            logging.info(f"Step 4: MP3 file generated successfully at: {output_path}")
        except Exception as e:
            logging.error(f"gTTS error: {e}", exc_info=True)
            return jsonify({"error": f"Text-to-speech conversion error: {str(e)}"}), 500

        # Step 5: Return the audio file as a response
        if not os.path.exists(output_path):
             logging.error(f"Generated MP3 file not found at {output_path} after saving.")
             return jsonify({"error": "Internal server error: failed to create audio response"}), 500

        logging.info(f"Sending audio file: {output_path}")
        # Use try-finally to ensure cleanup even if send_file fails
        try:
            return send_file(
                output_path,
                mimetype="audio/mpeg",
                as_attachment=False, # Send inline if possible, else browser might download
                download_name="response.mp3" # Suggest a filename if downloaded
            )
        finally:
            # Optional: Clean up files after sending
            # Consider a more robust cleanup strategy for production (e.g., background task, TTL)
             try:
                 if os.path.exists(filepath):
                     os.remove(filepath)
                     logging.info(f"Cleaned up input file: {filepath}")
                 if os.path.exists(output_path):
                     os.remove(output_path)
                     logging.info(f"Cleaned up output file: {output_path}")
             except Exception as e:
                 logging.error(f"Error during file cleanup: {e}")


    except Exception as e:
        # Catch-all for unexpected errors during request handling
        logging.error(f"An unexpected error occurred in /process-audio: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network
    # Set debug=False for production environments
    app.run(host='0.0.0.0', port=5000, debug=True)