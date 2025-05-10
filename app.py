from flask import Flask, request, render_template, send_from_directory, redirect, url_for, flash, jsonify
import os
import whisper
import re
import time
import sys
from datetime import datetime
from werkzeug.utils import secure_filename
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For flash messages
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "flac", "ogg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Terminal style console output function
def terminal_print(message, message_type="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    color_codes = {
        "INFO": "\033[36m",  # Cyan
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "RESET": "\033[0m"  # Reset color
    }
    
    print(f"{color_codes[message_type]}[{timestamp}] {message_type}: {message}{color_codes['RESET']}")

# Cleanup old files
def cleanup_old_files(folder, max_age_hours=24):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_age = time.time() - os.path.getmtime(file_path)
            if file_age > max_age_hours * 3600:
                try:
                    os.remove(file_path)
                    terminal_print(f"Deleted old file: {filename}", "INFO")
                except OSError as e:
                    terminal_print(f"Failed to delete {filename}: {str(e)}", "WARNING")

# Load the model once at startup
terminal_print("Starting Word-by-Word Subtitle Generator", "INFO")
terminal_print("Initializing Whisper model...", "INFO")
model = whisper.load_model("tiny")
terminal_print("Whisper model initialized successfully", "SUCCESS")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def write_srt_file(subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sub in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time(sub['start'])} --> {format_time(sub['end'])}\n")
            f.write(f"{sub['word']}\n\n")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the post request has the file part
        if "audio" not in request.files:
            return jsonify({"status": "error", "message": "No file part in the request"}), 400
            
        audio = request.files["audio"]
        
        # If user submits an empty form
        if audio.filename == "":
            return jsonify({"status": "error", "message": "No selected file"}), 400
            
        if audio and allowed_file(audio.filename):
            # Secure the filename and add unique ID
            filename = secure_filename(audio.filename)
            unique_id = uuid4().hex[:8]
            base_name = f"{os.path.splitext(filename)[0]}_{unique_id}"
            filepath = os.path.join(UPLOAD_FOLDER, f"{base_name}{os.path.splitext(filename)[1]}")
            output_srt = os.path.join(OUTPUT_FOLDER, f"{base_name}.srt")
            audio.save(filepath)
            
            # Clean up old files
            cleanup_old_files(UPLOAD_FOLDER)
            cleanup_old_files(OUTPUT_FOLDER)
            
            try:
                # Terminal-style output
                terminal_print(f"Processing file: {filename}", "INFO")
                terminal_print("Starting transcription...", "INFO")
                
                # Transcribe the audio file
                result = model.transcribe(filepath, word_timestamps=True)
                terminal_print("Audio transcription complete", "SUCCESS")
                
                terminal_print("Extracting word timestamps...", "INFO")
                subtitles = []
                for segment in result["segments"]:
                    for word_info in segment["words"]:
                        word = re.sub(r"[^\w\s]", "", word_info["word"]).strip()
                        if word:
                            subtitles.append({
                                "word": word,
                                "start": word_info["start"],
                                "end": word_info["end"]
                            })
                
                terminal_print(f"Creating SRT file: {os.path.basename(output_srt)}", "INFO")
                # Write the SRT file
                write_srt_file(subtitles, output_srt)
                
                terminal_print(f"Generated {len(subtitles)} subtitle entries", "SUCCESS")
                terminal_print(f"SRT file saved: {os.path.basename(output_srt)}", "SUCCESS")
                
                # Return JSON response with download URL
                download_url = url_for("download_file", filename=os.path.basename(output_srt))
                return jsonify({
                    "status": "success",
                    "message": "Subtitles generated! Download now.",
                    "download_url": download_url
                }), 200
            
            except whisper.WhisperError as e:
                return jsonify({"status": "error", "message": f"Transcription failed: {str(e)}"}), 500
            except OSError as e:
                return jsonify({"status": "error", "message": f"File processing error: {str(e)}"}), 500
            except Exception as e:
                return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500
        else:
            return jsonify({
                "status": "error",
                "message": f"Unsupported file format. Please upload a file in one of these formats: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
    
    return render_template("index.html")

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)