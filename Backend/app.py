from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import uuid
import shutil
import zipfile
import time
import threading
import re
import math
from pathlib import Path
import hashlib
import gc
import psutil
import logging
from datetime import datetime, timedelta
import json
from collections import deque
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
app = Flask(__name__)

# Enable CORS for all routes and all origins
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try different translation libraries
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_ENGINE = "deep_translator"
except ImportError:
    try:
        from translate import Translator
        TRANSLATOR_ENGINE = "translate"
    except ImportError:
        try:
            from googletrans import Translator as GTTranslator
            TRANSLATOR_ENGINE = "googletrans"
        except ImportError:
            TRANSLATOR_ENGINE = None
            logger.error("No translation library found!")

app = Flask(__name__)
CORS(app)

# Rate limiting with higher limits for large files
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour", "20 per minute"],
    storage_uri="memory://",
)

# ---------------- CONFIG FOR LARGE FILES ----------------
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB (increased for very large files)
BATCH_CHAR_LIMIT = 4000  # Conservative limit for Google Translate
BATCH_WORD_LIMIT = 500   # Smaller batches for better memory management
MAX_PARALLEL_BATCHES = 3  # Number of batches to translate concurrently
PROJECT_EXPIRY = 60 * 60  # 1 hour for large files
BASE_OUTPUT = Path("output_large")
BASE_OUTPUT.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'.txt', '.md'}
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ru': 'Russian',
    'ar': 'Arabic',
    'pt': 'Portuguese',
    'it': 'Italian',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'vi': 'Vietnamese',
    'th': 'Thai',
    'uk': 'Ukrainian',
    'ro': 'Romanian',
    'el': 'Greek',
    'he': 'Hebrew',
    'id': 'Indonesian',
    'ms': 'Malay',
    'fil': 'Filipino',
    'sw': 'Swahili'
}

# Global state for tracking large file processing
active_translations = {}
translation_progress = {}
completed_translations = deque(maxlen=50)  # Store last 50 translations
cleanup_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES)

# ---------------- MEMORY MANAGEMENT ----------------
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def check_memory_limit(additional_mb=100):
    """Check if we have enough memory for additional processing"""
    current_mb = get_memory_usage()
    total_mb = psutil.virtual_memory().total / 1024 / 1024
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    
    # Don't use more than 70% of available memory
    return available_mb > additional_mb and current_mb < total_mb * 0.7

# ---------------- LARGE FILE PROCESSING ----------------
def split_into_chunks(file_path, chunk_size=1024*1024):  # 1MB chunks
    """Split large file into manageable chunks"""
    chunk_paths = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        chunk_num = 1
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            # Ensure we break at sentence boundary if possible
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point != -1 and len(chunk) - break_point < 1000:
                actual_chunk = chunk[:break_point+1]
                # Seek back to the break point
                f.seek(f.tell() - (len(chunk) - break_point - 1))
            else:
                actual_chunk = chunk
            
            chunk_file = file_path.parent / f"chunk_{chunk_num:04d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as cf:
                cf.write(actual_chunk)
            
            chunk_paths.append(chunk_file)
            chunk_num += 1
    
    return chunk_paths

def estimate_word_count(file_path):
    """Quickly estimate word count without loading entire file"""
    word_count = 0
    sample_size = 1024 * 100  # 100KB sample
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Read first 100KB
        sample = f.read(sample_size)
        words_in_sample = len(re.findall(r'\b\w+\b', sample))
        
        # Get file size
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        
        # Estimate total words
        word_count = int((words_in_sample / sample_size) * file_size)
    
    return word_count

def process_chunk(chunk_path, target_lang, chunk_id, total_chunks):
    """Process a single chunk and return translated batches"""
    logger.info(f"Processing chunk {chunk_id}/{total_chunks}")
    
    with open(chunk_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Simple sentence splitting for speed
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    batches = []
    current_batch = []
    current_chars = 0
    
    for sentence in sentences:
        sentence_chars = len(sentence)
        if current_chars + sentence_chars > BATCH_CHAR_LIMIT and current_batch:
            batches.append(' '.join(current_batch))
            current_batch = [sentence]
            current_chars = sentence_chars
        else:
            current_batch.append(sentence)
            current_chars += sentence_chars + 1
    
    if current_batch:
        batches.append(' '.join(current_batch))
    
    # Translate batches in parallel
    translated_batches = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_BATCHES) as chunk_executor:
        future_to_batch = {
            chunk_executor.submit(translate_text, batch, target_lang): idx
            for idx, batch in enumerate(batches)
        }
        
        for future in as_completed(future_to_batch):
            idx = future_to_batch[future]
            try:
                translated = future.result()
                translated_batches.append((idx, translated))
            except Exception as e:
                logger.error(f"Error translating batch {idx}: {e}")
                translated_batches.append((idx, f"[Translation Error: {str(e)}]"))
    
    # Sort by original index
    translated_batches.sort(key=lambda x: x[0])
    return [tb[1] for tb in translated_batches]

def translate_text(text, target_lang, source_lang='auto'):
    """Translate text with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if TRANSLATOR_ENGINE == "deep_translator":
                translator = GoogleTranslator(source=source_lang, target=target_lang)
                # Split into smaller chunks if needed
                if len(text) > 3000:
                    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated = translator.translate(chunk)
                        translated_chunks.append(translated)
                        time.sleep(0.1)  # Small delay to avoid rate limiting
                    return ' '.join(translated_chunks)
                return translator.translate(text)
            
            elif TRANSLATOR_ENGINE == "translate":
                translator = Translator(to_lang=target_lang, from_lang=source_lang)
                if len(text) > 3000:
                    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated = translator.translate(chunk)
                        translated_chunks.append(translated)
                        time.sleep(0.1)
                    return ' '.join(translated_chunks)
                return translator.translate(text)
            
            elif TRANSLATOR_ENGINE == "googletrans":
                translator = GTTranslator()
                result = translator.translate(text, dest=target_lang, src=source_lang)
                return result.text
        
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay * (attempt + 1))
    
    return f"[Translation Failed]"

# ---------------- ROUTES FOR LARGE FILES ----------------
@app.route('/api/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_large_file():
    """Upload and start processing a large file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    target_lang = request.form.get('language', 'en')
    process_mode = request.form.get('mode', 'batches')  # batches or combined
    
    # Validate
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': 'Only .txt and .md files are supported for large files'}), 400
    
    # Generate project ID
    project_id = str(uuid.uuid4())
    project_folder = BASE_OUTPUT / f"project_{project_id}"
    project_folder.mkdir(exist_ok=True)
    
    # Save uploaded file
    original_filename = re.sub(r'[^\w\.\-]', '_', file.filename)[:100]
    original_path = project_folder / f"original_{original_filename}"
    
    # Save in chunks to avoid memory issues
    chunk_size = 1024 * 1024  # 1MB chunks
    with open(original_path, 'wb') as f:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    
    # Estimate file stats
    file_size_mb = os.path.getsize(original_path) / 1024 / 1024
    estimated_words = estimate_word_count(original_path)
    
    # Store in active translations
    with cleanup_lock:
        active_translations[project_id] = {
            'project_id': project_id,
            'filename': original_filename,
            'target_lang': target_lang,
            'file_size_mb': round(file_size_mb, 2),
            'estimated_words': estimated_words,
            'status': 'processing',
            'start_time': time.time(),
            'chunks_processed': 0,
            'total_chunks': 0,
            'mode': process_mode,
            'project_folder': str(project_folder)
        }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_large_file_background,
        args=(project_id, original_path, target_lang, process_mode),
        daemon=True
    )
    thread.start()
    
    return jsonify({
        'success': True,
        'project_id': project_id,
        'message': 'File uploaded successfully. Processing started.',
        'estimated_words': estimated_words,
        'file_size_mb': round(file_size_mb, 2),
        'status_url': f'/api/status/{project_id}'
    })

def process_large_file_background(project_id, file_path, target_lang, mode):
    """Background processing of large file"""
    try:
        project_folder = Path(file_path).parent
        
        # Update status
        with cleanup_lock:
            if project_id in active_translations:
                active_translations[project_id]['status'] = 'splitting'
        
        # Split into chunks
        chunk_paths = split_into_chunks(file_path)
        total_chunks = len(chunk_paths)
        
        with cleanup_lock:
            if project_id in active_translations:
                active_translations[project_id].update({
                    'total_chunks': total_chunks,
                    'chunks_processed': 0,
                    'status': 'translating'
                })
        
        # Process each chunk
        translated_files = []
        
        for chunk_idx, chunk_path in enumerate(chunk_paths, 1):
            # Check memory before processing
            if not check_memory_limit(200):  # Need at least 200MB free
                logger.warning(f"Low memory, pausing chunk {chunk_idx}")
                time.sleep(10)
                gc.collect()
            
            # Process chunk
            translated_batches = process_chunk(chunk_path, target_lang, chunk_idx, total_chunks)
            
            # Save translated chunk
            if mode == 'batches':
                # Save each batch as separate file
                for batch_idx, batch_text in enumerate(translated_batches, 1):
                    batch_file = project_folder / f"chunk{chunk_idx:04d}_batch{batch_idx:04d}.txt"
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        f.write(batch_text)
                    translated_files.append(batch_file)
            else:
                # Save as combined file per chunk
                combined_file = project_folder / f"chunk{chunk_idx:04d}_translated.txt"
                with open(combined_file, 'w', encoding='utf-8') as f:
                    for batch_text in translated_batches:
                        f.write(batch_text + '\n\n')
                translated_files.append(combined_file)
            
            # Update progress
            with cleanup_lock:
                if project_id in active_translations:
                    active_translations[project_id]['chunks_processed'] = chunk_idx
            
            # Clean up chunk file
            chunk_path.unlink()
            
            # Small delay to avoid overwhelming translation service
            if chunk_idx % 5 == 0:
                time.sleep(1)
        
        # Create zip file
        zip_filename = BASE_OUTPUT / f"translated_{project_id}.zip"
        
        if mode == 'batches':
            # For batches mode, organize by chunk folders
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in translated_files:
                    arcname = f"translated_batches/{file_path.name}"
                    zipf.write(file_path, arcname)
                    file_path.unlink()  # Clean up individual files
        else:
            # For combined mode, just zip the files
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in translated_files:
                    zipf.write(file_path, file_path.name)
                    file_path.unlink()
        
        # Create manifest file
        manifest = {
            'project_id': project_id,
            'original_filename': file_path.name.replace('original_', ''),
            'target_language': target_lang,
            'translation_date': datetime.now().isoformat(),
            'total_chunks': total_chunks,
            'mode': mode,
            'file_size_mb': round(os.path.getsize(file_path) / 1024 / 1024, 2)
        }
        
        manifest_file = project_folder / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        # Update status to completed
        with cleanup_lock:
            if project_id in active_translations:
                active_translations[project_id]['status'] = 'completed'
                active_translations[project_id]['completion_time'] = time.time()
                active_translations[project_id]['zip_file'] = str(zip_filename)
                
                # Move to completed
                completed_translations.append({
                    'project_id': project_id,
                    **active_translations[project_id]
                })
                
                # Remove from active (but keep in memory for status checks)
        
        logger.info(f"Completed processing project {project_id}")
        
    except Exception as e:
        logger.error(f"Error processing project {project_id}: {e}", exc_info=True)
        with cleanup_lock:
            if project_id in active_translations:
                active_translations[project_id]['status'] = 'error'
                active_translations[project_id]['error'] = str(e)

@app.route('/api/status/<project_id>', methods=['GET'])
def get_translation_status(project_id):
    """Get status of translation project"""
    with cleanup_lock:
        project_data = active_translations.get(project_id)
    
    if not project_data:
        # Check if completed recently
        for completed in completed_translations:
            if completed['project_id'] == project_id:
                project_data = completed
                break
    
    if not project_data:
        return jsonify({'error': 'Project not found'}), 404
    
    # Calculate progress
    if project_data['status'] == 'completed':
        progress = 100
    elif project_data['status'] == 'error':
        progress = 0
    else:
        total = project_data.get('total_chunks', 1)
        processed = project_data.get('chunks_processed', 0)
        progress = min(99, int((processed / total) * 100)) if total > 0 else 0
    
    response = {
        'project_id': project_id,
        'status': project_data['status'],
        'progress': progress,
        'filename': project_data['filename'],
        'target_lang': project_data['target_lang'],
        'file_size_mb': project_data['file_size_mb'],
        'estimated_words': project_data.get('estimated_words', 0),
        'chunks_processed': project_data.get('chunks_processed', 0),
        'total_chunks': project_data.get('total_chunks', 0),
        'mode': project_data.get('mode', 'batches')
    }
    
    if project_data['status'] == 'completed':
        response['download_url'] = f'/api/download/{project_id}'
        response['completion_time'] = project_data.get('completion_time', time.time())
    
    if project_data['status'] == 'error':
        response['error'] = project_data.get('error', 'Unknown error')
    
    return jsonify(response)

@app.route('/api/download/<project_id>', methods=['GET'])
def download_translated_file(project_id):
    """Download translated files"""
    with cleanup_lock:
        project_data = active_translations.get(project_id)
        if not project_data:
            # Check completed
            for completed in completed_translations:
                if completed['project_id'] == project_id:
                    project_data = completed
                    break
    
    if not project_data or project_data.get('status') != 'completed':
        return jsonify({'error': 'Translation not completed or not found'}), 404
    
    zip_file = project_data.get('zip_file')
    if not zip_file or not os.path.exists(zip_file):
        return jsonify({'error': 'Download file not found'}), 404
    
    original_name = project_data['filename']
    mode = project_data.get('mode', 'batches')
    
    if mode == 'batches':
        download_name = f"translated_batches_{original_name}.zip"
    else:
        download_name = f"translated_{original_name}.zip"
    
    return send_file(
        zip_file,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/zip'
    )

@app.route('/api/cancel/<project_id>', methods=['POST'])
def cancel_translation(project_id):
    """Cancel an ongoing translation"""
    with cleanup_lock:
        if project_id in active_translations:
            active_translations[project_id]['status'] = 'cancelled'
            
            # Try to clean up files
            try:
                project_folder = active_translations[project_id].get('project_folder')
                if project_folder and os.path.exists(project_folder):
                    shutil.rmtree(project_folder, ignore_errors=True)
            except:
                pass
            
            return jsonify({'success': True, 'message': 'Translation cancelled'})
    
    return jsonify({'error': 'Project not found'}), 404

@app.route('/api/stats', methods=['GET'])
def get_server_stats():
    """Get server statistics"""
    with cleanup_lock:
        active_count = len([p for p in active_translations.values() if p.get('status') in ['processing', 'splitting', 'translating']])
        completed_count = len(completed_translations)
    
    memory_mb = get_memory_usage()
    disk_usage = shutil.disk_usage(BASE_OUTPUT)
    
    return jsonify({
        'active_translations': active_count,
        'recently_completed': completed_count,
        'memory_usage_mb': round(memory_mb, 2),
        'disk_free_gb': round(disk_usage.free / 1024 / 1024 / 1024, 2),
        'max_file_size_mb': MAX_FILE_SIZE // 1024 // 1024,
        'supported_languages': len(SUPPORTED_LANGUAGES),
        'translation_engine': TRANSLATOR_ENGINE,
        'max_parallel_batches': MAX_PARALLEL_BATCHES
    })

@app.route('/api/languages', methods=['GET'])
def get_supported_languages():
    """Get supported languages"""
    return jsonify(SUPPORTED_LANGUAGES)

@app.route('/api/history', methods=['GET'])
def get_recent_history():
    """Get recent translation history"""
    recent = []
    cutoff = time.time() - (24 * 60 * 60)  # Last 24 hours
    
    with cleanup_lock:
        for project in list(completed_translations)[-20:]:  # Last 20
            if project.get('completion_time', 0) > cutoff:
                recent.append({
                    'project_id': project['project_id'],
                    'filename': project['filename'],
                    'language': project['target_lang'],
                    'file_size_mb': project['file_size_mb'],
                    'estimated_words': project.get('estimated_words', 0),
                    'completion_time': project.get('completion_time'),
                    'mode': project.get('mode', 'batches')
                })
    
    return jsonify({'recent_translations': recent})

# ---------------- CLEANUP ----------------
def cleanup_old_files():
    """Clean up old project files periodically"""
    while True:
        try:
            now = time.time()
            cutoff = now - PROJECT_EXPIRY
            
            # Clean up completed translations from memory
            with cleanup_lock:
                # Remove old completed entries
                while completed_translations and completed_translations[0].get('completion_time', 0) < cutoff:
                    completed_translations.popleft()
                
                # Clean up old active entries
                expired_projects = []
                for pid, project in list(active_translations.items()):
                    if project.get('start_time', 0) < cutoff and project.get('status') in ['completed', 'error', 'cancelled']:
                        expired_projects.append(pid)
                
                for pid in expired_projects:
                    # Try to delete files
                    try:
                        project_folder = active_translations[pid].get('project_folder')
                        if project_folder and os.path.exists(project_folder):
                            shutil.rmtree(project_folder, ignore_errors=True)
                        
                        zip_file = active_translations[pid].get('zip_file')
                        if zip_file and os.path.exists(zip_file):
                            os.remove(zip_file)
                    except:
                        pass
                    
                    del active_translations[pid]
            
            # Clean up old files in output directory
            for item in BASE_OUTPUT.iterdir():
                if item.is_file() or item.is_dir():
                    try:
                        mtime = item.stat().st_mtime
                        if mtime < cutoff:
                            if item.is_dir():
                                shutil.rmtree(item, ignore_errors=True)
                            else:
                                item.unlink()
                    except:
                        pass
            
            time.sleep(300)  # Run every 5 minutes
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            time.sleep(600)

# ---------------- HEALTH CHECK ----------------
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    memory_mb = get_memory_usage()
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'translation_engine': TRANSLATOR_ENGINE,
        'memory_usage_mb': round(memory_mb, 2),
        'active_projects': len(active_translations),
        'disk_space_gb': round(shutil.disk_usage('.').free / 1024 / 1024 / 1024, 2)
    })

# ---------------- MAIN ----------------
if __name__ == '__main__':
    # Check for translation engine
    if TRANSLATOR_ENGINE is None:
        logger.error("""
        ERROR: No translation library found!
        Please install: pip install deep-translator
        """)
        exit(1)
    
    logger.info(f"Starting Large File Translator")
    logger.info(f"Translation engine: {TRANSLATOR_ENGINE}")
    logger.info(f"Max file size: {MAX_FILE_SIZE // 1024 // 1024}MB")
    logger.info(f"Supported languages: {len(SUPPORTED_LANGUAGES)}")
    logger.info(f"Output directory: {BASE_OUTPUT}")
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_files, daemon=True)
    cleanup_thread.start()
    
    # Run app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
