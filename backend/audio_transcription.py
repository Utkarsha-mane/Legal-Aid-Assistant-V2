from faster_whisper import WhisperModel
import os
from pathlib import Path
from typing import Dict, Any
import tempfile
from pydub import AudioSegment


class AudioTranscriber:
    """
    Handles audio file transcription using Faster-Whisper (free, open-source).
    Supports multiple audio formats and converts them to text.
    Faster and more efficient than original OpenAI Whisper.
    """
    
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        """
        Initialize Faster-Whisper transcriber.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2')
                       - tiny: fastest, least accurate
                       - base: good balance (recommended)
                       - small/medium/large-v2: more accurate but slower
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        self.model_size = model_size
        self.device = device
        self.model = None
        self.supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma']
        
    def load_model(self):
        """Load Faster-Whisper model (lazy loading to save memory)."""
        if self.model is None:
            print(f"Loading Faster-Whisper {self.model_size} model...")
            # compute_type: "int8" for CPU (faster), "float16" for GPU
            compute_type = "int8" if self.device == "cpu" else "float16"
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type=compute_type
            )
            print("✓ Faster-Whisper model loaded successfully")
    
    def convert_to_wav(self, audio_path: Path) -> Path:
        """
        Convert audio file to WAV format if needed.
        
        Args:
            audio_path: Path to input audio file
            
        Returns:
            Path to WAV file
        """
        file_ext = audio_path.suffix.lower()
        
        # If already WAV, return as is
        if file_ext == '.wav':
            return audio_path
        
        print(f"Converting {file_ext} to WAV format...")
        
        # Load audio file
        audio = AudioSegment.from_file(str(audio_path))
        
        # Create temporary WAV file
        temp_wav = audio_path.with_suffix('.wav')
        audio.export(str(temp_wav), format='wav')
        
        print(f"✓ Converted to WAV: {temp_wav.name}")
        return temp_wav
    
    def transcribe_audio(self, audio_path: Path, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text using Faster-Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es', 'hi')
                     If None, Whisper will auto-detect
            
        Returns:
            Dictionary with transcription results
        """
        try:
            # Validate file format
            file_ext = audio_path.suffix.lower()
            if file_ext not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Unsupported audio format: {file_ext}. Supported formats: {", ".join(self.supported_formats)}',
                    'text': ''
                }
            
            # Load Faster-Whisper model
            self.load_model()
            
            # Convert to WAV if needed
            converted_path = None
            try:
                if file_ext != '.wav':
                    converted_path = self.convert_to_wav(audio_path)
                    transcribe_path = converted_path
                else:
                    transcribe_path = audio_path
                
                print(f"Transcribing audio file: {audio_path.name}")
                
                # Transcribe with Faster-Whisper
                segments, info = self.model.transcribe(
                    str(transcribe_path),
                    language=language,
                    beam_size=5,
                    vad_filter=True  # Voice Activity Detection for better accuracy
                )
                
                # Collect all segments
                segment_list = []
                full_text = []
                
                for segment in segments:
                    segment_list.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip()
                    })
                    full_text.append(segment.text.strip())
                
                transcript_text = ' '.join(full_text)
                detected_language = info.language
                
                print(f"✓ Transcription complete: {len(transcript_text)} characters")
                print(f"✓ Detected language: {detected_language}")
                print(f"✓ Language probability: {info.language_probability:.2%}")
                
                # Calculate duration
                duration = segment_list[-1]['end'] if segment_list else 0
                
                return {
                    'success': True,
                    'text': transcript_text,
                    'language': detected_language,
                    'language_probability': info.language_probability,
                    'segments': segment_list,
                    'duration': duration
                }
                
            finally:
                # Clean up temporary WAV file
                if converted_path and converted_path.exists() and converted_path != audio_path:
                    try:
                        converted_path.unlink()
                        print(f"✓ Cleaned up temporary file: {converted_path.name}")
                    except:
                        pass
                        
        except Exception as e:
            return {
                'success': False,
                'error': f'Transcription failed: {str(e)}',
                'text': ''
            }
    
    def transcribe_with_timestamps(self, audio_path: Path, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio with timestamp information for each segment.
        Useful for creating time-stamped chunks.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Dictionary with timestamped transcription
        """
        result = self.transcribe_audio(audio_path, language)
        
        if not result['success']:
            return result
        
        # Format segments with timestamps
        timestamped_text = []
        for segment in result.get('segments', []):
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            timestamped_text.append(f"[{start_time} - {end_time}] {segment['text']}")
        
        result['timestamped_text'] = '\n'.join(timestamped_text)
        return result
    
    def _format_timestamp(self, seconds: float) -> str:
        """
        Format seconds to HH:MM:SS format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return self.supported_formats.copy()
    
    def validate_audio_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate if audio file is supported and readable.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Validation result dictionary
        """
        if not file_path.exists():
            return {
                'valid': False,
                'error': 'File does not exist'
            }
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            return {
                'valid': False,
                'error': f'Unsupported format: {file_ext}. Supported: {", ".join(self.supported_formats)}'
            }
        
        # Check file size (warn if > 100MB)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            return {
                'valid': True,
                'warning': f'Large file ({file_size_mb:.1f} MB) - transcription may take several minutes'
            }
        
        return {
            'valid': True,
            'size_mb': file_size_mb
        }