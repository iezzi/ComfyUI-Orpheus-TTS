"""
Direct Path SoX Audio Effects Node for Orpheus TTS in ComfyUI
Uses explicit path to SoX executable for Windows compatibility
"""

import os
import tempfile
import numpy as np
import torch
import subprocess
import soundfile as sf
import shutil

class OrpheusAudioEffects:
    """
    ComfyUI Node for applying audio effects (pitch shift and speed adjustment) to TTS output
    Uses explicit SoX path for Windows compatibility
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "pitch_shift": ("FLOAT", {"default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5, 
                              "display": "slider", "label": "Pitch Shift (semitones)"}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, 
                               "display": "slider", "label": "Speed Factor"})
            },
            "optional": {
                "sox_path": ("STRING", {"default": "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"})
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process_audio"
    CATEGORY = "audio/effects"
    
    def process_audio(self, audio, pitch_shift=0.0, speed_factor=1.0, sox_path="C:\\Program Files (x86)\\sox-14-4-2\\sox.exe"):
        """Apply pitch shifting and speed adjustment to audio using direct SoX path"""
        if audio is None:
            print("No audio data to process")
            return (None,)
        
        # Check if no effects are needed
        if abs(pitch_shift) < 0.01 and abs(speed_factor - 1.0) < 0.01:
            print("No effects to apply, returning original audio")
            return (audio,)
        
        # Check if the SoX executable exists
        if not os.path.exists(sox_path):
            print(f"SoX executable not found at: {sox_path}")
            print("Please provide the correct path to the sox.exe file")
            
            # Try some common locations
            common_paths = [
                "C:\\Program Files (x86)\\sox-14-4-2\\sox.exe",
                "C:\\Program Files\\sox-14-4-2\\sox.exe",
                "C:\\Program Files (x86)\\sox-14.4.2\\sox.exe",
                "C:\\Program Files\\sox-14.4.2\\sox.exe",
                "C:\\Program Files (x86)\\sox\\sox.exe",
                "C:\\Program Files\\sox\\sox.exe"
            ]
            
            for path in common_paths:
                if os.path.exists(path):
                    print(f"Found SoX at: {path}")
                    sox_path = path
                    break
            else:
                print("SoX not found in any common location. Please install SoX or provide the correct path.")
                return (audio,)
        
        try:
            # Extract audio data and sample rate
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to numpy for processing
            # If waveform is [batch, channels, samples]
            if waveform.dim() == 3:
                audio_np = waveform[0, 0].cpu().numpy()
            # If waveform is [channels, samples]
            elif waveform.dim() == 2:
                audio_np = waveform[0].cpu().numpy()
            else:
                audio_np = waveform.cpu().numpy()
            
            print(f"Processing audio with SoX at: {sox_path}")
            print(f"- Pitch shift: {pitch_shift} semitones")
            print(f"- Speed factor: {speed_factor}x")
            
            # Create a temporary directory for this operation
            temp_dir = tempfile.mkdtemp(prefix="orpheus_sox_")
            
            try:
                # Create input and output file paths
                input_path = os.path.join(temp_dir, "input.wav")
                output_path = os.path.join(temp_dir, "output.wav")
                
                print(f"Writing input audio to {input_path}")
                sf.write(input_path, audio_np, sample_rate)
                
                # Verify the input file was written successfully
                if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
                    print("Failed to write input audio file")
                    return (audio,)
                
                # Create the SoX command with explicit parameters
                sox_cmd = [sox_path, input_path, output_path]
                
                if pitch_shift != 0:
                    # Convert semitones to cents
                    pitch_cents = int(pitch_shift * 100)
                    sox_cmd.extend(['pitch', str(pitch_cents)])
                    print(f"Added pitch effect: {pitch_cents} cents")
                
                if speed_factor != 1.0:
                    sox_cmd.extend(['tempo', '-s', str(speed_factor)])
                    print(f"Added tempo effect: {speed_factor}x")
                
                # Print the full SoX command
                print("Executing:", " ".join(sox_cmd))
                
                # Run SoX command and capture output
                process = subprocess.run(
                    sox_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Print SoX output for debugging
                if process.stdout:
                    print("SoX stdout:", process.stdout)
                if process.stderr:
                    print("SoX stderr:", process.stderr)
                
                # Check if SoX command succeeded
                if process.returncode != 0:
                    print(f"SoX command failed with return code {process.returncode}")
                    return (audio,)
                
                # Check if output file exists and has content
                if not os.path.exists(output_path):
                    print("Output file does not exist")
                    return (audio,)
                
                if os.path.getsize(output_path) == 0:
                    print("Output file is empty")
                    return (audio,)
                
                print(f"Reading processed audio from {output_path}")
                
                # Read processed audio
                processed_audio, new_sample_rate = sf.read(output_path)
                
                # Convert to tensor with proper dimensions
                processed_tensor = torch.tensor(processed_audio.astype(np.float32))
                
                # Ensure proper dimensions [batch, channels, samples]
                if processed_tensor.dim() == 1:
                    processed_tensor = processed_tensor.unsqueeze(0)  # Add channel dim
                
                if processed_tensor.dim() == 2:
                    processed_tensor = processed_tensor.unsqueeze(0)  # Add batch dim
                
                print(f"Processed audio shape: {processed_tensor.shape}")
                
                # Create result
                result_audio = {
                    "waveform": processed_tensor,
                    "sample_rate": new_sample_rate
                }
                
                return (result_audio,)
                
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    print(f"Error cleaning up temporary files: {e}")
                
        except Exception as e:
            print(f"Error in audio processing: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)

# Register nodes
NODE_CLASS_MAPPINGS = {
    "OrpheusAudioEffects": OrpheusAudioEffects
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OrpheusAudioEffects": "Orpheus Audio Effects"
}