"""
Orpheus TTS Nodes for ComfyUI
"""
import os
import re
import numpy as np
import torch
from pathlib import Path
import folder_paths
import soundfile as sf

# Import custom SNAC and other required libraries
# Note: These would need to be properly installed in the ComfyUI environment
try:
    from snac import SNAC
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    import nltk
    from nltk.tokenize import sent_tokenize
    
    # Download nltk data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install the required dependencies for Orpheus TTS")
    IMPORTS_SUCCESSFUL = False

# Constants for chunking
MAX_CHARS = 220  # Maximum characters per chunk

# List of paralinguistic elements
ELEMENTS = [
    "none", "laugh", "chuckle", "sigh", "cough", "sniffle", 
    "groan", "yawn", "gasp"
]

# Available voices
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe", "bob", "rebeca", "lisa"]


# Element positions
ELEMENT_POSITIONS = ["none", "append", "prepend"]

class OrpheusModelLoader:
    """
    ComfyUI Node for loading and initializing Orpheus TTS models
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "snac_model_path": ("STRING", {"default": "hubertsiuzdak/snac_24khz"}),
                "orpheus_model_path": ("STRING", {"default": "canopylabs/orpheus-3b-0.1-ft"}),
            }
        }
    
    RETURN_TYPES = ("ORPHEUS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_models"
    CATEGORY = "audio/tts"
    
    def load_models(self, snac_model_path="hubertsiuzdak/snac_24khz", 
                   orpheus_model_path="canopylabs/orpheus-3b-0.1-ft"):
        if not IMPORTS_SUCCESSFUL:
            raise ImportError("Required libraries are not installed. Please install the required dependencies.")
        
        print("Loading Orpheus TTS models...")
        
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load SNAC model
        print("Loading SNAC model...")
        snac_model = SNAC.from_pretrained(snac_model_path)
        snac_model = snac_model.to(device)
        
        # Download only model config and safetensors if using Hugging Face model
        if not os.path.isdir(orpheus_model_path):
            print(f"Downloading Orpheus model: {orpheus_model_path}")
            snapshot_download(
                repo_id=orpheus_model_path,
                allow_patterns=[
                    "config.json",
                    "*.safetensors",
                    "model.safetensors.index.json",
                ],
                ignore_patterns=[
                    "optimizer.pt",
                    "pytorch_model.bin",
                    "training_args.bin",
                    "scheduler.pt",
                    "tokenizer.json",
                    "tokenizer_config.json", 
                    "special_tokens_map.json",
                    "vocab.json",
                    "merges.txt",
                    "tokenizer.*"
                ]
            )
        
        # Load Orpheus model and tokenizer
        print("Loading Orpheus model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(orpheus_model_path, torch_dtype=torch.bfloat16)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(orpheus_model_path)
        
        print("Models loaded successfully")
        
        # Return models and device as a tuple
        return ((snac_model, model, tokenizer, device),)


class OrpheusGenerate:
    """
    ComfyUI Node for generating speech using Orpheus TTS
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ORPHEUS_MODEL",),
                "text": ("STRING", {"multiline": True}),
                "voice": (VOICES, {"default": "tara"}),
                "element": (ELEMENTS, {"default": "none"}),
                "element_position": (ELEMENT_POSITIONS, {"default": "none"}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 2700, "min": 100, "max": 4000, "step": 100}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/tts"
    
    # Define the audio output structure expected by ComfyUI
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")  # Always process
    
    def generate_speech(self, model, text, voice, element, element_position, 
                       temperature, top_p, repetition_penalty, max_new_tokens):
        """Generate speech from text using Orpheus TTS"""
        if not text.strip():
            return (None,)
        
        # Unpack the model tuple
        snac_model, orpheus_model, tokenizer, device = model
        
        # Apply selected element if needed
        if element != "none" and element_position != "none":
            element_tag = f"<{element}>"
            if element_position == "append":
                text = f"{text} {element_tag}"
            elif element_position == "prepend":
                text = f"{element_tag} {text}"
        
        try:
            # Check if the text is long and needs chunking
            if len(text) <= MAX_CHARS:
                # Short text - process normally
                input_ids, attention_mask = self.process_prompt(text, voice, tokenizer, device)
                
                with torch.no_grad():
                    generated_ids = orpheus_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        num_return_sequences=1,
                        eos_token_id=128258,
                    )
                
                code_list = self.parse_output(generated_ids)
                audio_samples = self.redistribute_codes(code_list, snac_model)
                
                # Convert to the exact format expected by ComfyUI
                # Format needs to be [batch, channels, samples]
                audio_samples_float = audio_samples.astype(np.float32)
                waveform = torch.tensor(audio_samples_float)
                
                # Ensure we have a channel dimension
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
                
                # Add batch dimension 
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                
                print(f"Final audio tensor shape: {waveform.shape}, 24000Hz")
                
                return ({"waveform": waveform, "sample_rate": 24000},)
            else:
                # Long text - needs chunking
                chunks = self.split_into_chunks(text, MAX_CHARS)
                
                # Create a message for the user
                chunk_message = f"Processing text in {len(chunks)} chunks for better quality..."
                print(chunk_message)
                
                all_audio_samples = []
                chunk_count = len(chunks)
                
                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i+1}/{chunk_count}: {chunk[:30]}{'...' if len(chunk) > 30 else ''}")
                    
                    # Process this chunk
                    chunk_audio = self.process_audio_chunk(
                        chunk, voice, snac_model, orpheus_model, tokenizer, device,
                        temperature, top_p, repetition_penalty, max_new_tokens
                    )
                    
                    all_audio_samples.append(chunk_audio)
                
                combined_audio = self.combine_audio_files(all_audio_samples)
                
                # Convert to the exact format expected by ComfyUI
                # Format needs to be [batch, channels, samples]
                combined_audio_float = combined_audio.astype(np.float32)
                waveform = torch.tensor(combined_audio_float)
                
                # Ensure we have a channel dimension
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)  # [samples] -> [1, samples]
                
                # Add batch dimension 
                waveform = waveform.unsqueeze(0)  # [channels, samples] -> [1, channels, samples]
                
                print(f"Final audio tensor shape: {waveform.shape}, 24000Hz")
                
                return ({"waveform": waveform, "sample_rate": 24000},)
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty audio on failure (1 second of silence at our sample rate)
            dummy_waveform = torch.zeros(1, 1, 24000)  # [batch, channels, samples]
            return ({"waveform": dummy_waveform, "sample_rate": 24000},)
    
    def process_prompt(self, prompt, voice, tokenizer, device):
        """Process text prompt for the model"""
        prompt = f"{voice}: {prompt}"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        
        start_token = torch.tensor([[128259]], dtype=torch.int64)  # Start of human
        end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)  # End of text, End of human
        
        modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)  # SOH SOT Text EOT EOH
        
        # No padding needed for single input
        attention_mask = torch.ones_like(modified_input_ids)
        
        return modified_input_ids.to(device), attention_mask.to(device)
    
    def parse_output(self, generated_ids):
        """Parse output tokens to audio codes"""
        token_to_find = 128257
        token_to_remove = 128258
        
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_occurrence_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        else:
            cropped_tensor = generated_ids

        processed_rows = []
        for row in cropped_tensor:
            masked_row = row[row != token_to_remove]
            processed_rows.append(masked_row)

        code_lists = []
        for row in processed_rows:
            row_length = row.size(0)
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)
            
        return code_lists[0]  # Return just the first one for single sample
    
    def redistribute_codes(self, code_list, snac_model):
        """Redistribute codes for audio generation"""
        device = next(snac_model.parameters()).device  # Get the device of SNAC model
        
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
            
        # Move tensors to the same device as the SNAC model
        codes = [
            torch.tensor(layer_1, device=device).unsqueeze(0),
            torch.tensor(layer_2, device=device).unsqueeze(0),
            torch.tensor(layer_3, device=device).unsqueeze(0)
        ]
        
        audio_hat = snac_model.decode(codes)
        return audio_hat.detach().squeeze().cpu().numpy()  # Always return CPU numpy array
    
    def split_into_chunks(self, text, max_chars=MAX_CHARS):
        """
        Split text into chunks of reasonable size, trying to break at sentence boundaries.
        Returns a list of text chunks, each smaller than max_chars.
        """
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed the limit and we already have content, start a new chunk
            if len(current_chunk) + len(sentence) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            # If the sentence alone is longer than max_chars, we need to split it further
            elif len(sentence) > max_chars:
                # Add current chunk if not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split long sentence into smaller parts (at commas, or just by characters if needed)
                parts = re.split(r'(?<=,)\s+', sentence)
                sub_chunk = ""
                
                for part in parts:
                    if len(sub_chunk) + len(part) > max_chars:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                            sub_chunk = part
                        else:
                            # Even a single part is too long, so just split it arbitrarily
                            for i in range(0, len(part), max_chars):
                                chunks.append(part[i:i+max_chars].strip())
                    else:
                        sub_chunk += " " + part if sub_chunk else part
                
                # Add any remaining sub-chunk
                if sub_chunk:
                    current_chunk = sub_chunk
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def combine_audio_files(self, audio_data_list):
        """Combine multiple audio arrays into a single audio array."""
        return np.concatenate(audio_data_list)
    
    def process_audio_chunk(self, chunk_text, voice, snac_model, orpheus_model, tokenizer, device,
                          temperature, top_p, repetition_penalty, max_new_tokens):
        """Generate and process audio for a single text chunk"""
        input_ids, attention_mask = self.process_prompt(chunk_text, voice, tokenizer, device)
        
        with torch.no_grad():
            generated_ids = orpheus_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=min(max_new_tokens, 2000),  # Cap at 2000 tokens per chunk
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                eos_token_id=128258,
            )
        
        code_list = self.parse_output(generated_ids)
        chunk_audio = self.redistribute_codes(code_list, snac_model)
        
        return chunk_audio





# Register nodes
NODE_CLASS_MAPPINGS = {
    "OrpheusModelLoader": OrpheusModelLoader,
    "OrpheusGenerate": OrpheusGenerate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OrpheusModelLoader": "Orpheus TTS Model Loader",
    "OrpheusGenerate": "Orpheus TTS Generate"
}