# ComfyUI-Orpheus-TTS
This extension adds high-quality Text-to-Speech capabilities to ComfyUI using the Orpheus TTS model. Create natural-sounding voices with emotional expressions, multilingual support, and audio effects.

![Orpheus TTS Banner](https://github.com/yourusername/ComfyUI-Orpheus-TTS/raw/main/images/banner.jpg)

## Features

- 🎙️ High-quality, natural-sounding speech synthesis
- 🎭 Support for emotional expressions and paralinguistic elements
- 👥 Multiple voice options (tara, leah, jess, leo, dan, mia, zac, zoe, etc.)
- 📝 Long text handling with automatic chunking for consistent output
- 🎛️ Professional audio effects:
  - Pitch shifting (-12 to +12 semitones)
  - Speed adjustment (0.5x to 2.0x speed)
- 🌐 Optional support for private Hugging Face models

## Installation

### 1. Install the Extension

Clone this repository into your ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ShmuelRonen/ComfyUI-Orpheus-TTS.git
```

### 2. Install Required Python Dependencies

```bash
pip install torch numpy soundfile transformers huggingface_hub nltk snac
```

### 3. Install SoX (Required for Audio Effects)

#### Windows

1. Download SoX for Windows from the [official SourceForge page](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
   - Download the `.exe` installer (e.g., `sox-14.4.2-win32.exe`)

2. Run the installer:
   - Follow the installation prompts
   - **Important**: Note the installation directory (default is usually `C:\Program Files (x86)\sox-14-4-2\`)
   
3. No need to add to PATH - the extension uses the direct path to SoX

#### WSL 2 (Ubuntu)

```bash
sudo apt-get update
sudo apt-get install sox
```

#### macOS

```bash
brew install sox
```

### 4. Restart ComfyUI

After installing all required components, restart ComfyUI to load the extension.

## Configuration

### Hugging Face Authentication (Optional)

To access private models on Hugging Face, create a file named `hf_config.json` in the extension directory and insert your HF Token KEY:

   ```json
   {
     "token": "YOUR_HUGGING_FACE_TOKEN_HERE"
   }
   ```

3. Save the file and restart ComfyUI

Your token will be used to authenticate with Hugging Face when downloading models. This is only required if you're using private models or if you need higher rate limits.

## Nodes

### Orpheus TTS Model Loader

Loads the required models for Orpheus TTS.

**Inputs:**
- `snac_model_path` (optional): Path to SNAC model (default: "hubertsiuzdak/snac_24khz")
- `orpheus_model_path` (optional): Path to Orpheus model (default: "canopylabs/orpheus-3b-0.1-ft")

**Outputs:**
- `model`: Model reference to be passed to the generate node

### Orpheus TTS Generate

Generates speech from text input.

**Inputs:**
- `model`: Model reference from the loader node
- `text`: Text to convert to speech
- `voice`: Voice selection (tara, leah, jess, leo, dan, mia, zac, zoe, etc.)
- `element`: Paralinguistic element to insert (laugh, chuckle, sigh, etc.)
- `element_position`: Where to insert the element (none, append, prepend)
- `temperature`: Controls randomness (0.1-1.5)
- `top_p`: Controls diversity (0.1-1.0)
- `repetition_penalty`: Penalizes repetition (1.0-2.0)
- `max_new_tokens`: Maximum generation length (100-4000)

**Outputs:**
- `audio`: Generated audio that can be played, saved, or further processed

### Orpheus Audio Effects

Applies pitch shifting and speed adjustment to audio using SoX.

**Inputs:**
- `audio`: Audio input from TTS Generate node
- `pitch_shift`: Semitone adjustment (-12 to +12)
- `speed_factor`: Playback speed (0.5 to 2.0)
- `sox_path` (optional): Path to SoX executable (default works for typical Windows installations)

**Outputs:**
- `audio`: Processed audio with effects applied

## Usage Examples

### Basic Text-to-Speech

1. Add "Orpheus TTS Model Loader"
2. Add "Orpheus TTS Generate"
3. Connect the model loader's output to the generate node's input
4. Enter your text and select voice options
5. Connect to "Preview Audio" node to hear the result

![Basic TTS Workflow](https://github.com/yourusername/ComfyUI-Orpheus-TTS/raw/main/images/basic_workflow.jpg)

### Advanced: TTS with Audio Effects

1. Add "Orpheus TTS Model Loader"
2. Add "Orpheus TTS Generate"
3. Add "Orpheus Audio Effects"
4. Connect in sequence: Model Loader → Generate → Audio Effects → Preview Audio
5. Adjust pitch shift and speed factor sliders

![Advanced TTS Workflow](https://github.com/yourusername/ComfyUI-Orpheus-TTS/raw/main/images/advanced_workflow.jpg)

## Paralinguistic Elements

You can add expressive elements to the speech by inserting these tags:

|
 Element 
|
 Tag 
|
 Description 
|
|
---------
|
-----
|
-------------
|
|
 Laugh 
|
`<laugh>`
|
 Natural laughter 
|
|
 Chuckle 
|
`<chuckle>`
|
 Light, subtle laughter 
|
|
 Sigh 
|
`<sigh>`
|
 Exhaling with emotion 
|
|
 Cough 
|
`<cough>`
|
 Clearing throat 
|
|
 Sniffle 
|
`<sniffle>`
|
 Subtle nasal sound 
|
|
 Groan 
|
`<groan>`
|
 Low, grumbling sound 
|
|
 Yawn 
|
`<yawn>`
|
 Tired exhale 
|
|
 Gasp 
|
`<gasp>`
|
 Sudden intake of breath 
|

### Example:
```
I can't believe it! <laugh> That's the funniest thing I've heard all day.
```

## SoX Troubleshooting

### Windows

If you encounter issues with SoX:

1. Verify the SoX path in the "Orpheus Audio Effects" node:
   - Default: `C:\Program Files (x86)\sox-14-4-2\sox.exe`
   - If your installation is in a different location, provide the full path to sox.exe

2. Check if SoX is installed correctly:
   - Open Command Prompt
   - Run `"C:\Program Files (x86)\sox-14-4-2\sox.exe" --version`
   - If you get an error, reinstall SoX

### WSL 2 (Ubuntu)

1. Verify SoX installation:
   ```bash
   sox --version
   ```

2. If SoX is not found, install it:
   ```bash
   sudo apt-get update
   sudo apt-get install sox
   ```

## Model Details

This extension uses the following models:

- **Orpheus TTS**: A powerful text-to-speech model developed by Canopy AI
- **SNAC Codec**: A high-quality neural audio codec for voice synthesis

## License

This project uses models with their own licenses:
- Orpheus Model: [Canopy AI's Orpheus-TTS](https://github.com/canopyai/Orpheus-TTS)
- SNAC Model: [Hubert Siuzdak's model](https://huggingface.co/hubertsiuzdak/snac_24khz)

Please consult these licenses for usage terms and restrictions.

## Credits

- Original Orpheus TTS implementation by [Canopy AI](https://github.com/canopyai/Orpheus-TTS)
- SoX audio processing library: [SoX - Sound eXchange](http://sox.sourceforge.net/)
- ComfyUI: [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
