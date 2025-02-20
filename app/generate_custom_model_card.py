import random
import yaml

from format_quant_type import format_quant_type

def generate_custom_model_card(model_id, base_model_name, quant_method, username, save_folder, license="mit", datasets=None):
    """
    Generate a custom model card with a randomly selected image and audio file.
    """
    # Format the quantization type (e.g., 'i1-GGUF' without specific methods like IQ2_XXS)
    formatted_qtype = format_quant_type(quant_method)

    # Prepare metadata for the model card
    custom_metadata = {
        "quantized_by": "SpongeQuant",
        "base_model": model_id,
        "language": ["en"],
        "license": license,
        "tags": ["SpongeQuant", formatted_qtype],
    }
    if datasets and isinstance(datasets, list) and any(datasets):
        custom_metadata["datasets"] = datasets

    # Define the list of images with associated captions.
    images = [
        {"file": "013.png", "caption": "Egypt, Red Sea, Sinai Peninsula and the Nile"},
        {"file": "044.png", "caption": "Sand dunes"},
        {"file": "045.png", "caption": "Monument Valley"},
        {"file": "046.png", "caption": "Forest scene with mushrooms"},
        {"file": "047.png", "caption": "Leaf"},
        {"file": "048.png", "caption": "Autumn Fallen leaves"},
        {"file": "049.png", "caption": "Snowflakes over Sequoia"},
        {"file": "050.png", "caption": "Tree with daffodils"},
        {"file": "051.png", "caption": "Flying insect with flowers"},
        {"file": "055.png", "caption": "School of fish"},
        {"file": "058.png", "caption": "Eagle"},
        {"file": "070.png", "caption": "Mountain climber"},
        {"file": "071.png", "caption": "Gymnast"},
        {"file": "072.png", "caption": "Sprinters (Valeriy Borzov of the U.S.S.R. in lead)"},
        {"file": "078.png", "caption": "Underwater scene with diver and fish"},
        {"file": "083.png", "caption": "Great Wall of China"},
        {"file": "091.png", "caption": "English city (Oxford)"},
        {"file": "092.png", "caption": "Boston"},
        {"file": "093.png", "caption": "UN Building Day"},
        {"file": "094.png", "caption": "UN Building Night"},
        {"file": "095.png", "caption": "Sydney Opera House"},
        {"file": "097.png", "caption": "Factory interior"},
        {"file": "098.png", "caption": "Museum"},
        {"file": "099.png", "caption": "X-ray of hand"},
        {"file": "101.png", "caption": "Street scene, Asia (Pakistan)"},
        {"file": "103.png", "caption": "Modern highway (Ithaca, NY)"},
        {"file": "104.png", "caption": "Golden Gate Bridge"},
        {"file": "105.png", "caption": "Train"},
        {"file": "106.png", "caption": "Airplane in flight"},
        {"file": "107.png", "caption": "Airport (Toronto)"},
        {"file": "108.png", "caption": "Antarctic Expedition"},
        {"file": "109.png", "caption": "Radio telescope (Westerbork, Netherlands)"},
        {"file": "110.png", "caption": "Radio telescope (Arecibo)"},
        {"file": "112.png", "caption": "Astronaut in space"},
        {"file": "113.png", "caption": "Titan Centaur launch"},
        {"file": "114.png", "caption": "Sunset with birds"},
    ]
    
    # Define the list of 31 audio files (001.mp3 to 031.mp3) with associated captions.
    audios = [
      {"file": "001.mp3", "caption": "Flawed Mangoes - Dramamine (USA, 2024)"},
      {"file": "002.mp3", "caption": "Iggy Pop – The Passenger (USA, 1977)"},
      {"file": "003.mp3", "caption": "Queen & David Bowie – Under Pressure (UK, 1981)"},
      {"file": "004.mp3", "caption": "Frank Sinatra – My Way (USA, 1969)"},
      {"file": "005.mp3", "caption": "Vangelis – Conquest of Paradise (Greece, 1992)"},
      {"file": "006.mp3", "caption": "Journey – Don't Stop Believin' (USA, 1981)"},
      {"file": "007.mp3", "caption": "Blur – Song 2 (UK, 1997)"},
      {"file": "008.mp3", "caption": "Pixies – Where Is My Mind (USA, 1988)"},
      {"file": "009.mp3", "caption": "M83 – Midnight City (France, 2011)"},
      {"file": "010.mp3", "caption": "El Cascabel – Antonio Maciel and Los Aguilillas with Mariachi México de Pepe Villa / Rafael Carrión (Mexico, Unknown)"},
      {"file": "011.mp3", "caption": "Chuck Berry – Johnny B. Goode (USA, 1958)"},
      {"file": "012.mp3", "caption": "NENA – 99 Luftballons (Germany, 1983)"}
    ]
    
    # Randomly select one image and one audio.
    selected_image = random.choice(images)
    selected_audio = random.choice(audios)
    
    # Build the custom content with the selected image and audio.
    custom_content = f"""
Quantized to `{formatted_qtype}` using [SpongeQuant](https://github.com/SpongeEngine/SpongeQuant), the Oobabooga of LLM quantization.

<div style="display: flex; gap: 20px; align-items: center; margin-top:0; ">
  <a href="https://github.com/SpongeEngine/SpongeQuant">
    <img src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/github-button.png".png" width="173">
  </a>
  <a href="https://discord.gg/azNmr2Gdgy">
    <img src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/discord-button.png".png" width="173">
  </a>
</div>

***
<figure>
  <img src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/{selected_image['file']}" alt="{selected_image['caption']}">
  <figcaption>{selected_image['caption']}</figcaption>
</figure>

<figure>
  <audio controls>
    <source src="https://huggingface.co/spaces/SpongeEngine/README/resolve/main/{selected_audio['file']}" type="audio/mp3">
    Your browser does not support the audio element.
  </audio>
  <figcaption>{selected_audio['caption']}</figcaption>
</figure>

***
{f"""
### What is a GGUF?
GGUF is a file format used for running large language models (LLMs) on different types of computers. It supports both regular processors (CPUs) and graphics cards (GPUs), making it easier to run models across a wide range of hardware. Many LLMs require powerful and expensive GPUs, but GGUF improves compatibility and efficiency by optimizing how models are loaded and executed. If a GPU doesn’t have enough memory, GGUF can offload parts of the model to the CPU, allowing it to run even when GPU resources are limited. GGUF is designed to work well with quantized models, which use less memory and run faster, making them ideal for lower-end hardware. However, it can also store full-precision models when needed. Thanks to these optimizations, GGUF allows LLMs to run efficiently on everything from high-end GPUs to laptops and even CPU-only systems.
""" if "GGUF" in formatted_qtype else ""}
{f"""
### What is an i1-GGUF?
i1-GGUF is an enhanced type of GGUF model that uses imatrix quantization—a smarter way of reducing model size while preserving key details. Instead of shrinking everything equally, it analyzes the importance of different model components and keeps the most crucial parts more accurate. Like standard GGUF, i1-GGUF allows LLMs to run on various hardware, including CPUs and lower-end GPUs. However, because it prioritizes important weights, i1-GGUF models deliver better responses than traditional GGUF models while maintaining efficiency.
""" if formatted_qtype == "i1-GGUF" else ""}
"""
    # Convert custom metadata into YAML format
    merged_yaml = yaml.dump(custom_metadata, default_flow_style=False)
    
    # Build the full model card by merging metadata and custom content
    full_card = f"---\n{merged_yaml}---\n\n{custom_content}"
    
    return full_card