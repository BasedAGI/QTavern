import random
import yaml

from format_quant_type import format_quant_type

def generate_custom_model_card(model_id, base_model_name, quant_method, username, save_folder, license="mit", datasets=None):
    """
    Generate a custom model card with a randomly selected image and audio file.
    """
    formatted_qtype = format_quant_type(quant_method)
    custom_metadata = {
        "quantized_by": "SpongeQuant",
        "base_model": model_id,
        "language": ["en"],
        "license": license,
        "tags": ["SpongeQuant", formatted_qtype],
    }
    if datasets and isinstance(datasets, list) and any(datasets):
        custom_metadata["datasets"] = datasets

    # Define the list of 122 images (001.png to 122.png) with associated captions.
    images = [
        {"file": "001.png", "caption": "1. Calibration circle"},
        {"file": "002.png", "caption": "2. Solar location map"},
        {"file": "003.png", "caption": "3. Mathematical definitions"},
        {"file": "004.png", "caption": "4. Physical unit definitions"},
        {"file": "005.png", "caption": "5. Solar system parameters"},
        {"file": "006.png", "caption": "6. Solar system parameters"},
        {"file": "007.png", "caption": "7. The Sun"},
        {"file": "008.png", "caption": "8. Solar spectrum"},
        {"file": "009.png", "caption": "9. Mercury"},
        {"file": "010.png", "caption": "10. Mars"},
        {"file": "011.png", "caption": "11. Jupiter"},
        {"file": "012.png", "caption": "12. Earth"},
        {"file": "013.png", "caption": "13. Egypt, Red Sea, Sinai Peninsula and the Nile"},
        {"file": "014.png", "caption": "14. Chemical definitions"},
        {"file": "015.png", "caption": "15. DNA Structure"},
        {"file": "016.png", "caption": "16. DNA Structure magnified, light hit"},
        {"file": "017.png", "caption": "17. Cells and cell division"},
        {"file": "018.png", "caption": "18. Anatomy 1 (Skeleton front)"},
        {"file": "019.png", "caption": "19. Anatomy 2 (Internal organs front)"},
        {"file": "020.png", "caption": "20. Anatomy 3 (Skeleton and muscles back)"},
        {"file": "021.png", "caption": "21. Anatomy 4 (Internal organs back)"},
        {"file": "022.png", "caption": "22. Anatomy 5 (Ribcage)"},
        {"file": "023.png", "caption": "23. Anatomy 6 (Muscles front)"},
        {"file": "024.png", "caption": "24. Anatomy 7 (Heart, lungs, kidneys and main blood vessels back)"},
        {"file": "025.png", "caption": "25. Anatomy 8 (Heart, lungs, kidneys and main blood vessels front)"},
        {"file": "026.png", "caption": "26. Human sex organs"},
        {"file": "027.png", "caption": "27. Diagram of conception"},
        {"file": "028.png", "caption": "28. Conception"},
        {"file": "029.png", "caption": "29. Fertilized ovum"},
        {"file": "030.png", "caption": "30. Fetus diagram"},
        {"file": "031.png", "caption": "31. Fetus"},
        {"file": "032.png", "caption": "32. Diagram of male and female"},
        {"file": "033.png", "caption": "33. Birth"},
        {"file": "034.png", "caption": "34. Nursing mother"},
        {"file": "035.png", "caption": "35. Father and daughter (Malaysia)"},
        {"file": "036.png", "caption": "36. Group of children"},
        {"file": "037.png", "caption": "37. Diagram of family ages"},
        {"file": "038.png", "caption": "38. Family portrait"},
        {"file": "039.png", "caption": "39. Diagram of continental drift"},
        {"file": "040.png", "caption": "40. Structure of the Earth"},
        {"file": "041.png", "caption": "41. Heron Island (Great Barrier Reef of Australia)"},
        {"file": "042.png", "caption": "42. Seashore"},
        {"file": "043.png", "caption": "43. Snake River and Grand Tetons"},
        {"file": "044.png", "caption": "44. Sand dunes"},
        {"file": "045.png", "caption": "45. Monument Valley"},
        {"file": "046.png", "caption": "46. Forest scene with mushrooms"},
        {"file": "047.png", "caption": "47. Leaf"},
        {"file": "048.png", "caption": "48. Autumn Fallen leaves"},
        {"file": "049.png", "caption": "49. Snowflakes over Sequoia"},
        {"file": "050.png", "caption": "50. Tree with daffodils"},
        {"file": "051.png", "caption": "51. Flying insect with flowers"},
        {"file": "052.png", "caption": "52. Diagram of vertebrate evolution"},
        {"file": "053.png", "caption": "53. Seashell (Xancidae)"},
        {"file": "054.png", "caption": "54. Dolphins"},
        {"file": "055.png", "caption": "55. School of fish"},
        {"file": "056.png", "caption": "56. Tree toad"},
        {"file": "057.png", "caption": "57. Crocodile"},
        {"file": "058.png", "caption": "58. Eagle"},
        {"file": "059.png", "caption": "59. Waterhole"},
        {"file": "060.png", "caption": "60. Jane Goodall and chimps"},
        {"file": "061.png", "caption": "61. Sketch of bushmen"},
        {"file": "062.png", "caption": "62. Bushmen hunters"},
        {"file": "063.png", "caption": "63. Man from Guatemala"},
        {"file": "064.png", "caption": "64. Dancer from Bali"},
        {"file": "065.png", "caption": "65. Andean girls"},
        {"file": "066.png", "caption": "66. Thailand master craftsman"},
        {"file": "067.png", "caption": "67. Elephant"},
        {"file": "068.png", "caption": "68. Old man with beard and glasses (Turkey)"},
        {"file": "069.png", "caption": "69. Old man with dog and flowers"},
        {"file": "070.png", "caption": "70. Mountain climber"},
        {"file": "071.png", "caption": "71. Gymnast"},
        {"file": "072.png", "caption": "72. Sprinters (Valeriy Borzov of the U.S.S.R. in lead)"},
        {"file": "073.png", "caption": "73. Schoolroom"},
        {"file": "074.png", "caption": "74. Children with globe"},
        {"file": "075.png", "caption": "75. Cotton harvest"},
        {"file": "076.png", "caption": "76. Grape picker"},
        {"file": "077.png", "caption": "77. Supermarket"},
        {"file": "078.png", "caption": "78. Underwater scene with diver and fish"},
        {"file": "079.png", "caption": "79. Fishing boat with nets"},
        {"file": "080.png", "caption": "80. Cooking fish"},
        {"file": "081.png", "caption": "81. Chinese dinner party"},
        {"file": "082.png", "caption": "82. Demonstration of licking, eating and drinking"},
        {"file": "083.png", "caption": "83. Great Wall of China"},
        {"file": "084.png", "caption": "84. House construction (African)"},
        {"file": "085.png", "caption": "85. Construction scene (Amish country)"},
        {"file": "086.png", "caption": "86. House (Africa)"},
        {"file": "087.png", "caption": "87. House (New England)"},
        {"file": "088.png", "caption": "88. Modern house (Cloudcroft, New Mexico)"},
        {"file": "089.png", "caption": "89. House interior with artist and fire"},
        {"file": "090.png", "caption": "90. Taj Mahal"},
        {"file": "091.png", "caption": "91. English city (Oxford)"},
        {"file": "092.png", "caption": "92. Boston"},
        {"file": "093.png", "caption": "93. UN Building Day"},
        {"file": "094.png", "caption": "94. UN Building Night"},
        {"file": "095.png", "caption": "95. Sydney Opera House"},
        {"file": "096.png", "caption": "96. Artisan with drill"},
        {"file": "097.png", "caption": "97. Factory interior"},
        {"file": "098.png", "caption": "98. Museum"},
        {"file": "099.png", "caption": "99. X-ray of hand"},
        {"file": "100.png", "caption": "100. Woman with microscope"},
        {"file": "101.png", "caption": "101. Street scene, Asia (Pakistan)"},
        {"file": "102.png", "caption": "102. Rush hour traffic, India"},
        {"file": "103.png", "caption": "103. Modern highway (Ithaca, NY)"},
        {"file": "104.png", "caption": "104. Golden Gate Bridge"},
        {"file": "105.png", "caption": "105. Train"},
        {"file": "106.png", "caption": "106. Airplane in flight"},
        {"file": "107.png", "caption": "107. Airport (Toronto)"},
        {"file": "108.png", "caption": "108. Antarctic Expedition"},
        {"file": "109.png", "caption": "109. Radio telescope (Westerbork, Netherlands)"},
        {"file": "110.png", "caption": "110. Radio telescope (Arecibo)"},
        {"file": "111.png", "caption": "111. Page of book (Newton, System of the World)"},
        {"file": "112.png", "caption": "112. Astronaut in space"},
        {"file": "113.png", "caption": "113. Titan Centaur launch"},
        {"file": "114.png", "caption": "114. Sunset with birds"},
        {"file": "115.png", "caption": "115. String Quartet (Quartetto Italiano)"},
        {"file": "116.png", "caption": "116. Violin with music score (Cavatina)"},
        {"file": "117.png", "caption": "117. Statement 1/2"},
        {"file": "118.png", "caption": "118. Statement 2/2"},
        {"file": "119.png", "caption": "119. Credits 1/4"},
        {"file": "120.png", "caption": "120. Credits 2/4"},
        {"file": "121.png", "caption": "121. Credits 3/4"},
        {"file": "122.png", "caption": "122. Credits 4/4"}
    ]
    
    # Define the list of 31 audio files (001.mp3 to 031.mp3) with associated captions.
    audios = [
        {"file": "001.mp3", "caption": "1. Greetings from the Secretary-General of the UN – Kurt Waldheim"},
        {"file": "002.mp3", "caption": "2. Greetings In 55 Languages"},
        {"file": "003.mp3", "caption": "3. United Nations Greetings / Whale Songs"},
        {"file": "004.mp3", "caption": "4. The Sounds Of Earth"},
        {"file": "005.mp3", "caption": "5. Brandenburg Concerto No. 2 in F Major, BWV 1047: I. Allegro – Munich Bach Orchestra / Karl Richter (Johann Sebastian Bach)"},
        {"file": "006.mp3", "caption": "6. Ketawang: Puspåwårnå (Kinds of Flowers) – Pura Paku Alaman Palace Orchestra / K.R.T. Wasitodipuro"},
        {"file": "007.mp3", "caption": "7. Cengunmé – Mahi musicians"},
        {"file": "008.mp3", "caption": "8. Alima Song – Mbuti of the Ituri Rainforest"},
        {"file": "009.mp3", "caption": "9. Barnumbirr & Moikoi Song – Tom Djawa, Mudpo and Waliparu"},
        {"file": "010.mp3", "caption": "10. El Cascabel – Antonio Maciel and Los Aguilillas with Mariachi México de Pepe Villa / Rafael Carrión (Lorenzo Barcelata)"},
        {"file": "011.mp3", "caption": "11. Johnny B. Goode – Chuck Berry"},
        {"file": "012.mp3", "caption": "12. Mariuamangɨ – Pranis Pandang and Kumbui of the Nyaura Clan"},
        {"file": "013.mp3", "caption": "13. Sokaku-Reibo (Depicting the Cranes in Their Nest) – Goro Yamaguchi"},
        {"file": "014.mp3", "caption": "14. Partita for Violin Solo No. 3 in E Major, BWV 1006: III. Gavotte en Rondeau – Arthur Grumiaux (Johann Sebastian Bach)"},
        {"file": "015.mp3", "caption": "15. The Magic Flute (Die Zauberflöte), K. 620, Act II: Hell’s Vengeance Boils in My Heart – Bavarian State Opera Orchestra and Chorus / Wolfgang Sawallisch (Wolfgang Amadeus Mozart)"},
        {"file": "016.mp3", "caption": "16. Chakrulo – Georgian State Merited Ensemble of Folk Song and Dance / Anzor Kavsadze"},
        {"file": "017.mp3", "caption": "17. Roncadoras and Drums – Musicians from Ancash"},
        {"file": "018.mp3", "caption": "18. Melancholy Blues – Louis Armstrong and His Hot Seven (Marty Bloom / Walter Melrose)"},
        {"file": "019.mp3", "caption": "19. Muğam – Kamil Jalilov"},
        {"file": "020.mp3", "caption": "20. The Rite of Spring (Le Sacre du Printemps), Part II—The Sacrifice: VI. Sacrificial Dance (The Chosen One) – Columbia Symphony Orchestra / Igor Stravinsky"},
        {"file": "021.mp3", "caption": "21. The Well-Tempered Clavier, Book II: Prelude & Fugue No. 1 in C Major, BWV 870 – Glenn Gould (Johann Sebastian Bach)"},
        {"file": "022.mp3", "caption": "22. Symphony No. 5 in C Minor, Opus 67: I. Allegro Con Brio – Philharmonia Orchestra / Otto Klemperer (Ludwig Van Beethoven)"},
        {"file": "023.mp3", "caption": "23. Izlel e Delyu Haydutin – Valya Balkanska"},
        {"file": "024.mp3", "caption": "24. Navajo Night Chant, Yeibichai Dance – Ambrose Roan Horse, Chester Roan and Tom Roan"},
        {"file": "025.mp3", "caption": "25. The Fairie Round – Early Music Consort of London / David Munrow (Anthony Holborne)"},
        {"file": "026.mp3", "caption": "26. Naranaratana Kookokoo (The Cry of the Megapode Bird) – Maniasinimae and Taumaetarau Chieftain Tribe of Oloha and Palasu'u Village Community"},
        {"file": "027.mp3", "caption": "27. Wedding Song – Young girl of Huancavelica"},
        {"file": "028.mp3", "caption": "28. Liu Shui (Flowing Streams) – Guan Pinghu"},
        {"file": "029.mp3", "caption": "29. Bhairavi: Jaat Kahan Ho – Kesarbai Kerkar"},
        {"file": "030.mp3", "caption": "30. Dark Was the Night, Cold Was the Ground – Blind Willie Johnson"},
        {"file": "031.mp3", "caption": "31. String Quartet No. 13 in B-flat Major, Opus 130: V. Cavatina – Budapest String Quartet (Ludwig Van Beethoven)"},
    ]
    
    # Randomly select one image and one audio.
    selected_image = random.choice(images)
    selected_audio = random.choice(audios)
    
    # Build the custom content with the selected image and audio.
    custom_content = f"""
Quantized to `{formatted_qtype}` using [SpongeQuant](https://github.com/SpongeEngine/SpongeQuant), the Oobabooga of LLM quantization. Chat & support at [Sponge Engine](https://discord.gg/azNmr2Gdgy).

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
"""
    merged_yaml = yaml.dump(custom_metadata, default_flow_style=False)
    full_card = f"---\n{merged_yaml}---\n\n{custom_content}"
    return full_card