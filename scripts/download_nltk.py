import os

import nltk

# Set NLTK data path to local directory
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

print(f"Downloading vader_lexicon to {nltk_data_path}...")
nltk.download("vader_lexicon", download_dir=nltk_data_path)
print("Done.")
