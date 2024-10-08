import subprocess
import sys

# Required packages; Standard libraries and third-party libraries
required_libraries = [
    "pandas",          # Data analysis and manipulation library
    "openai",          # OpenAI API client for accessing GPT models and other services
    "beautifulsoup4",  # Web scraping library for parsing HTML and XML documents
    "nltk",            # Natural Language Toolkit for text processing and analysis
    "unidecode"        # Translates Unicode text to ASCII, useful for normalizing text
]


# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Check and install third-party packages if necessary
for packages in required_libraries:
    try:
        __import__(packages)
        print(f"'{packages}' is already installed.")
    except ImportError:
        print(f"'{packages}' not found, installing...")
        install(packages)

print("All packages are successfully imported and ready to use.")
