# setup_config.py

import os


def read_config():
    """Read current configuration from config.py."""
    config = {}
    if os.path.exists('config.py'):
        with open('config.py', 'r') as f:
            lines = f.readlines()
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip("'\"")
                    config[key] = value
    return config


def update_config(new_settings):
    """
    Update the config.py file with new settings.
    :param new_settings: Dictionary of settings to update.
    """
    config = read_config()
    config.update(new_settings)

    # Write updated settings back to config.py
    with open('config.py', 'w') as f:
        for key, value in config.items():
            if value and isinstance(value, str):
                f.write(f"{key} = '{value}'\n")
            elif value:
                f.write(f"{key} = {value}\n")
    print("Configurations have been updated in config.py.")
    print("Current config settings:")
    for key, value in config.items():
        print(f"{key} = {value}")


def setup_root_folder():
    """Prompt user for root folder and update config.py."""
    root_folder = input("Please specify the root folder: ").strip()

    # Hardcoded subfolders
    scrape_result_folder = os.path.join(root_folder, 'scrape_results')
    parsing_result_folder = os.path.join(root_folder, 'parsing_results')

    # Ensure the subfolders exist
    if not os.path.exists(scrape_result_folder):
        os.makedirs(scrape_result_folder)
    if not os.path.exists(parsing_result_folder):
        os.makedirs(parsing_result_folder)

    # Update config.py with the folder paths
    update_config({
        'ROOT_FOLDER': root_folder,
        'SCRAPE_RESULT_FOLDER': scrape_result_folder,
        'PARSING_RESULT_FOLDER': parsing_result_folder
    })


def check_and_setup_config():
    """Check if config.py is properly configured or ask for input."""
    config = read_config()

    # Check if any critical config values are missing or invalid
    if not all(config.get(key) for key in ['ROOT_FOLDER', 'SCRAPE_RESULT_FOLDER', 'PARSING_RESULT_FOLDER']):
        print("One or more critical configuration settings are missing or invalid. Let's configure them.")
        setup_root_folder()
    else:
        print(f"Current configuration settings:")
        for key in ['ROOT_FOLDER', 'SCRAPE_RESULT_FOLDER', 'PARSING_RESULT_FOLDER']:
            print(f"{key} = {config.get(key)}")


def prompt_for_config():
    """Prompt the user for configuration settings if not already set."""
    print("Configuration settings are missing or incomplete.")

    api_key = input("Enter your OpenAI API key: ")
    database_url = input("Enter your database URL: ")
    logging_level = input("Enter logging level (e.g., INFO, DEBUG): ")

    with open('config.py', 'w') as f:
        f.write(f"OPENAI_API_KEY = '{api_key}'\n")
        f.write(f"DATABASE_URL = '{database_url}'\n")
        f.write(f"LOGGING_LEVEL = '{logging_level}'\n")

    print("Configuration settings have been written to 'config.py'.")
