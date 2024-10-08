# Initialization script that checks for required external packages
# import initialize

# Configuration script that contains variables and script to change config
import config

# Scraper script that contains functions for FRUS scraping
import scraper

#
from parsing_module import read_date_and_keywords, process_epub_file, process_with_openai, process_existing_csv, transform_to_longform, write_final_output, split_long_cells

# Logging script that creates log file and adds to said file
from logger import create_log_file, log_message

import os
# import glob
import logging
# import requests
import re
# import time
# from lxml import html
# from unidecode import unidecode
# from urllib.parse import urljoin
from datetime import datetime as dt
# from tqdm import tqdm
# from random import randint
# import zipfile

import pandas as pd
# import openai
# from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def check_scrape_result_folder(scrape_result_folder):
    """Check if the scrape result folder contains any EPUB files."""
    if not os.path.exists(scrape_result_folder):
        return False
    epub_files = [file for file in os.listdir(scrape_result_folder) if file.endswith('.epub')]
    return len(epub_files) > 0


def prompt_for_scraping(root_folder, scrape_result_folder):
    """Prompt the user for scraping action, check for url_list.txt, and start the scraper if necessary."""
    url_list_path = os.path.join(root_folder, 'url_list.txt')

    if not os.path.exists(url_list_path):
        # Generate an empty url_list.txt if it doesn't exist
        with open(url_list_path, 'w') as f:
            pass
        print(f"No url_list.txt found. A new file has been generated at {url_list_path}.")
        print("Please fill it out with URLs before proceeding.")
        return False

    # If url_list.txt exists, display the URLs to the user for verification
    with open(url_list_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print(f"url_list.txt is empty. Please fill it out with URLs before proceeding.")
        return False

    print("The following URLs have been found in url_list.txt:")
    for url in urls:
        print(f"- {url}")

    # Ask user to verify
    user_verification = input("Do you want to proceed with scraping? (1 for yes, 0 for no): ")
    if user_verification == '1':
        scraper.process_urls_from_file(
            scrape_result_folder,
            file_extension='epub',  # Assuming EPUB format for scraping results
            start_year=1945,
            end_year=1985,
            test_mode=False
        )
        return True
    else:
        print("Scraping process terminated.")
        return False


def prompt_for_parsing():
    """Prompt the user to start fresh parsing or load an existing CSV."""
    # Path to the 'data' directory inside the root folder
    data_folder = config.PARSING_RESULT_FOLDER
    csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

    if csv_files:
        choice = input("Would you like to (1) Start Fresh or (2) Load an Existing CSV? Enter 1 or 2: ").strip()
    else:
        choice = "1"  # Default to start fresh if no intermediate CSV exists

    return choice


def handle_parsing(log_file):
    """Handles the full parsing and OpenAI integration."""
    # Prompt the user to either load existing CSV or start fresh
    choice = prompt_for_parsing()
    processed_df = None  # Initialize processed_df to prevent reference errors

    if choice == "1":
        # Start fresh: parse EPUB files, run OpenAI API, and calculate consensus
        log_message(log_file, "main/handle_parsing",
                    "Starting fresh parsing of EPUB files.")

        # Load dictionary keywords and date ranges
        date_keyword_map = read_date_and_keywords(log_file)
        log_message(log_file, "main/handle_parsing",
                    f"Keyword Map: {date_keyword_map}")
        if date_keyword_map is None:
            log_message(log_file, "main/handle_parsing",
                        "Failed to load dictionary CSV. Exiting.")
            return

        # Get the list of EPUB files to process
        epub_files = [f for f in config.SCRAPE_RESULT_FOLDER.glob('*.epub')]
        if not epub_files:
            log_message(log_file, "main/handle_parsing",
                        "No EPUB files found in the 'scraped' folder. Exiting.")
            return

        all_dfs = []

        for i, epub_path in enumerate(epub_files):
            log_message(log_file, "main/handle_parsing",
                        f"Processing file {i + 1}/{len(epub_files)}: {epub_path}")

            # Extract year from the EPUB file name
            file_year_match = re.search(r'\d{4}', epub_path.name)
            if not file_year_match:
                log_message(log_file, "main/handle_parsing",
                            f"No year found in file name {epub_path}, skipping.")
                continue

            file_year = int(file_year_match.group())

            # Find relevant date ranges in date_keyword_map for the file's year
            relevant_date_ranges = [
                date_range for date_range in date_keyword_map.keys()
                if any(int(start[:4]) <= file_year <= int(end[:4]) for start, end in date_keyword_map.keys())
            ]
            log_message(log_file, "handle_parsing",
                        f"Relevant date ranges for {file_year}: {relevant_date_ranges}")

            if not relevant_date_ranges:
                log_message(log_file, "main/handle_parsing",
                            f"No relevant date ranges for file {epub_path}, skipping.")
                continue

            # Process EPUB file if its year is within relevant date ranges
            try:
                df, num_lines, skipped_count, skipped_docs, processed_count = process_epub_file(
                    epub_path,
                    date_keyword_map={date_range: date_keyword_map[date_range] for date_range in relevant_date_ranges},
                    log_filename=log_file
                )

                if df.empty:
                    log_message(log_file, "main/handle_parsing",
                                f"No data parsed from {epub_path}, skipping to next file.")
                    continue

                all_dfs.append(df)
                log_message(log_file, "main",
                            f"Processed {epub_path}. Documents processed: {processed_count}. Skipped: {skipped_count}.")

            except Exception as e:
                log_message(log_file, "main/handle_parsing",
                            f"Error processing {epub_path}: {str(e)}")
                continue

        if not all_dfs:
            log_message(log_file, "main/handle_parsing",
                        "No documents were processed from the EPUB files. Exiting.")
            return

        # Dataframe construction
        # 1. Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # 2. Process using OpenAI API and consensus algorithm
        processed_df = process_with_openai(combined_df, log_file)
        # 3. Assign the flag based on the consensus score
        processed_df['flag'] = processed_df['consensus_score'].apply(
            lambda score: 1 if score < config.DESIRED_CONSENSUS else 0)

        # Define columns as guideline for .to_csv
        # Todo: Change from hard-coded list to variable, preferably move to config.py
        metadata_columns = ['date', 'docid', 'stateb', 'disno', 'title']
        final_columns = metadata_columns + ['text', 'response_text', 'speaker', 'consensus_score', 'flag']

        # Save intermediate CSV
        intermediate_csv_path = config.PARSING_RESULT_FOLDER / f"intermediate_{config.LOG_FILENAME}{dt.now().strftime('%y%m%d_%H%M%S')}.csv"
        processed_df = split_long_cells(processed_df)
        processed_df.to_csv(intermediate_csv_path, index=False)
        log_message(log_file, "main/handle_parsing",
                    f"Intermediate CSV saved at: {intermediate_csv_path}")

    elif choice == "2":
        # Load existing intermediate CSV and reprocess flagged documents
        intermediate_csv_path = input("Enter the path to the existing CSV: ").strip()
        log_message(log_file, "main/handle_parsing",
                    f"Loading existing CSV from {intermediate_csv_path}")
        processed_df = process_existing_csv(intermediate_csv_path, log_file)

    # Reprocessing flagged documents (optional based on user input)
    flagged_df = processed_df[processed_df['flag'] == 1].copy()
    if not flagged_df.empty:
        reprocess_choice = input(
            "There are flagged documents. Would you like to (1) Reprocess them or (2) Skip reprocessing? Enter 1 or 2: ").strip()
        if reprocess_choice == "1":
            log_message(log_file, "main/handle_parsing",
                        "Reprocessing flagged documents.")
            reprocessed_df = process_with_openai(flagged_df, log_file)
            processed_df.update(reprocessed_df)

    # Split text into communication acts for documents with flag == 0
    unflagged_df = processed_df[(processed_df['flag'] == 0) & (processed_df['consensus_score'] >= config.DESIRED_CONSENSUS)].copy()
    if unflagged_df.empty:
        log_message(log_file, "main/handle_parsing",
                    "No unflagged documents with sufficient consensus to process for communication acts. Exiting.")
        return

    log_message(log_file, "main/handle_parsing",
                "Splitting text into communication acts and assigning speakers.")

    # Retain only the first instance per document based on 'docid'
    unflagged_df = unflagged_df.drop_duplicates(subset=['docid'], keep='first')

    # Transform unflagged_df into longform format
    longform_df = transform_to_longform(unflagged_df, log_file)

    # Write final output to CSV
    final_csv, flagged_csv = write_final_output(final_df=longform_df, flagged_df=flagged_df, log_filename=log_file)

    if final_csv:
        log_message(log_file, "main/handle_parsing",
                    f"Final CSV saved at: {final_csv}")
    if flagged_csv:
        log_message(log_file, "main/handle_parsing",
                    f"Flagged documents CSV saved at: {flagged_csv}")
    else:
        log_message(log_file, "main/handle_parsing",
                    "No flagged documents CSV created.")

    log_message(log_file, "main/handle_parsing",
                "Parsing and processing complete.")


def main():
    # Logger module for logging begins
    log_file = create_log_file()

    # Ensure config is imported correctly
    try:
        # Check if all required configurations are provided in config.py
        if not (config.ROOT_FOLDER and config.SCRAPE_RESULT_FOLDER and config.PARSING_RESULT_FOLDER):
            print("Configuration is incomplete. Please ensure all required settings are provided in config.py.")
            return

        # Step 1: Check if the scrape_result_folder contains EPUB files
        if not check_scrape_result_folder(config.SCRAPE_RESULT_FOLDER):
            print("No EPUB files found in the scrape results folder.")
            user_input = input("Do you want to run the scraping module? (1 for yes, 0 for no): ").strip()

            if user_input == '1':
                # Check for url_list.txt and run scraping
                if not prompt_for_scraping(config.ROOT_FOLDER, config.SCRAPE_RESULT_FOLDER):
                    return
            else:
                print("No EPUB files available and scraping was not initiated. Terminating program.")
                return

        # Step 2: Proceed to parsing module
        print("Proceeding to the parsing module...")
        handle_parsing(log_file)

    except ImportError:
        print("Error importing configuration. Please ensure config.py exists and is properly formatted.")


if __name__ == "__main__":
    main()
