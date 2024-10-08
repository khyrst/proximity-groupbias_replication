import config
from gpt import get_consensus_results, get_consensus_results_with_retries
from logger import log_message

import os
import ast
import time
import zipfile
import re
import logging
import html
import unicodedata
import pandas as pd
import concurrent.futures
import threading
from tqdm import tqdm
from unidecode import unidecode
from bs4 import BeautifulSoup
from datetime import datetime as dt

# Dictionaries
date_patterns = [
    r'\wndated.',  # Undated
    r'\b\w+ \d{1,2} \(.\), \d{4}',  # e.g., August 21 (?), 1943
    r'\b\w+ \d{1,2}-\d{1,2}\W{1,2}\d{4}',  # e.g., August 12-30, 1943
    r'\b\w+ \d{1,2}(?:th|nd|st|rd)\W{1,2}\d{4}',  # e.g., August 3rd 1945 ; August 18th, 1945
    r'\b\w+ \d{1,2}\W{1,2}\d{4}',  # e.g., August 3 1945 ; August 29, 1949
    r'\b\d{1,2}(?:th|nd|st|rd) \w+..\d{4}',  # e.g., 19th August 1945 ; 8th August, 1945
    r'\b\d{1,2} \w+..\d{4}',  # e.g., 19 August 1945 ; 19 August, 1945
    r'\b\w+\W{1,2}\d{4}',  # e.g., July, 1945 ; July 1945
    r'\b\d{1,2}.\d{1,2}.\d{2,4}'  # e.g., 16/7/43
]

lock = threading.Lock()


def normalize_unicode_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\n', ' ').replace('\r', '')  # Replace newlines with spaces
        return text
    return text


def decode_unicode_dataframe(dataframe):
    dataframe = dataframe.applymap(lambda x: unidecode(x) if isinstance(x, str) else x)
    return dataframe


def clean_text(text):
    # 1. Normalize special characters (convert curly quotes to straight, etc.)
    normalized_text = unicodedata.normalize('NFKC', text)
    # 2. Unescape HTML entities (like &nbsp;, &mdash;)
    unescaped_text = html.unescape(normalized_text)
    # 3. Replace problematic dashes with regular ones (optional, but anything goes)
    cleaned_text = unescaped_text.replace('—', '-').replace('–', '-')
    return cleaned_text


def split_long_cells(df, text_column='text', max_length=32000):
    """
    Splits rows in the DataFrame where the text column exceeds the max_length limit, copying metadata to new rows.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        text_column (str): The name of the column containing the text to split.
        max_length (int): The maximum length of text allowed in a single cell (Excel limit ~32767).

    Returns:
        pd.DataFrame: A DataFrame where long text rows are split, with metadata copied to new rows.
    """
    split_rows = []

    # Iterate through each row in the DataFrame
    for idx, row in df.iterrows():
        text = row[text_column]

        # If the text length exceeds the limit, split it
        if isinstance(text, str) and len(text) > max_length:
            # Split the text into chunks of max_length
            chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]

            # Create a new row for each chunk, copying the metadata
            for chunk in chunks:
                new_row = row.copy()  # Copy the entire row
                new_row[text_column] = chunk  # Replace the text with the chunk
                split_rows.append(new_row)
        else:
            split_rows.append(row)  # If text is within limit, just append the row as is

    # Return a DataFrame with split rows
    return pd.DataFrame(split_rows)


def merge_split_cells(df):
    """
    Concatenates rows in a DataFrame with matching metadata fields, joining their text if they were split.
    Assumes the metadata fields are ['date', 'docid', 'stateb', 'disno', 'title'].

    Args:
        df (pd.DataFrame): The DataFrame with split rows (from a CSV).

    Returns:
        pd.DataFrame: A cleaned DataFrame with concatenated text and no duplicates.
    """
    merged_rows = []
    metadata_cols = ['date', 'docid', 'stateb', 'disno', 'title']
    previous_row = None
    concatenated_text = None

    for idx, row in df.iterrows():
        if previous_row is None:
            previous_row = row
            concatenated_text = row['text']
        else:
            # Check if the metadata matches with the previous row
            if all(row[col] == previous_row[col] for col in metadata_cols):
                concatenated_text += " " + row['text']
            else:
                previous_row['text'] = concatenated_text  # Update text with concatenated value
                merged_rows.append(previous_row)  # Append to merged_rows list

                # Start a new sequence
                previous_row = row
                concatenated_text = row['text']

    # Append the last row
    if previous_row is not None:
        previous_row['text'] = concatenated_text
        merged_rows.append(previous_row)

    # Create a new DataFrame from the merged rows
    cleaned_df = pd.DataFrame(merged_rows)

    return cleaned_df


def extract_document_date(document_soup):
    """
    Function to iterate through "date_patterns" and extract date from the HTML document.
    :param document_soup: BeautifulSoup object of the document.
    :return: A non-standardized date format as it appears in the document.
    """
    text = document_soup.get_text(separator=' ', strip=True)

    # First, try to extract dates from <span> with class 'tei-date'
    date_span = document_soup.find('span', class_='tei-date')
    if date_span:
        date_text = re.sub(r'\s+', ' ', date_span.get_text().strip())
        if re.match(r'\b\w+ \d{1,2}, \d{4}', date_text):  # Example: January 19, 1950
            logging.debug(f"Date match found in <span class='tei-date'>: {date_text}")
            return date_text

    # Iterate through predefined date patterns
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Exclude matches that are part of URLs or HTML tags
            if 'http' in match or 'org' in match or '/' in match:
                continue
            logging.debug(f"Date match found: {match}")
            return match

    logging.debug("Date not found or Undated")
    return "Undated"


def standardize_date(date_str):
    """
    Function to take a date string and standardize it into YYYYMMDD format
    :param date_str: string of the date to be standardized
    :return: a string of the standardized date in YYYYMMDD format, or "Undated"
    """
    months = {
        'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06',
        'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12'
    }
    month_text_regex = r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b'

    def replace_month_name(s):
        for month, number in months.items():
            s = re.sub(r'\b' + month + r'\b', number, s, flags=re.IGNORECASE)
        return s

    # Normalize the date string by removing commas and extra spaces
    date_str = date_str.replace(',', '').strip()
    logging.debug(f"Normalized date string: {date_str}")

    try:
        # Month day, year -> MMDDYYYY
        if re.match(rf'{month_text_regex} \d{{1,2}} \d{{4}}', date_str):
            date_str = replace_month_name(date_str)
            date_str = re.sub(r'(th|nd|st|rd)', '', date_str)
            logging.debug(f"Date string after replacing month and removing ordinals: {date_str}")
            date_obj = dt.strptime(date_str, '%m %d %Y')
            return date_obj.strftime('%Y%m%d')
        # Day month year -> DDMMYYYY
        elif re.match(rf'\d{{1,2}} {month_text_regex} \d{{4}}', date_str):
            date_str = replace_month_name(date_str)
            date_str = re.sub(r'(th|nd|st|rd)', '', date_str)
            logging.debug(f"Date string after replacing month and removing ordinals: {date_str}")
            date_obj = dt.strptime(date_str, '%d %m %Y')
            return date_obj.strftime('%Y%m%d')
    except ValueError as e:
        logging.error(f"Error when standardizing date: {e}")

    # Last resort: Check for standalone year
    if re.search(r'\b\d{4}\b', date_str) and not re.search(r'\b\d{1,2} \d{1,2} \d{4}', date_str):
        year = re.search(r'\b\d{4}\b', date_str).group()
        return f"{year}0101"

    return "Undated"


def read_date_and_keywords(log_filename):
    """
    Reads the dictionary CSV file and compiles a map of date ranges to associated `disno` and keywords.
    Logs the process for debugging purposes.

    Returns:
        date_keyword_map: A dictionary with date ranges as keys and a set of keywords for each `disno`.
    """
    log_message(log_filename, "parsing_module/read_date_and_keywords",
                "Starting to read the dictionary CSV file.")

    date_keyword_map = {}

    try:
        # Read the CSV file from the path specified in config
        df = pd.read_csv(config.DICTIONARY_CSV)
        log_message(log_filename, "parsing_module/read_date_and_keywords",
                    f"Successfully read the dictionary file: {config.DICTIONARY_CSV}")
    except Exception as e:
        log_message(log_filename, "parsing_module/read_date_and_keywords",
                    f"Error reading dictionary CSV: {str(e)}")
        return None

    # Specify the columns for start_date, end_date, and disno (dispute number)
    start_date_col = 'start_date_padded'
    end_date_col = 'end_date'
    disno_col = 'disno'

    # Specify the columns for keywords (can be extended as needed)
    keyword_columns = ['StateNme', 'StateNme_more', 'capital', 'Execname', 'leader']

    # Iterate through the CSV rows to build the date-keyword map
    for _, row in df.iterrows():
        start_date = str(row[start_date_col])
        end_date = str(row[end_date_col])
        disno = str(row[disno_col])

        # Combine date range as the key
        date_range_key = (start_date, end_date)

        # Initialize the entry in the map if not present
        if date_range_key not in date_keyword_map:
            date_keyword_map[date_range_key] = {}

        # Create a set of keywords for this disno
        if disno not in date_keyword_map[date_range_key]:
            date_keyword_map[date_range_key][disno] = set()

        # Add keywords from the row into the set
        for col in keyword_columns:
            if col in df.columns:
                cell_value = row[col]
                if pd.notna(cell_value):
                    # Split keywords by comma and add them to the set
                    keywords = str(cell_value).split(', ')
                    date_keyword_map[date_range_key][disno].update(keywords)

    log_message(log_filename, "parsing_module/read_date_and_keywords",
                "Finished processing dictionary CSV file.")
    return date_keyword_map


def process_epub_file(epub_path, date_keyword_map, limit=False, limit_num=None, log_filename=None):
    """
    Processes a single EPUB file, extracting content from HTML/XHTML documents,
    filtering based on date and keywords, and returning a DataFrame of relevant text.

    Args:
        epub_path: The path to the EPUB file.
        date_keyword_map: Dictionary of date ranges to keywords for filtering.
        limit: Boolean flag to limit the number of processed documents.
        limit_num: The number of documents to limit if applicable.
        log_filename: The log file where logs will be written.

    Returns:
        df: A pandas DataFrame of processed content.
        num_lines: Number of individual lines parsed.
        skipped_count: Count of skipped documents.
        skipped_documents: List of skipped document names.
        processed_file_count: Total number of processed documents.
    """
    all_text_blocks = []
    processed_file_count = 0
    skipped_count = 0
    skipped_documents = []

    epub_file_name = os.path.basename(epub_path).replace('.epub', '')
    start_time = time.time()  # Start time for timeout check

    try:
        book = zipfile.ZipFile(epub_path, 'r')
        log_message(log_filename, "parsing_module/process_epub_file",
                    f"Processing EPUB file from path: {epub_path}")

        for idx, item in enumerate(book.namelist()):
            # Timeout after 300 seconds (5 minutes) to prevent infinite loop
            if time.time() - start_time > 300:
                log_message(log_filename, "process_epub_file",
                            "Processing timeout, terminating loop.")
                break

            log_message(log_filename, "parsing_module/process_epub_file",
                        f"Processing item {idx + 1}/{len(book.namelist())}: {item}")

            # Only process files with names like 'd1', 'd2', etc.
            if not re.match(r'd\d+', os.path.splitext(os.path.basename(item))[0]):
                continue

            if item.endswith('.html') or item.endswith('.xhtml'):
                with book.open(item) as file:
                    document_soup = BeautifulSoup(file.read(), 'html.parser')

                # Use HTML file names as document numbers
                doc_num = os.path.splitext(os.path.basename(item))[0]

                # Skip documents with title containing "Editorial Note"
                title_tag = document_soup.title
                if title_tag and re.search(r'\bEditorial Note\b', title_tag.string, re.IGNORECASE):
                    log_message(log_filename, "process_epub_file",
                                "Document is Editorial Note, skipped.")
                    skipped_count += 1
                    skipped_documents.append(f"Editorial Note: {item}")
                    continue

                log_message(log_filename, "parsing_module/process_epub_file",
                            f"Found Document {doc_num}")

                # Extract and standardize document date
                doc_date_raw = extract_document_date(document_soup)
                if not doc_date_raw:
                    log_message(log_filename, "parsing_module/process_epub_file",
                                "Document date not found, skipping.")
                    skipped_count += 1
                    skipped_documents.append(f"No Document Date: {item}")
                    continue

                formatted_date = standardize_date(doc_date_raw)
                if not formatted_date or formatted_date == "Undated":
                    log_message(log_filename, "parsing_module/process_epub_file",
                                "Invalid or undated document, skipping.")
                    skipped_count += 1
                    skipped_documents.append(f"Invalid/Undated Date: {item}")
                    continue

                # Compare dates to filter
                try:
                    doc_date_dt = dt.strptime(formatted_date, '%Y%m%d')
                except Exception as e:
                    log_message(log_filename, "parsing_module/process_epub_file",
                                f"Error comparing dates: {e}")
                    skipped_count += 1
                    skipped_documents.append(f"Date Comparison Error: {item}")
                    continue

                matched_disno = set()
                matched_keywords = []

                # Iterate over date ranges and keywords to find matches
                for (start_date, end_date), disno_keywords_map in date_keyword_map.items():
                    try:
                        start_date_dt = dt.strptime(start_date, '%Y%m%d')
                        end_date_dt = dt.strptime(end_date, '%Y%m%d')
                        if start_date_dt <= doc_date_dt <= end_date_dt:
                            for disno, keywords in disno_keywords_map.items():
                                document_text = document_soup.get_text(separator=' ', strip=True)
                                for keyword in keywords:
                                    if keyword.lower() in document_text.lower():
                                        matched_keywords.append(keyword)
                                        matched_disno.add(disno)
                    except Exception as e:
                        log_message(log_filename, "parsing_module/process_epub_file",
                                    f"Error processing date range {start_date} to {end_date}: {e}")

                if not matched_disno:
                    log_message(log_filename, "parsing_module/process_epub_file",
                                "No keywords found in document, skipping.")
                    skipped_count += 1
                    skipped_documents.append(f"No Keywords: {item}")
                    continue

                disno_str = ', '.join(matched_disno)

                # Remove superscript and footnotes
                # for sup_tag in document_soup.find_all('sup'):
                #     sup_tag.decompose()
                for footnotes_div in document_soup.find_all('div', class_='footnotes'):
                    footnotes_div.decompose()

                # Grab the full text from the document using BeautifulSoup's get_text() method
                document_text = document_soup.get_text(separator=' ', strip=True)
                document_text = clean_text(document_text)

                # document_text = document_text.encode('ascii', 'ignore').decode('ascii')

                if not document_text:
                    log_message(log_filename, "parsing_module/process_epub_file",
                                "No relevant text found, skipping.")
                    skipped_count += 1
                    skipped_documents.append(f"No Relevant Text: {item}")
                    continue

                # Build document ID and append the entire document text
                doc_id = f"{epub_file_name}_{doc_num}"

                # noinspection PyTypeChecker
                all_text_blocks.append((
                    formatted_date,
                    doc_num,
                    doc_id,
                    disno_str,
                    str(title_tag.string) if title_tag and title_tag.string else "Untitled",  # Ensure title is a string
                    document_text
                ))

                processed_file_count += 1
                if limit and processed_file_count >= limit_num:
                    break

    except Exception as e:
        log_message(log_filename, "parsing_module/process_epub_file",
                    f"Error processing EPUB file {epub_path}: {e}")

    # Create DataFrame and handles unicode characters with normalize_unicode_text()
    df = pd.DataFrame(all_text_blocks, columns=['date', 'docid', 'stateb', 'disno', 'title', 'text'])
    df = df.applymap(normalize_unicode_text)
    df = decode_unicode_dataframe(df)

    num_lines = len(df)
    log_message(log_filename, "parsing_module/process_epub_file",
                f"Processed {processed_file_count} files. Total number of lines parsed: {num_lines}")

    return df, num_lines, skipped_count, skipped_documents, processed_file_count


def process_with_openai(df, log_file, max_workers=config.MAX_WORKERS):
    """
    Process dataframe using OpenAI API and consensus algorithm in parallel,
    and return a dataframe of results.

    Args:
        df (pd.DataFrame): DataFrame containing input text and metadata.
        log_file (str): Log file for logging information.
        max_workers (int): Maximum number of threads to run concurrently.

    Returns:
        pd.DataFrame: Processed DataFrame with consensus results.
    """
    results = []

    def api_call_wrapper(row):
        """Helper function to process each row and call OpenAI API."""
        try:
            text = row['text']
            api_results, consensus_score = get_consensus_results(text, log_file)

            # Prepare the row with the consensus results
            row_data = row.to_dict()  # Convert original row (including metadata) to dictionary
            for api_result in api_results:
                row_data_copy = row_data.copy()  # Copy row data for each API result
                row_data_copy.update({
                    'response_text': api_result['text'],
                    'speaker': api_result['speaker'],
                    'consensus_score': consensus_score
                })

                # Safely append to results list
                with lock:
                    results.append(row_data_copy)
        except Exception as e:
            log_message(log_file, "process_with_openai", f"Error processing row with index {row.name}: {e}")

    # Use ThreadPoolExecutor to run API requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to show progress bar
        list(tqdm(executor.map(api_call_wrapper, [row for _, row in df.iterrows()]), total=len(df)))

    # Create a new dataframe from the results
    result_df = pd.DataFrame(results)

    return result_df

#    for index, row in df.iterrows():
#        text = row['text']
#
#        # Get results from OpenAI API and consensus score
#        api_results, consensus_score = get_consensus_results(text, log_file)
#
#        # Log each result into the dataframe, even if no consensus
#        for api_result in api_results:
#            # Append metadata from the original dataframe to results
#            row_data = row.to_dict()  # Convert the entire row (including metadata) to a dictionary
#            row_data.update({
#                # 'text': text,
#                'response_text': api_result['text'],
#                'speaker': api_result['speaker'],
#                'consensus_score': consensus_score
#            })
#            results.append(row_data)
#
#    # Create a dataframe from the results
#    result_df = pd.DataFrame(results)
#
#    return result_df


def process_existing_csv(intermediate_csv_path, log_filename):
    """
    Loads an existing intermediate CSV and reprocesses flagged rows only.
    """
    df = pd.read_csv(intermediate_csv_path)

    # First, concatenate split rows
    log_message(log_filename, "parsing_module/process_existing_csv",
                "Concatenating split rows in the existing CSV.")
    df = merge_split_cells(df)

    # Then, identify flagged rows
    flagged_df = df[df['flag'] == 1].copy()  # Filters flagged rows
    if flagged_df.empty:
        log_message(log_filename, "parsing_module/process_existing_csv",
                    "No flagged documents to reprocess.")
        return df  # Nothing to reprocess

    log_message(log_filename, "parsing_module/process_existing_csv",
                f"Reprocessing {len(flagged_df)} flagged documents.")

    # Reprocess flagged rows
    reprocessed_df = process_with_openai(flagged_df, log_filename)

    # Merge reprocessed flagged rows back into the original dataframe
    df.update(reprocessed_df)

    # Save updated CSV
    updated_csv_path = f"updated_{config.LOG_FILENAME}{dt.now().strftime('%y%m%d_%H%M%S')}.csv"
    df.to_csv(updated_csv_path, index=False)
    log_message(log_filename, "parsing_module/process_existing_csv",
                f"Saved updated CSV: {updated_csv_path}")

    return df


def reprocess_flagged_documents(intermediate_csv_path, log_filename):
    # Potentially deprecated, kept for legacy
    # Might need in future
    """
    Reprocess flagged documents that didn't meet the consensus threshold.
    Updates the CSV after reprocessing.

    Args:
        intermediate_csv_path (str): Path to the intermediate CSV with flagged documents.
        log_filename (str): Path to the log file for detailed logging.

    Returns:
        pd.DataFrame: Updated DataFrame after reprocessing flagged documents.
    """
    # Load the existing intermediate CSV
    df = pd.read_csv(intermediate_csv_path)

    # Merge split rows before reprocessing flagged rows
    log_message(log_filename, "parsing_module/reprocess_flagged_documents",
                "Concatenating split rows based on metadata.")
    df = merge_split_cells(df)

    # Identify flagged documents (those with flag == 1)
    flagged_df = df[df['flag'] == 1].copy()
    if flagged_df.empty:
        log_message(log_filename, "parsing_module/reprocess_flagged_documents",
                    "No flagged documents to reprocess.")
        return df  # No documents to reprocess

    log_message(log_filename, "parsing_module/reprocess_flagged_documents",
                f"Reprocessing {len(flagged_df)} flagged documents.")

    # Reprocess flagged documents using OpenAI API
    reprocessed_df = process_with_openai(flagged_df, log_filename)

    # Merge the reprocessed rows back into the original DataFrame
    df.update(reprocessed_df)

    # Save the updated DataFrame as a new CSV file
    updated_csv_path = f"updated_{config.LOG_FILENAME}{dt.now().strftime('%y%m%d_%H%M%S')}.csv"
    df.to_csv(updated_csv_path, index=False)
    log_message(log_filename, "parsing_module/reprocess_flagged_documents",
                f"Updated CSV saved at: {updated_csv_path}")

    # Prompt the user to proceed or terminate
    user_choice = input("Reprocessing complete. Do you want to proceed with splitting text (Y/N)? ").strip().lower()

    if user_choice == 'y':
        return df  # Proceed to the next step
    else:
        log_message(log_filename, "parsing_module/reprocess_flagged_documents",
                    "User chose to terminate after reprocessing.")
        print(f"Reprocessing is complete. The updated CSV has been saved at: {updated_csv_path}.")
        return None  # Terminate the process


def transform_to_longform(df, log_filename):
    """
    Transforms the dataframe to a longform format where each response_text and speaker pair gets its own row.

    Args:
        df (pd.DataFrame): DataFrame containing metadata, response_text (as list), and speaker (as list).
        log_filename (str): Log file to track any issues.

    Returns:
        pd.DataFrame: Longform dataframe with one response_text and speaker per row.
    """
    # Initialize an empty list to store the expanded rows
    longform_data = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Parse the 'response_text' and 'speaker' columns as lists
        try:
            # Ensure the response_text and speaker are actual lists, not string representations of lists
            response_texts = ast.literal_eval(row['response_text']) if isinstance(row['response_text'], str) else row['response_text']
            speakers = ast.literal_eval(row['speaker']) if isinstance(row['speaker'], str) else row['speaker']

            # Check if the lengths of response_texts and speakers match
            if len(response_texts) != len(speakers):
                log_message(log_filename, "parsing_module/transform_to_longform",
                            f"Mismatch in response_text and speaker lengths for docid {row['docid']}, skipping.")
                continue

            # For each response text and corresponding speaker, create a new row
            for response, speaker in zip(response_texts, speakers):
                longform_data.append({
                    'date': row['date'],
                    'docid': row['docid'],
                    'stateb': row['stateb'],
                    'disno': row['disno'],
                    'title': row['title'],
                    'response_text': response.strip(),
                    'speaker': speaker.strip()
                })

        except (ValueError, SyntaxError) as e:
            log_message(log_filename, "parsing_module/transform_to_longform",
                        f"Error evaluating response_text or speaker for docid {row['docid']}: {e}")
            continue

    # Create a new dataframe for the expanded rows
    longform_df = pd.DataFrame(longform_data, columns=['date', 'docid', 'stateb', 'disno', 'title', 'response_text', 'speaker'])

    log_message(log_filename, "parsing_module/transform_to_longform",
                f"Created longform dataframe with {len(longform_df)} rows.")
    return longform_df


def split_text_into_comm_acts(df, log_filename):
    # Deprecated, kept for legacy
    """
    Splits text into communication acts based on speaker transitions and assigns corresponding speakers.

    Args:
        df (pd.DataFrame): DataFrame containing the processed text, index, and speaker columns.
        log_filename (str): Path to the log file for detailed logging.

    Returns:
        pd.DataFrame: A new DataFrame with each communication act and corresponding speaker.
    """
    comm_acts_data = []

    # Iterate through each row of the DataFrame
    for idx, row in df.iterrows():
        # Only process unflagged documents (flag == 0)
        if row['flag'] == 1:
            log_message(log_filename, "parsing_module/split_text_into_comm_acts",
                        f"Skipping flagged document with docid: {row['docid']}")
            continue

        text = row['text']
        indexes = eval(row['index'])  # Convert the stored string list back to a list of lists
        speakers = eval(row['speaker'])  # Convert the stored string list back to a list of speakers

        # Ensure indexes and speakers are valid and of the same length
        if not indexes or not speakers or len(indexes) != len(speakers):
            log_message(log_filename, "parsing_module/split_text_into_comm_acts", f"Invalid data for docid {row['docid']}, skipping.")
            continue

        # Split text into communication acts based on the indexes
        for i, (start_idx, end_idx) in enumerate(indexes):
            comm_act_text = text[start_idx:end_idx].strip()  # Extract the text block for the current communication act
            speaker = speakers[i]  # Get the corresponding speaker

            # Append to the new DataFrame structure
            comm_acts_data.append({
                'date': row['date'],
                'docid': row['docid'],
                'stateb': row['stateb'],
                'disno': row['disno'],
                'title': row['title'],
                'comm_act': comm_act_text,
                'speaker': speaker
            })

    # Create a new DataFrame for communication acts
    comm_acts_df = pd.DataFrame(comm_acts_data,
                                columns=['date', 'docid', 'stateb', 'disno', 'title', 'comm_act', 'speaker'])

    log_message(log_filename, "parsing_module/split_text_into_comm_acts",
                f"Created DataFrame with {len(comm_acts_df)} communication acts.")
    return comm_acts_df


def create_final_dataframe(comm_acts_df, log_filename):
    # Deprecated, kept for legacy
    # Probably won't need it tho
    """
    Creates the final DataFrame for communication acts and writes it to CSV.

    Args:
        comm_acts_df (pd.DataFrame): DataFrame containing the communication acts, metadata, and speakers.
        log_filename (str): Path to the log file for detailed logging.

    Returns:
        str: Path to the final CSV file.
    """
    # Check if the communication acts DataFrame is empty
    if comm_acts_df.empty:
        log_message(log_filename, "parsing_module/create_final_dataframe", "No communication acts to process.")
        return None

    # Final DataFrame should already be structured from Step 4, no need for further modifications
    final_csv_path = f"final_comm_acts_{config.LOG_FILENAME}{dt.now().strftime('%y%m%d_%H%M%S')}.csv"

    try:
        # Write the final DataFrame to CSV
        comm_acts_df.to_csv(final_csv_path, index=False)
        log_message(log_filename, "parsing_module/create_final_dataframe", f"Final CSV created: {final_csv_path}")
    except Exception as e:
        log_message(log_filename, "parsing_module/create_final_dataframe", f"Error writing final CSV: {str(e)}")
        return None

    return final_csv_path


def write_final_output(final_df, flagged_df=None, log_filename=None, max_lines=config.EXCEL_MAX_ROWS):
    """
    Writes the final DataFrame to CSV, splitting into multiple files if needed.
    Also optionally writes flagged documents to a separate CSV.

    Args:
        final_df (pd.DataFrame): The DataFrame containing communication acts and metadata.
        flagged_df (pd.DataFrame, optional): DataFrame containing flagged documents for manual review.
        log_filename (str, optional): The log file path for logging.
        max_lines (int): Maximum desired number of rows in final output CSV, set in config.py.

    Returns:
        tuple: Paths to the final CSV and (optionally) the flagged CSV files.
    """
    # Ensure the PARSING_RESULT_FOLDER exists
    if not os.path.exists(config.PARSING_RESULT_FOLDER):
        os.makedirs(config.PARSING_RESULT_FOLDER)

    try:
        # Define file paths for the final output and flagged documents
        timestamp = dt.now().strftime('%y%m%d_%H%M%S')

        # Handle final_df by splitting if it exceeds max_lines
        total_rows = len(final_df)
        num_chunks = (total_rows // max_lines) + 1
        final_csv_paths = []

        # Only keep the necessary columns in final_df
        # These are the columns we are interested in for the final CSV
        final_df = final_df[['date', 'docid', 'stateb', 'disno', 'title', 'response_text', 'speaker']]

        # Split final_df into chunks and save each chunk into a separate file
        for i in range(num_chunks):
            start_idx = i * max_lines
            end_idx = min((i + 1) * max_lines, total_rows)
            final_chunk = final_df.iloc[start_idx:end_idx]

            # Create a file name for each chunk
            final_csv_path = os.path.join(config.PARSING_RESULT_FOLDER,
                                          f"final_comm_acts_log_{timestamp}_{i + 1}.csv")
            final_chunk.to_csv(final_csv_path, index=False)
            final_csv_paths.append(final_csv_path)
            log_message(log_filename, "parsing_module/write_final_output",
                        f"Final CSV part {i + 1} created: {final_csv_path}")

            # Handle flagged_df if it exists
            flagged_csv_path = None
            if flagged_df is not None and not flagged_df.empty:
                flagged_csv_path = os.path.join(config.PARSING_RESULT_FOLDER, f"flagged_documents_log_{timestamp}.csv")
                flagged_df.to_csv(flagged_csv_path, index=False)
                log_message(log_filename, "parsing_module/write_final_output",
                            f"Flagged CSV created: {flagged_csv_path}")
            else:
                log_message(log_filename, "parsing_module/write_final_output", "No flagged documents CSV created.")

            return final_csv_paths, flagged_csv_path

    except Exception as e:
        log_message(log_filename, "parsing_module/write_final_output", f"Error writing final CSV: {e}")
        return None, None
