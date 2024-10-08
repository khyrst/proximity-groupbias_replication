# scraper.py
import os
import requests
import logging
import re
import time
from lxml import html
from urllib.parse import urljoin
from tqdm import tqdm
from random import randint
import config

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def download_file(url, destination, destination_folder_path):
    full_path = os.path.join(destination_folder_path, destination)
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()  # Ensures we notice bad responses
    with open(full_path, 'wb') as file:
        file.write(response.content)


def get_links_from_hub(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    tree = html.fromstring(response.content)
    links = tree.xpath('//a[@data-template="app:parse-params"]/@href')
    return [link for link in links if isinstance(link, str)]  # Ensures only strings (URLs) are returned.


def get_download_link(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    tree = html.fromstring(response.content)
    download_links = tree.xpath('//a[@data-template="frus:epub-href-attribute"]/@href')
    return download_links[0] if download_links else None


def get_document_title(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    response.raise_for_status()
    tree = html.fromstring(response.content)
    title_elements = tree.xpath('/html/body/div[1]/section/div/main/div/div[2]/div[1]/div/h1/text()')  # Tailored for FRUS
    if not title_elements:
        return None
    # Join elements, replace multiple spaces with a single space
    title = ''.join(title_elements).strip()
    title = re.sub(r'\W\s+', ' ', title)

    # Replaces specific string with "FRUS"
    title = title.replace("Foreign Relations of the United States", "FRUS")

    # Replaces any non-alphanumeric and non-dash characters (excluding underscore and space) with a single dash
    title = re.sub(r'[^a-zA-Z0-9_ \-]', '-', title)

    # Replaces 3 or more dashes (---) with single dash
    title = re.sub(r'-{3,}', '-', title)

    # Replaces spaces with underscores
    title_with_underscores = title.replace(' ', '_')

    # Truncates underscored title to manageable length
    if len(title_with_underscores) > 80:
        front_part = title_with_underscores[:50]
        # Finds the next underscore to cut off at the next word
        next_underscore = front_part.rfind('_')
        if next_underscore != -1:
            front_part = front_part[:next_underscore]
        end_underscore = title_with_underscores[:-30].rfind('_')
        back_part = title_with_underscores[end_underscore + 1:]
        title_with_underscores = f"{front_part}...{back_part}"

    # Extracts the first occurrence of four consecutive digits
    year_match = re.search(r'\d{4}', title)
    year = year_match.group(0) if year_match else ''

    # Prepends year to the title, if found
    formatted_title = f'{year}_{title_with_underscores}' if year else title_with_underscores
    return formatted_title


def get_downloaded_files(destination_fp):
    return {file for file in os.listdir(destination_fp) if
            os.path.isfile(os.path.join(destination_fp, file))}


def scrape_and_download(url, destination_folder_path, file_extension='epub', start_year=1945, end_year=1985,
                        test_mode=False):
    """Scrape and download files from a given URL."""
    document_links = get_links_from_hub(url)
    if test_mode:
        document_links = document_links[:3]
    if not document_links:
        logging.warning(f"No documents found for URL: {url}")
        return

    with tqdm(total=len(document_links), desc="Downloading", unit="file") as pbar:
        for link in document_links:
            document_link = urljoin(url, link)
            document_title = get_document_title(document_link)
            if not document_title:
                logging.warning(f"Could not retrieve title for document: {document_link}")
                pbar.update(1)
                continue

            should_download = True
            year_match = re.search(r'\d{4}', document_title)
            if year_match:
                year = int(year_match.group(0))
                if year < start_year or year > end_year:
                    should_download = False
                    if test_mode:
                        logging.debug(f"Found year: {year}, skipping...")

            if should_download:
                download_link = get_download_link(document_link)
                if download_link and document_title:
                    absolute_download_link = urljoin(document_link, download_link)
                    file_name = f'{document_title}.{file_extension}'
                    if file_name in get_downloaded_files(destination_folder_path):
                        pbar.update(1)
                        if test_mode:
                            logging.debug(f"File already downloaded: {file_name}")
                        continue
                    if test_mode:
                        logging.debug(f"Downloading...")
                    download_file(absolute_download_link, file_name, destination_folder_path)
                    time.sleep(randint(1, 5))  # Delay to avoid overloading the server
            pbar.update(1)
            time.sleep(randint(1, 5))  # Delay to avoid overloading the server


def process_urls_from_file(destination_folder_path, file_extension='epub', start_year=1945, end_year=1985,
                           test_mode=False):
    """Process URLs from a file and scrape and download the documents."""
    url_list_path = os.path.join(config.ROOT_FOLDER, 'url_list.txt')

    if not os.path.exists(url_list_path):
        raise FileNotFoundError(f"{url_list_path} does not exist. Please create this file with URLs to scrape.")

    with open(url_list_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("url_list.txt is empty. Please add URLs to the file before proceeding.")
        return

    for url in urls:
        scrape_and_download(url, destination_folder_path, file_extension, start_year, end_year, test_mode)
