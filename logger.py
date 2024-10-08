# logger.py

import os
from datetime import datetime
import config


def create_log_file():
    """
    Creates a log file in the logs directory with the current date and time in the filename.
    """
    # Ensure the logs folder exists
    if not os.path.exists(config.LOGS_FOLDER):
        os.makedirs(config.LOGS_FOLDER)

    # Generate the log filename with timestamp
    timestamp = datetime.now().strftime('%y%m%d_%H%M%S')
    log_filename = os.path.join(config.LOGS_FOLDER, f'log_{timestamp}.txt')

    # Create the log file
    with open(log_filename, 'w') as log_file:
        log_file.write(f"Log created at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("="*40 + "\n")

    return log_filename


def log_message(log_filename, function_name, message):
    """
    Logs a message to both a log file and the console.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} [{function_name}] {message}"

    # Print to console
    print(log_entry)

    # Write to log file
    if log_filename:
        with open(log_filename, 'a', encoding='utf-8') as log_file:
            log_file.write(log_entry + '\n')
