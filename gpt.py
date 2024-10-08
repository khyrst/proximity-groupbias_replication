import config
from logger import log_message

from openai import OpenAI
import json
import collections
import tiktoken
import numpy as np
import httpx
import time

MAX_RETRIES = config.MAX_RETRIES
INITIAL_BACKOFF = config.INITIAL_BACKOFF
RATE_LIMIT_RPM = config.RATE_LIMIT_REQUESTS_PER_MINUTE
RATE_LIMIT_TPR = config.TOKENS_PER_REQUEST

client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_token_count(text, model=config.OPENAI_API_KEY):
    # Unused in main; for debugging / console
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_text(text, model=config.OPENAI_API_MODEL, max_tokens=config.OPENAI_TOKEN_LIMIT, reserved_tokens=config.TOKENS_PER_REQUEST):
    """
    Splits the text into chunks that fit within the model's token limit,
    ensuring space is left for the response tokens.

    Args:
        text (str): The input text to be split into chunks.
        model (str): The OpenAI model being used (for token encoding).
        max_tokens (int): Maximum number of tokens allowed by the model.
        reserved_tokens (int): Number of tokens reserved for the model's response.

    Yields:
        str: Chunks of text that fit within the token limit.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    # Adjust the max tokens for the input text, reserving space for output tokens
    available_tokens = max_tokens - reserved_tokens

    # Split the tokens into chunks
    for i in range(0, len(tokens), available_tokens):
        chunk = tokens[i:i + available_tokens]
        yield encoding.decode(chunk)  # Convert tokens back to text for each chunk


def call_openai_api(text, prompt, log_filename=None):
    """
    Calls the OpenAI API to process the text and return speaker transitions.
    If the text exceeds the token limit, it will be split into chunks and processed separately. (Handled by chunk_text()
    """
    if prompt is None:
        prompt = config.OPENAI_PROMPT

    try:
        log_message(log_filename, "call_openai_api",
                    f"Making OpenAI API call with text length: {len(text)} characters")

        # Split the text into manageable chunks if it exceeds token limits
        chunks = list(chunk_text(text, model=config.OPENAI_API_MODEL, max_tokens=config.OPENAI_TOKEN_LIMIT))

        all_texts = []
        all_speakers = []

        responses = []
        for chunk in chunks:
            # OpenAI API Call
            response = client.chat.completions.create(
                model=config.OPENAI_API_MODEL,
                messages=[
                    {"role": "system","content": prompt},
                    {"role": "user","content": chunk}
                ],
                temperature=config.OPENAI_TEMP,
                max_tokens=config.OPENAI_TOKEN_LIMIT,  # adjust based on the response size expected
                top_p= config.OPENAI_TOP_P,
                frequency_penalty=0,
                presence_penalty=0,
                response_format={"type": "text"}
            )

            log_message(log_filename, "gpt/call_openai_api",
                        f"Full response: {response}")

            output = response.choices[0].message.content.strip()
            log_message(log_filename, "gpt/call_openai_api",
                        f"Raw response from OpenAI: {output}")

            # Check if the response is wrapped with '''json or similar and remove it
            if output.startswith("```json"):
                output = output.replace("```json", "").replace("```", "").strip()

            try:
                # Parse the chunk's output using the parse_openai_output function
                texts, speakers = parse_openai_output(output)

                if texts and speakers:
                    # Append the texts and speakers from this chunk to the full response
                    all_texts.extend(texts)
                    all_speakers.extend(speakers)

            except json.JSONDecodeError as e:
                log_message(log_filename, "gpt/call_openai_api",
                            f"Error parsing JSON output: {e}. Response: {output}")
                # Optionally, retry the API call or handle it based on severity
                continue

        # At this point, all texts and speakers from the chunks should be combined into lists
        return all_texts, all_speakers

    except Exception as e:
        log_message(log_filename, "gpt/call_openai_api",
                    f"Error with OpenAI API call: {str(e)}")
        return None, None


def call_openai_api_with_retries(text, log_file, max_retries=config.MAX_RETRIES, initial_delay=config.INITIAL_BACKOFF):
    """
    Call the OpenAI API with retries and exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            # API call via connection pooling
            response_text, speaker = call_openai_api(text, config.OPENAI_PROMPT, log_file)
            if response_text is not None:
                return response_text, speaker  # Success

        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            # Log the error and implement exponential backoff
            wait_time = initial_delay * (2 ** attempt)
            log_message(log_file, "gpt/call_openai_api_with_retries",
                        f"API call failed ({e}). Retrying in {wait_time} seconds...")
            time.sleep(wait_time)  # Wait before retrying

        except Exception as e:
            # For any other type of error, we log and break out without retries
            log_message(log_file, "gpt/call_openai_api_with_retries",
                        f"Unrecoverable error: {e}")
            return None, None

    # If all retries fail
    log_message(log_file, "gpt/call_openai_api_with_retries",
                f"Max retries reached for text: {text}.")
    return None, None


def parse_openai_output(output):
    """
    Parses the output from OpenAI API and returns list of indexes and speakers.
    Handles JSON format with 'start', 'end', and 'speaker' keys.
    """
    try:
        # Parse the output as JSON
        parsed_output = json.loads(output)

        # Extract the text and speakers
        texts = [item['text'] for item in parsed_output]
        speakers = [item['speaker'] for item in parsed_output]

        return texts, speakers
    except (json.JSONDecodeError, KeyError) as e:
        log_message(None, "gpt/parse_openai_output",
                    f"Error parsing JSON output: {str(e)}")
        return None, None

    # Deprecated legacy code, kept just in case
    # Not done as version control because 1. I'm not too good with it yet and 2. I'm not in the mood to figure that out
    # v1
    # try:
    #     # Logs raw output to debug format issues
    #     log_message(None, "gpt/parse_openai_output", f"Raw output: {output}")
    #
    #     parsed_output = json.loads(output)  # Parse the JSON output
    #
    #     # Check if "transitions" key exists in the response
    #     if "transitions" not in parsed_output:
    #         raise KeyError(f"'transitions' key missing in response: {parsed_output}")
    #
    #     transitions = parsed_output["transitions"]
    #     indexes = [(item['start'], item['end']) for item in parsed_output]  # Extract start and end indexes
    #     speakers = [item['speaker'] for item in parsed_output]  # Extract speakers
    #
    #     # Extract the start, end, and speaker values
    #     for item in transitions:
    #         if 'start' in item and 'end' in item and 'speaker' in item:
    #             indexes.append([item['start'], item['end']])
    #             speakers.append(item['speaker'])
    #         else:
    #             raise KeyError(f"Missing 'start', 'end', or 'speaker' in item: {item}")
    #
    #     return indexes, speakers
    # v2
    # try:
    #     # Parse the output as JSON
    #     parsed_output = json.loads(output)
    #
    #     # Extract the indexes and speakers
    #     indexes = [[item['start'], item['end']] for item in parsed_output]
    #     speakers = [item['speaker'] for item in parsed_output]
    #
    #     return indexes, speakers


def get_consensus_results(text, log_filename=None, tolerance=config.CONSENSUS_TOLERANCE, mode="detailed"):
    """
    Calls the OpenAI API multiple times and logs the API response.
    Based on the logged results, calculates a "consensus score," denoting how many of the API calls were in agreement wih each other.
    Desired consensus is set under config.py (default = 0.8)
    Returns: tuple (results_list, consensus_score).
    """

    results = []

    # Call the API a minimum number of times (config.MIN_PROMPT_CALLS)
    for _ in range(config.MIN_PROMPT_CALLS):
        response_text, speaker = call_openai_api_with_retries(text, log_filename)
        if response_text and speaker:
            results.append({"text": response_text, "speaker": speaker})

    # Check for inconsistent number of segments before computing consensus
    if not all(len(result['text']) == len(results[0]['text']) for result in results):
        log_message(log_filename, "gpt/get_consensus_results",
                    "Inconsistent number of segments. Proceeding with leniency.")
        # Proceed with best attempt by using the majority length
        segment_length_counts = [len(result['text']) for result in results]
        most_common_length = max(set(segment_length_counts), key=segment_length_counts.count)
        results = [result for result in results if len(result['text']) == most_common_length]

    # Calculate initial consensus score based on the minimum number of calls
    consensus_score = compute_consensus_lse(results, tolerance=tolerance, mode=mode)

    # If initial consensus is reached, return results early
    if consensus_score >= config.DESIRED_CONSENSUS:
        return results, consensus_score

    # If not, let's first determine the number of remaining API calls
    remaining_calls = config.MAX_PROMPT_CALLS - config.MIN_PROMPT_CALLS

    # If min & max are equal (i.e., no prompt calls remain), return the initial consensus score
    if remaining_calls <= 0:
        return results, consensus_score

    # Continue API calls if consensus is not reached, but check if reaching consensus is possible
    for i in range(remaining_calls):
        # Count how many responses fit within the tolerance
        valid_responses = sum(
            compute_consensus_lse(results[:n], tolerance=tolerance, mode=mode) >= tolerance
            for n in range(1, len(results))
        )
        # Check if it is mathematically possible to reach the desired consensus
        # For example, if you need 8 out of 10 and already have fewer than 8 valid responses with limited calls remaining
        required_valid_responses = config.DESIRED_CONSENSUS * (config.MIN_PROMPT_CALLS + i + 1)
        if valid_responses < required_valid_responses and remaining_calls - i < required_valid_responses - valid_responses:
            # If mathematically impossible to reach consensus, stop early
            log_message(log_filename, "gpt/get_consensus_results",
                        "Mathematically impossible to reach desired consensus.")
            return results, consensus_score

        # Make additional API calls
        response_text, speaker = call_openai_api(text, config.OPENAI_PROMPT, log_filename)
        if response_text and speaker:
            results.append({"text": response_text, "speaker": speaker})

        # Recompute consensus score after adding new results
        consensus_score = compute_consensus_lse(results, tolerance=tolerance, mode=mode)

        # If consensus is reached, return results
        if consensus_score >= config.DESIRED_CONSENSUS:
            return results, consensus_score

    # If consensus is not reached, return the current results and the score
    log_message(log_filename, "gpt/get_consensus_results",
                "No consensus reached after maximum API calls.")
    return results, consensus_score


def get_consensus_results_with_retries(text, log_file):
    """Wrapper to handle retries and API request with connection pooling"""
    response_text, speaker = call_openai_api_with_retries(text, log_file)
    if response_text and speaker:
        # Process consensus logic if response is successful
        api_results, consensus_score = compute_consensus_lse([response_text], tolerance=config.CONSENSUS_TOLERANCE)
        return api_results, consensus_score
    return [], 0


def compute_consensus_lse(results, tolerance=config.CONSENSUS_TOLERANCE, mode="detailed"):
    """
    Computes consensus score based on the least squares error of text segment lengths.
    Takes the average length of each segment, calculates LSE, and checks against tolerance.
    """
    # Extract text segments from results
    all_segments = [result['text'] for result in results]

    # Ensure all results have the same number of segments (fail early otherwise)
    if not all(len(segments) == len(all_segments[0]) for segments in all_segments):
        return 0.0  # No consensus if lengths of segments don't match

    # Detailed mode: Transpose the list to group segments by position
    transposed_segments = list(zip(*all_segments))

    if len(transposed_segments) == 0:
        log_message(None, "gpt/compute_consensus_lse",
                    "No valid segments to compare, returning 0 consensus.")
        return 0.0  # No consensus if there are no segments

    # Simple mode: Compare overall lengths of the full text responses
    if mode == "simple":
        overall_lengths = [len(" ".join(segments)) for segments in all_segments]
        avg_length = np.mean(overall_lengths)
        squared_errors = [(length - avg_length) ** 2 for length in overall_lengths]
        rmse = np.sqrt(np.mean(squared_errors))
        return 1.0 if rmse <= tolerance else 0.0

    total_error = 0
    for segment_group in transposed_segments:
        # Compute the average length of the group
        lengths = [len(segment) for segment in segment_group]
        avg_length = np.mean(lengths)

        # Calculate least square error for this group
        squared_errors = [(len(segment) - avg_length) ** 2 for segment in segment_group]
        root_mean_squared_error = np.sqrt(np.mean(squared_errors))

        # Accumulate the error
        total_error += root_mean_squared_error

    # Average error over all segments
    average_error = total_error / len(transposed_segments)

    # If the average error is within the tolerance, consider it as consensus
    consensus_score = 1.0 if average_error <= tolerance else 0.0

    return consensus_score


def majority_vote(results):
    # Deprecated, kept for legacy
    """
    Takes the results of multiple API calls and applies majority voting on indexes and speakers.
    Only processes results with matching lengths.
    """
    indexes_list = [result[0] for result in results]
    speakers_list = [result[1] for result in results]

    # Ensure consistent lengths before applying majority vote
    if not all(len(x) == len(indexes_list[0]) for x in indexes_list):
        return None, None

    # Majority vote for indexes (start, end) tuples
    all_indexes = list(zip(*indexes_list))  # Group by position across results
    final_indexes = []

    for index_group in all_indexes:
        count_index = collections.Counter([tuple(idx) for idx in index_group])  # Convert lists to tuples for counting
        most_common_index, count = count_index.most_common(1)[0]  # Get the most common (start, end) pair
        final_indexes.append(list(most_common_index))  # Append the tuple back as a list

    # Majority vote for speakers
    all_speakers = list(zip(*speakers_list))  # Group by position across results
    final_speakers = []

    for speaker_group in all_speakers:
        count_speaker = collections.Counter(speaker_group)
        most_common_speaker, count = count_speaker.most_common(1)[0]  # Get the most common speaker
        final_speakers.append(most_common_speaker)

    return final_indexes, final_speakers


def compute_consensus_score(results, tolerance=5):
    # Deprecated, kept for legacy
    """
    Computes a consensus score based on how often the same indexes and speakers appear.
    """
    indexes_list = [result[0] for result in results]
    speakers_list = [result[1] for result in results]

    # Check if lengths match before proceeding
    if not all(len(a) == len(indexes_list[0]) for a in indexes_list):
        return 0.0  # No consensus if lengths don't match

    # Calculate index agreement using tolerance
    index_consensus = np.mean([
        np.all(np.abs(np.array(a) - np.array(b)) <= tolerance)
        for a, b in zip(indexes_list, indexes_list[1:])
    ])

    # Calculate speaker agreement
    speaker_consensus = np.mean([a == b for a, b in zip(speakers_list, speakers_list[1:])])

    return (index_consensus + speaker_consensus) / 2.0
