# config.py
from pathlib import Path

# Default settings (updated via setup_config.py)
ROOT_FOLDER = Path('C:/Users/tomc9/Dropbox/0.Research/AggregationProblem')
SCRAPE_RESULT_FOLDER = ROOT_FOLDER / 'scraped'
PARSING_RESULT_FOLDER = ROOT_FOLDER / 'data'
DICTIONARY_CSV = ROOT_FOLDER / 'dictionary.csv'

# OpenAI API
OPENAI_API_KEY = ''
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# DO NOT RELEASE WITH KEY

OPENAI_API_MODEL = 'gpt-4o-mini'
# Models:
# GPT-4o: 'gpt-4o-2024-08-06' (16384 token limit)
# GPT-4o mini: 'gpt-4o-mini' (16384 token limit)
OPENAI_PROMPT = """You will receive a block of text that may not have clear paragraph breaks. Your task is to split this text into meaningful paragraphs, focusing on speaker changes.

Guidelines:
- Prioritize splitting where a new speaker begins speaking.
- If there are no speaker changes, split based on logical content.
- Avoid creating very short paragraphs. Each paragraph should have at least several sentences unless there is a clear speaker or topic change.
- Do not treat short interjections (e.g., "Yes," "No," brief questions) as a new speaker unless they significantly interrupt the flow.
- Return the output as a JSON object where the first term is the paragraph and the second term is the speaker.
- Do not include metadata (e.g., headers, dates, footnotes, participant lists, introductory information).
- Ignore signatures or closing remarks.

Example Input:
Memorandum of Telephone Conversations by the Acting Secretary of State  
[Washington,] February 20, 1945.  
Subjects: Invitations for United Nations Conference; Announcement of Voting Procedure  
Participants: Mr. Alger Hiss; Acting Secretary, Mr. Grew  
Mr. Grew said, "I telephoned Mr. Alger Hiss at Mexico City and said that with regard to the question of issuing invitations..." Mr. Grew added that the Department had inquired about a list of participants.  
Mr. Hiss responded, "The Secretary wanted me to call Ambassador Caffery in Paris as soon as possible..."  
Later, Mr. Hiss called back to continue the discussion. He said, "With regard to France, the Secretary was glad that his message had gone straight through..."  
Mr. Hiss continued, shifting to the topic of the upcoming conference: "I also wanted to discuss the agenda for the United Nations Conference at San Francisco..." Mr. Grew said it was a good idea to handle that first. Mr. Hiss agreed, saying it was wise to make good use of time while they were here.

Example Output:
[
  {
    "text": "Mr. Grew said, \\"I telephoned Mr. Alger Hiss at Mexico City and said that with regard to the question of issuing invitations...\\" Mr. Grew added that the Department had inquired about a list of participants.",
    "speaker": "Acting Secretary Grew"
  },
  {
    "text": "Mr. Hiss responded, \\"The Secretary wanted me to call Ambassador Caffery in Paris as soon as possible...\\" Later, Mr. Hiss called back to continue the discussion. He said, \\"With regard to France, the Secretary was glad that his message had gone straight through...\\" Mr. Hiss continued, shifting to the topic of the upcoming conference: \\"I also wanted to discuss the agenda for the United Nations Conference at San Francisco...\\" Mr. Grew said it was a good idea to handle that first. Mr. Hiss agreed, saying it was wise to make good use of time while they were here.",
    "speaker": "Alger Hiss"
  }
]

Now, analyze the following block of text and split it into meaningful paragraphs following the rules above:
"""
OPENAI_TEMP = 0.2
OPENAI_TOKEN_LIMIT = 16384
OPENAI_TOP_P = 1

# OpenAI API parallel processing related parameters
RATE_LIMIT_TOKENS_PER_MINUTE = 4000000
RATE_LIMIT_REQUESTS_PER_MINUTE = 5000
TOKENS_PER_REQUEST = 5120  # Approximate tokens per request (to be updated based on actual usage)

MAX_WORKERS = 100  # Max parallel work at once, adjust according to OpenAI API rate limit
MAX_RETRIES = 5  # Max retries for exponential backoff
INITIAL_BACKOFF = 1  # Initial backoff in seconds

# Consensus-related parameters
MIN_PROMPT_CALLS = 1  # Minimum times to call the API per document
MAX_PROMPT_CALLS = 1  # Maximum times to call the API for additional consensus runs
DESIRED_CONSENSUS = 0.8  # Desired consensus threshold (e.g., 0.8 means 80% agreement needed)
CONSENSUS_TOLERANCE = 30

# User-Agent string for requests
USER_AGENT = 'Mozilla/5.0 (compatible; FRUSCrawler/2.0; +https://github.com/khyrst; khyrst@korea.ac.kr)'

# Other configuration settings
DATABASE_URL = 'sqlite:///my_database.db'
LOGGING_LEVEL = 'INFO'
EXCEL_CELL_LIMIT = 32000
EXCEL_MAX_ROWS = 1000000

# Debug mode flag
DEBUG_MODE = False

# Log Settings
LOGS_FOLDER = ROOT_FOLDER / 'logs'
LOG_FILENAME = "log_"


# Pre-check for API
if OPENAI_API_KEY:
    print("API Key successfully loaded")
else:
    print("API Key not found")


# Other prompts:

# Complicated one
"""
You will receive a block of text containing multiple paragraphs. Your task is to identify the exact character positions (relative to the input text) where either:
1. A new speaker begins speaking.
2. The same speaker transitions to a new subject or topic.

**Guidelines**:
- Prioritize grouping content into **paragraph-sized chunks** where possible, instead of splitting text into smaller fragments.
- Avoid marking **short, interjected statements** (e.g., "Yes," "No," or single questions) as speaker changes unless they constitute a meaningful interruption.
- **Do not create segments shorter than 20 characters.** If a segment would be too short, group it with surrounding text to form a longer section.
- Focus on **speaker transitions**, but also account for **subject changes** if they lead to a meaningful shift in the conversation.
- **Ignore metadata** such as headers, footnotes, dates, participant lists, and introductory information.
- **Ignore signatures** at the end of the document.
- **Footnotes should not be included** in the identified segments.
- **Do not mark** repeated statements by the same speaker unless the topic changes or another speaker intervenes.
- If multiple people are speaking in short bursts, group their dialogue under the main speaker until a meaningful transition occurs.

**Output Format**:
- Return a JSON array of arrays, where each array contains the start index, end index, and speaker’s name.
- Segments should be based on meaningful content or speaker transitions. Avoid creating excessively small or minimal segments.
- Example format:
  ```json
  [
    {"start": 138, "end": 587, "speaker": "Mr. Grew"},
    {"start": 588, "end": 1200, "speaker": "Mr. Hiss"}
  ]
"""

# Another one, leaner
"""
You will receive a block of text that may not have clearly defined paragraphs. Your task is to split this text into meaningful paragraphs, with a focus on identifying where a new speaker begins speaking.

Guidelines:
- Prioritize splitting the text at points where speakers change.
- If there are no clear speaker transitions, split the text into logical paragraphs based on content.
- Avoid creating overly small chunks. Each paragraph should contain at least a few sentences unless there is a speaker change that necessitates a shorter paragraph.

Output Format:
- Return a JSON array, where each object contains the start index, end index, and speaker’s name (if identifiable).
- If the speaker is unknown, mark the speaker as "Unknown".

Example Input:
Memorandum of Telephone Conversations by the Acting Secretary of State  
[Washington,] February 20, 1945.  
Subjects: Invitations for United Nations Conference;  
Announcement of Voting Procedure  
Participants: Mr. Alger Hiss; Acting Secretary, Mr. Grew

Mr. Grew said, "I telephoned Mr. Alger Hiss at Mexico City and said that with regard to the question of issuing invitations..."  
Mr. Hiss responded, "The Secretary wanted me to call Ambassador Caffery in Paris as soon as possible..."  
Later, Mr. Hiss called back to continue the discussion. He said, "With regard to France, the Secretary was glad that his message had gone straight through..."  
Mr. Hiss continued, shifting to the topic of the upcoming conference: "I also wanted to discuss the agenda for the United Nations Conference at San Francisco..."

Example:
[
    {"start": 138, "end": 287, "speaker": "Alger Grew"},
    {"start": 288, "end": 522, "speaker": "Secretary Hiss"}
]

Now, analyze the following block of text and split it into meaningful paragraphs, following the rules above:
"""