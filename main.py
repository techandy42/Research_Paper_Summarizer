import feedparser
from urllib.parse import quote_plus
import requests
import PyPDF2
import os
import shutil
import openai
import tiktoken
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

INSTRUCTION = """
Summerize the following research paper:
1. The resulting summary should be in concise, easy-to-read markdown format.
2. The resulting summary should have the title of the research paper as the header.
3. The resulting summary should be 5-7 point-form sentences.
4. The resulting summary should cover all the important topics.
5. Do not include any text other than the summary.

Research Paper:
<<CONTENT>>
"""

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete all files in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def extract_content_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

start = 0
max_results = 5

# Ask user for the query string
search_query = input("Enter your search query: ")
encoded_search_query = quote_plus(search_query)

query = f"http://export.arxiv.org/api/query?search_query={encoded_search_query}&start={start}&max_results={max_results}"

# Use feedparser to fetch results
feed = feedparser.parse(query)

# Create folders if they don't exist
papers_folder = 'papers'
extract_folder = 'extract'
summary_folder = 'summary'
os.makedirs(papers_folder, exist_ok=True)
os.makedirs(extract_folder, exist_ok=True)
os.makedirs(summary_folder, exist_ok=True)

# Delete files in /papers and /extract folders
delete_files_in_folder('papers')
delete_files_in_folder('extract')
delete_files_in_folder('summary')

# Download PDFs, extract content
for entry in feed.entries:
    print(f"Downloading PDF for {entry.title}...")
    # Extract the PDF link
    pdf_link = entry.link.replace('abs', 'pdf')
    
    # Download PDF
    response = requests.get(pdf_link)
    file_path = os.path.join(papers_folder, entry.title.replace('/', '-') + '.pdf')
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"Extracting content from {entry.title}...")
    # Extract content
    content = extract_content_from_pdf(file_path)
    
    # Save content to text file in extract folder
    txt_file_path = os.path.join(extract_folder, entry.title.replace('/', '-') + '.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(entry.title + '\n' + content)

    print(f"Summarizing {entry.title}...")

    # half the content until the token size is equal or less than 16385
    while num_tokens_from_string(content, "cl100k_base") > 16385:
        content = content[:len(content)//2]

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k", messages=[{"role": "user", "content": INSTRUCTION.replace('<<CONTENT>>', content)}])
    summary = chat_completion.choices[0].message.content

    # Save summary to markdown file in summary folder
    summary_file_path = os.path.join(summary_folder, entry.title.replace('/', '-') + '.md')
    with open(summary_file_path, 'w', encoding='utf-8') as file:
        file.write(summary)

print("All PDFs downloaded and content extracted successfully.")
