import requests
import xml.etree.ElementTree as ET
import os
import json
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
metadata_dir = os.path.join(base_dir, 'data', 'metadata')
pdf_dir = os.path.join(base_dir, 'data', 'pdfs')
os.makedirs(metadata_dir, exist_ok=True)
os.makedirs(pdf_dir, exist_ok=True)


def search_arxiv(query, start=0, max_results=5,sortBY='relevance'):
    
    arxiv_base_url = "http://export.arxiv.org/api/query?"
    url = f"{arxiv_base_url}search_query={query}&start={start}&max_results={max_results}&sortBy={sortBY}&sortOrder=descending"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        return root
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
    return None


def sanitize_filename(name):
    return re.sub(r'[^\w\-_. ]', '_', name)


def save_metadata_and_pdf(entry):
    """Save metadata and download the PDF"""
    title = entry.find('{http://www.w3.org/2005/Atom}title').text
    authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
    summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
    published = entry.find('{http://www.w3.org/2005/Atom}published').text
    pdf_url = entry.find('{http://www.w3.org/2005/Atom}id').text.replace('abs', 'pdf') + ".pdf"
    arxiv_id = entry.find('{http://www.w3.org/2005/Atom}id').text.split('/')[-1]

    print(f"\nTitle: {title}")
    print(f"Authors: {', '.join(authors)}")
    print(f"Published: {published}")
    print(f"Summary: {summary}")
    print(f"PDF URL: {pdf_url}")
   

    metadata = {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "published": published,
        "summary": summary,
        "pdf_url": pdf_url
    }

    
    metadata_filename = os.path.join(metadata_dir, f"{sanitize_filename(arxiv_id)}.json")
    with open(metadata_filename, 'w', encoding='utf-8') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

    try:
        pdf_response = requests.get(pdf_url, timeout=15)
        pdf_response.raise_for_status()
        pdf_filename = os.path.join(pdf_dir, f"{sanitize_filename(arxiv_id)}.pdf")
        with open(pdf_filename, 'wb') as pdf_file:
            pdf_file.write(pdf_response.content)
        print(f" Saved metadata and PDF for {arxiv_id}")
    except requests.RequestException as e:
        print(f"Failed to download PDF: {e}")
    
    print("/n")


def main():
    query = input("Enter search query: ")
    root = search_arxiv(query,sortBY='relevance')

    if root is None:
        print("No results found.")
        return

    entries = root.findall('{http://www.w3.org/2005/Atom}entry')
    if not entries:
        print("No papers found for your query.")
        return

    print(f"\nFound {len(entries)} papers:\n")
    for idx, entry in enumerate(entries, start=1):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        print(f"{idx}. {title}\n   Authors: {', '.join(authors)}\n   Published: {published}\n")

    paper_choices = input("Enter the number(s) of the paper(s) you want to download (e.g., 1 3 5), or 'q' to quit: ").strip()

    if paper_choices.lower() == 'q':
        print("Exiting...")
        return

    try:
        indices = [int(i) - 1 for i in paper_choices.split()]
        for i in indices:
            if 0 <= i < len(entries):
                save_metadata_and_pdf(entries[i])
            else:
                print(f"Invalid paper number: {i+1}")
    except ValueError:
        print("Invalid input. Please enter valid numbers or 'q'.")


if __name__ == "__main__":
    main()
