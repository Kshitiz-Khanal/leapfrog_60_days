import json
import re
import textwrap
import warnings
import logging
import os
# from model.scraper import Scrappping
from pathlib import Path
from langchain.schema import Document
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from IPython.display import Markdown, display

from datetime import datetime
# Suppress warnings
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Configure Gemini API key
# Replace with your real key
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)


# Initialize the Gemini text model
text_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Initialize LangChain LLM wrapper
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True,
)


class RAGPipeline:
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path
        self.vector_index = None
        self.qa_chain = None
        self.full_text = None

    def load_and_process_documents(self, pdf_path=None):
        path_to_use = pdf_path if pdf_path else self.pdf_path
        if not path_to_use:
            raise ValueError("No PDF path provided")

        try:
            print(f"Loading PDF from {path_to_use}...")
            loader = PyPDFLoader(path_to_use)
            pages = loader.load_and_split()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=1000
            )

            self.full_text = "\n\n".join(str(p.page_content) for p in pages)
            texts = text_splitter.split_text(self.full_text)

            print("Creating embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )

            self.vector_index = FAISS.from_texts(
                texts, embeddings).as_retriever(search_kwargs={"k": 5})
            print("Document processing complete!")
            return True
        except Exception as e:
            print(f"Error loading documents: {e}")
            return False

    def _create_summary_chain(self):
        summary_template = """Generate an EXTREMELY DETAILED structured summary of this document with these EXACT sections:

1. **OVERVIEW** - Provide a comprehensive 3-4 paragraph summary covering:
   - Main purpose and objectives
   - Core thesis or argument
   - Key context and background
   - Overall significance

2. **KEY FINDINGS** - List ALL important findings as:
   - Detailed bullet points (10-15 items)
   - Include specific data points where available
   - Note any surprising or counterintuitive results

3. **METHODOLOGIES** - Describe ALL approaches used:
   - Research methods and techniques
   - Data collection procedures
   - Analysis frameworks
   - Any innovative methodologies

4. **RECOMMENDATIONS** - Provide complete suggested actions:
   - Immediate next steps
   - Long-term proposals
   - Policy implications
   - Future research directions

Document:
{text}
"""
        prompt = PromptTemplate.from_template(summary_template)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="text"
        )

    def generate_structured_summary(self):
        if not self.full_text:
            print("Error: Document not loaded")
            return None

        try:
            docs = [Document(page_content=self.full_text[:50000])]
            summary_chain = self._create_summary_chain()
            full_summary = summary_chain.run(docs)

            # Parse detailed summary sections
            sections = {
                "overview": "",
                "key_findings": "",
                "methodologies": "",
                "recommendations": ""
            }

            section_pattern = re.compile(
                r'\d+\.\s*\*\*(OVERVIEW|KEY FINDINGS|METHODOLOGIES|RECOMMENDATIONS)\*\*[-\s]*(.*?)(?=\d+\.\s*\*\*|\Z)',
                re.DOTALL | re.IGNORECASE
            )
            matches = section_pattern.finditer(full_summary)
            for match in matches:
                section_name = match.group(1).lower().replace(' ', '_')
                content = match.group(2).strip()
                if section_name in sections:
                    sections[section_name] = content

            # Fill missing sections using the base model
            for section in sections:
                if not sections[section]:
                    detail_prompt = f"""Generate an EXTREMELY DETAILED {section.replace('_', ' ')} section for this document.
Include all relevant details, examples, and specific information. Document excerpt:\n\n{self.full_text[:20000]}"""
                    response = text_model.generate_content(detail_prompt)
                    sections[section] = response.text

            # Enhance short sections
            for section, content in sections.items():
                if len(content.split()) < 100:
                    enhancement_prompt = f"Expand this {section} section with more details, examples, and analysis:\n\n{content}"
                    enhanced = text_model.generate_content(enhancement_prompt)
                    sections[section] = enhanced.text

            return sections

        except Exception as e:
            print(f"Detailed summary generation failed: {e}")
            # Fallback summary
            try:
                basic_summary = text_model.generate_content(
                    f"Generate a comprehensive summary of this document:\n\n{self.full_text[:50000]}"
                )
                return {
                    "overview": basic_summary.text,
                    "key_findings": "See overview for key findings",
                    "methodologies": "See overview for methodologies",
                    "recommendations": "See overview for recommendations"
                }
            except:
                return {
                    "overview": "Error generating summary",
                    "key_findings": "",
                    "methodologies": "",
                    "recommendations": ""
                }


class MultiPDFProcessor:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
        self.summaries = {}

    def load_metadata_for_pdf(self, pdf_path):
        json_path = os.path.splitext(pdf_path)[0] + ".json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return None

    def process_all_pdfs(self):
        for pdf_path in self.pdf_paths:
            print(f"\n=== Processing {pdf_path} ===")
            rag = RAGPipeline(pdf_path)
            if rag.load_and_process_documents():
                summary = rag.generate_structured_summary()
                if summary:
                    self.summaries[pdf_path] = summary
                    print(f"Summary generated for {pdf_path}")
                else:
                    print(f"Failed to generate summary for {pdf_path}")
            else:
                print(f"Failed to load {pdf_path}")

    def save_summaries_to_json(self, json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.summaries, f, ensure_ascii=False, indent=4)
        print(f"Saved summaries to {json_path}")

    def load_summaries_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.summaries = json.load(f)
        print(f"Loaded summaries from {json_path}")

    def generate_overall_summary(self):
        combined_text = ""
        for pdf, summary in self.summaries.items():
            metadata = self.load_metadata_for_pdf(pdf)
            if metadata:
                citation = f"**Title**: {metadata.get('title')}\n" \
                    f"**Authors**: {', '.join(metadata.get('authors', []))}\n" \
                    f"**arXiv ID**: {metadata.get('arxiv_id')}\n" \
                    f"**Published**: {metadata.get('published')}\n" \
                    f"**PDF**: {metadata.get('pdf_url')}\n"
            else:
                citation = f"**Document**: {pdf}\n"

            combined_text += f"{citation}\n"
            for section, content in summary.items():
                combined_text += f"### {section.upper()}:\n{content}\n\n"

        combined_text = combined_text[:50000]  # Truncate if needed

        overall_summary_prompt = f"""
    You are given summaries of multiple documents. Provide a comprehensive overall summary synthesizing the information, highlighting common themes, differences, and key insights. Each document summary is preceded by citation metadata.

    Summaries:
    {combined_text}
    """

        response = text_model.generate_content(overall_summary_prompt)
        return response.text


def get_recent_arxiv_metadata(folder_path, num_files=5):
    metadata_list = []

    # Get all JSON files in the folder with their modification times
    json_files = [(f, os.path.getmtime(os.path.join(folder_path, f)))
                  for f in os.listdir(folder_path)
                  if f.endswith('.json') and os.path.isfile(os.path.join(folder_path, f))]

    # Sort files by modification time (newest first)
    json_files.sort(key=lambda x: x[1], reverse=True)

    # Process only the most recent files
    for file, _ in json_files[:num_files]:
        file_path = os.path.join(folder_path, file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

                # Format the published date for better readability
                if 'published' in metadata:
                    try:
                        published_dt = datetime.strptime(
                            metadata['published'], "%Y-%m-%dT%H:%M:%SZ")
                        metadata['published'] = published_dt.strftime(
                            "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pass

                # Add filename to the metadata
                metadata['filename'] = file
                metadata_list.append(metadata)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error reading {file}: {str(e)}")
            continue

    return metadata_list


def main_model():
    # pdf_list = [
    #     "data/pdfs/insides/0712.1443v1.pdf",
    #     "data/pdfs/insides/0808.3541v1.pdf",
    #     "data/pdfs/insides/1006.5209v2.pdf",
    # ]
    # Scrappping()
    folder_path = 'output\data\pdf'

    # Get a list of all files in the folder
    file_names = [f for f in os.listdir(
        folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files_with_mtime = [(f, os.path.getmtime(
        os.path.join(folder_path, f))) for f in file_names]
    files_with_mtime_sorted = sorted(
        files_with_mtime, key=lambda x: x[1], reverse=True)
    most_recent_files = [f[0] for f in files_with_mtime_sorted[:3]]

    # Print the file names
    for file in most_recent_files:
        pdf_list = [
            os.path.join(folder_path, file) if file.endswith('.pdf') else None
        ]

    processor = MultiPDFProcessor(pdf_list)
    processor.process_all_pdfs()
    processor.save_summaries_to_json("all_pdf_summaries.json")

    # Load summaries and generate overall summary
    processor.load_summaries_from_json("all_pdf_summaries.json")
    overall_summary = processor.generate_overall_summary()

    print("\n=== OVERALL SUMMARY ===\n")
    print(overall_summary)


if __name__ == "__main__":
    main_model()
