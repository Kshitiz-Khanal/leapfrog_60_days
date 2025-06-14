import json 
import re
import textwrap
import warnings
import logging
import os
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
import glob

# Suppress warnings
logging.getLogger('pypdf').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


# Configure Gemini API key
GOOGLE_API_KEY = "AIzaSyA10ceOZNiCW-lEOYUWD3WARt-6EXjY4BA"  # Replace with your real key
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
        self.metadata = None

    def load_metadata(self):
        if not self.pdf_path:
            return None
        json_path = os.path.splitext(self.pdf_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                return self.metadata
            except Exception as e:
                print(f"Error loading metadata: {e}")
                return None
        return None

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
            
            # Load metadata after successful document processing
            self.load_metadata()
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

5. **METADATA** - Include all available metadata about the document in this format:
   - **Title**: [document title]
   - **Authors**: [comma-separated list of authors]
   - **Publication Date**: [date]
   - **arXiv ID**: [ID if available]
   - **PDF URL**: [URL if available]
   - **Summary**: [brief summary from metadata if available]

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
                "recommendations": "",
                "metadata": ""
            }

            section_pattern = re.compile(
                r'\d+\.\s*\*\*(OVERVIEW|KEY FINDINGS|METHODOLOGIES|RECOMMENDATIONS|METADATA)\*\*[-\s]*(.*?)(?=\d+\.\s*\*\*|\Z)',
                re.DOTALL | re.IGNORECASE
            )
            matches = section_pattern.finditer(full_summary)
            for match in matches:
                section_name = match.group(1).lower().replace(' ', '_')
                content = match.group(2).strip()
                if section_name in sections:
                    sections[section_name] = content

            # If metadata section is empty but we have metadata, create it
            if not sections["metadata"] and self.metadata:
                metadata_content = []
                if "title" in self.metadata:
                    metadata_content.append(f"**Title**: {self.metadata['title']}")
                if "authors" in self.metadata:
                    metadata_content.append(f"**Authors**: {', '.join(self.metadata['authors'])}")
                if "published" in self.metadata:
                    metadata_content.append(f"**Publication Date**: {self.metadata['published']}")
                if "arxiv_id" in self.metadata:
                    metadata_content.append(f"**arXiv ID**: {self.metadata['arxiv_id']}")
                if "pdf_url" in self.metadata:
                    metadata_content.append(f"**PDF URL**: {self.metadata['pdf_url']}")
                if "summary" in self.metadata:
                    metadata_content.append(f"**Summary**: {self.metadata['summary'][:500]}...")
                
                sections["metadata"] = "\n".join(metadata_content)

            # Fill missing sections using the base model
            for section in sections:
                if not sections[section]:
                    detail_prompt = f"""Generate an EXTREMELY DETAILED {section.replace('_', ' ')} section for this document.
Include all relevant details, examples, and specific information. Document excerpt:\n\n{self.full_text[:20000]}"""
                    response = text_model.generate_content(detail_prompt)
                    sections[section] = response.text

            # Enhance short sections
            for section, content in sections.items():
                if len(content.split()) < 100 and section != "metadata":
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
                    "recommendations": "See overview for recommendations",
                    "metadata": self.metadata if self.metadata else "No metadata available"
                }
            except:
                return {
                    "overview": "Error generating summary",
                    "key_findings": "",
                    "methodologies": "",
                    "recommendations": "",
                    "metadata": self.metadata if self.metadata else "No metadata available"
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
            # Use the metadata from the summary if available
            metadata_content = summary.get("metadata", "")
            if metadata_content:
                combined_text += f"{metadata_content}\n\n"
            else:
                # Fallback to loading from JSON file
                metadata = self.load_metadata_for_pdf(pdf)
                if metadata:
                    citation = f"**Title**: {metadata.get('title')}\n" \
                            f"**Authors**: {', '.join(metadata.get('authors', []))}\n" \
                            f"**arXiv ID**: {metadata.get('arxiv_id')}\n" \
                            f"**Published**: {metadata.get('published')}\n" \
                            f"**PDF**: {metadata.get('pdf_url')}\n"
                    combined_text += f"{citation}\n"

            for section, content in summary.items():
                if section != "metadata":  # Skip metadata as we already added it
                    combined_text += f"### {section.upper()}:\n{content}\n\n"

            # Add the PDF name at the end of the summary for this document
            pdf_name = os.path.basename(pdf).replace(".pdf", "")
            combined_text += f"***{pdf_name}***\n\n"

        combined_text = combined_text[:50000]  # Adjust length limit as necessary

        # Final overall summary prompt
        overall_summary_prompt = f"""
Synthesize the provided document summaries into a cohesive overall summary, highlighting common themes, key differences, and insights, and include citations for each point in parentheses (arxiv_id).

Summaries:
{combined_text}
"""

        response = text_model.generate_content(overall_summary_prompt)
        return response.text

def combine_metadata_files(input_dir='data', output_file='combined_metadata.json'):
    """
    Combine all metadata JSON files from a directory, sorted by modification time (newest first).
    
    Args:
        input_dir (str): Directory containing metadata JSON files
        output_file (str): Path for output combined JSON file
    """
    # Get all JSON files in directory with their modification times
    json_files = [f for f in glob.glob(os.path.join(input_dir, 'meta*.json')) 
                 if os.path.isfile(f)]
    files_with_mtime = [(f, os.path.getmtime(f)) for f in json_files]
    
    # Sort by modification time (newest first)
    files_sorted = sorted(files_with_mtime, key=lambda x: x[1], reverse=True)
    
    # Load all metadata files in sorted order
    combined_metadata = []
    for file_path, _ in files_sorted:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                combined_metadata.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse {file_path} - skipping")
    
    # Save combined data
    with open(output_file, 'w') as outfile:
        json.dump(combined_metadata, outfile, indent=4)
    
    print(f"Successfully combined {len(combined_metadata)} metadata files into {output_file}")
    print(f"Newest file was: {os.path.basename(files_sorted[0][0])}")


if __name__ == "__main__":
    folder_path = 'model/data/pdf'
   
    file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    pdf_list = [
        os.path.join(folder_path, file) for file in file_names if file.endswith('.pdf')
    ]
    combine_metadata_files(input_dir='model/data/meta', 
                      output_file='data/combined_metadata.json')

    processor = MultiPDFProcessor(pdf_list)
    processor.process_all_pdfs()
    processor.save_summaries_to_json("all_pdf_summaries.json")

    # Generate and display the overall summary
    processor.load_summaries_from_json("all_pdf_summaries.json")
    overall_summary = processor.generate_overall_summary()

    print("\n=== OVERALL SUMMARY ===\n")
    print(overall_summary)
