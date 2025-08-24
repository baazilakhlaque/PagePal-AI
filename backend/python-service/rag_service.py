import os
import pinecone
import re
import tempfile
import hashlib
import time
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Dict


# Load environment variables
load_dotenv()

# Import your existing libraries 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import google.generativeai as genai

app = FastAPI(title="UniRAG Python Service", version="1.0.0")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your Next.js app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=pinecone_api_key)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Pydantic models for request/response

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    history: List[ChatMessage] = []

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

class ProcessResponse(BaseModel):
    message: str
    chunks_created: int
    file_name: str

# Configure text splitter (exactly as your Colab code)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # Larger chunks since llama-text-embed-v2 can handle more
    chunk_overlap=200,  # 10% overlap
    length_function=len,
    separators=[
        "\n\n",  # Paragraph breaks (highest priority)
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "! ",    # Exclamation sentences
        "? ",    # Question sentences
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Spaces
        ""       # Character-level (last resort)
    ],
    is_separator_regex=False,
)

# Create index if it doesn't exist
index_name = os.getenv('PINECONE_INDEX_NAME')
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model":"llama-text-embed-v2",
            "field_map":{"text": "chunk_text"}
        }
    )

# Get the index
dense_index = pc.Index(index_name)

def clean_text(text: str) -> str:
    """Clean text data (exactly as your Colab code)"""
    # Remove extra newlines within paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace single newlines with space
    
    # Replace multiple newlines with a single newline (to preserve paragraph breaks)
    text = re.sub(r'\n+', '\n', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

async def process_pdf_file(file_path: str, file_name: str) -> int:
    """Process PDF file and store in Pinecone (your Colab logic)"""
    try:
        print(f"Processing PDF: {file_name}")
        
        # Load PDF 
        loader = PyPDFLoader(file_path)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        
        print(f"Loaded {len(pages)} pages from {file_name}")
        
        # Clean the data 
        cleaned_pages = []
        for page in pages:
            text = page.page_content
            text = clean_text(text)
            page.page_content = text
            cleaned_pages.append(page)
        
        print("Starting document chunking...")
        all_chunks = []

        for page_idx, page in enumerate(cleaned_pages):
            print(f"Processing page {page_idx + 1}/{len(cleaned_pages)}")
            
            # Split each page individually to maintain page metadata
            page_chunks = text_splitter.split_documents([page])
            
            # Add chunk-specific metadata 
            for i, chunk in enumerate(page_chunks):
                # Preserve original metadata and add chunk info
                chunk.metadata.update({
                    'chunk_id': f"page_{chunk.metadata['page']}_chunk_{i}",
                    'chunk_index': i,
                    'total_chunks_in_page': len(page_chunks),
                    'chunk_size': len(chunk.page_content),
                    'source_file': file_name,
                    'page_number': chunk.metadata['page']
                })
                all_chunks.append(chunk)

        print(f"Created {len(all_chunks)} chunks from {len(cleaned_pages)} pages")
        
        # Display chunk distribution 
        page_chunk_counts = {}
        for chunk in all_chunks:
            page_num = chunk.metadata['page_number']
            page_chunk_counts[page_num] = page_chunk_counts.get(page_num, 0) + 1

        print(f"\nChunk distribution across pages:")
        for page, count in sorted(page_chunk_counts.items())[:10]:  # Show first 10 pages
            print(f"Page {page}: {count} chunks")
        if len(page_chunk_counts) > 10:
            print(f"... and {len(page_chunk_counts) - 10} more pages")
        
        # Prepare records for Pinecone upsert 
        print("Preparing records for Pinecone upsert...")
        records_to_upsert = []
        failed_records = 0

        for i, chunk in enumerate(tqdm(all_chunks, desc="Preparing records")):
            try:
                # Create unique ID for this chunk
                chunk_id = hashlib.md5(
                    f"{file_name}_{chunk.metadata['chunk_id']}".encode()
                ).hexdigest()
                
                # Prepare record for upsert_records - note the correct format
                record = {
                    '_id': chunk_id,  # Use _id instead of id
                    'chunk_text': chunk.page_content,  # This matches your field_map
                    # Add metadata fields directly (not nested in metadata object)
                    'page_number': chunk.metadata.get('page', 0),
                    'chunk_id': chunk.metadata['chunk_id'],
                    'chunk_index': chunk.metadata['chunk_index'],
                    'source_file': chunk.metadata['source_file'],
                    'content_length': len(chunk.page_content),
                    'total_chunks_in_page': chunk.metadata['total_chunks_in_page']
                }
                
                records_to_upsert.append(record)
                
            except Exception as e:
                print(f"Error preparing record for chunk {i}: {e}")
                failed_records += 1
                continue

        print(f"Prepared {len(records_to_upsert)} records")
        if failed_records > 0:
            print(f"⚠️  Failed to prepare {failed_records} records")
        
        # Upsert to Pinecone 
        print("Upserting records to Pinecone...")
        batch_size = 100
        successful_upserts = 0
        failed_batches = []

        # Process in batches with progress bar
        total_batches = (len(records_to_upsert) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(total_batches), desc="Upserting batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(records_to_upsert))
            batch = records_to_upsert[start_idx:end_idx]
            
            try:
                # Correct upsert_records syntax with namespace as first parameter
                upsert_response = dense_index.upsert_records(
                    namespace="documents",  # Namespace is required as first parameter
                    records=batch
                )
                successful_upserts += len(batch)
                
                # Add small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in batch {batch_idx + 1}: {e}")
                failed_batches.append(batch_idx + 1)
                continue

        print(f"\n🎉 Upsert Summary:")
        print(f"Successfully upserted: {successful_upserts}/{len(records_to_upsert)} records")
        print(f"Success rate: {(successful_upserts/len(records_to_upsert)*100):.1f}%")

        if failed_batches:
            print(f"Failed batches: {failed_batches}")
        
        return len(all_chunks)
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

def rag_doc_qa(query: str, chatHistory: List[Dict[str, str]]) -> Dict[str, Any]:
    """RAG query function"""

    gemini_api_key = os.getenv('GEMINI_API_KEY')

    # client = genai.Client(api_key=gemini_api_key)
    try:
        # Search the dense index
        results = dense_index.search(
            namespace="documents",
            query={
                "top_k": 3,
                "inputs": {
                    'text': query
                }
            },
            rerank={
                "model": "bge-reranker-v2-m3",
                "top_n": 10,
                "rank_fields": ["chunk_text"]
            }
        )
        
        # Extract context and sources
        context_parts = []
        sources = []
        
        for hit in results['result']['hits']:
            if 'chunk_text' in hit['fields']:
                context_parts.append(hit['fields']['chunk_text'])
            if 'source_file' in hit['fields']:
                sources.append(hit['fields']['source_file'])
        
        context = "\n\n".join(context_parts)

        # Format history as dialogue
        history_text = ""
        for msg in chatHistory:
            role = "User" if msg.role == "user" else "Assistant"
            history_text += f"{role}: {msg.content}\n"
        
        # Generate answer using Gemini 
        # Prompt including context and chat historyß
        prompt = f"""You are PagePal, an AI study assistant that helps students understand their course materials through intelligent document analysis.

        DOCUMENT CONTEXT:
        {context}

        CONVERSATION HISTORY:
        {history_text}

        STUDENT QUESTION: {query}

        RESPONSE GUIDELINES:
        1. **Accuracy First**: Base your answer ONLY on the provided document context
        2. **Source Attribution**: When possible, mention which document or section the information comes from
        3. **Educational Value**: Explain concepts clearly, as if teaching a fellow student
        4. **Scope Management**: If the question requires information not in the documents, clearly state this limitation
        5. **Academic Tone**: Be helpful and encouraging while maintaining academic rigor
        6. **Conversation Flow**: Reference previous parts of the conversation when relevant
        7. **Clarity**: Use clear, well-structured responses with proper formatting

        SPECIFIC INSTRUCTIONS:
        - For exam-related questions: Focus on topics, dates, and requirements mentioned in the documents
        - For concept explanations: Use examples from the documents when available
        - For assignment questions: Provide guidance based on the specific requirements in the documents
        - For general questions: Offer insights while staying within the document scope

        If the documents don't contain sufficient information: "I don't have enough information from your uploaded materials to provide a complete answer. Consider uploading additional relevant documents or asking a more specific question about the materials you have."

        RESPONSE:"""

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # answer = response.candidates[0].content.parts[0].text
        answer = response.text

        
        return {
    "answer": answer,
    "sources": list(set(sources)),  # Remove duplicates
    "confidence": 0.85  # You can calculate this from Pinecone scores
}
        
    except Exception as e:
        print(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def clean_markdown(text):
    # Remove bold (**text**)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic or single asterisks (*text*) if any
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove bullet point asterisks at the start of lines
    text = re.sub(r'^\s*\*\s+', '', text, flags=re.MULTILINE)
    
    return text


def process_markdown(text, mode="plain"):
    """
    Process markdown text.
    
    mode="plain" → remove markdown formatting
    mode="html" → convert markdown to HTML tags
    """

    if mode == "plain":
        # Remove bold (**text**)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove italic (*text*)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Remove headings (# Heading)
        text = re.sub(r'^\s*#+\s+', '', text, flags=re.MULTILINE)
        return text.strip()

    elif mode == "html":
        # Convert bold (**text**) → <b>text</b>
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Convert italic (*text*) → <i>text</i>
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        # Convert headings (# Heading) → <h1>Heading</h1>, ## → <h2>, etc.
        def heading_replace(match):
            level = len(match.group(1))
            content = match.group(2)
            return f"<h{level}>{content}</h{level}>"
        text = re.sub(r'^(#{1,6})\s+(.*)', heading_replace, text, flags=re.MULTILINE)
        return text.strip()

    else:
        raise ValueError("Invalid mode. Use 'plain' or 'html'.")


@app.get("/")
async def root():
    return {"message": "UniRAG Python Service is running!"}

@app.post("/process-pdf", response_model=ProcessResponse)
async def process_pdf(file: UploadFile = File(...)):
    """Upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Process the PDF
        chunks_created = await process_pdf_file(tmp_file_path, file.filename)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return ProcessResponse(
            message="PDF processed successfully",
            chunks_created=chunks_created,
            file_name=file.filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system"""
    try:
        result = rag_doc_qa(request.question, request.history)

        #result["answer"] = clean_markdown(result["answer"])
        result["answer"] = process_markdown(result["answer"], mode="plain")

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-file/{file_name}")
async def delete_file_chunks(file_name: str):
    """
    Delete all chunks for a given file from Pinecone based on source_file metadata.
    """
    try:
        delete_response = dense_index.delete(
            filter={
                "source_file": {"$eq": file_name}
            },
            namespace="documents"  # match the namespace you used in upsert
        )

        return {
            "message": f"Deleted chunks for file: {file_name}",
            "pinecone_response": delete_response
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "UniRAG Python Service"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)