import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from streamlit_pdf_viewer import pdf_viewer


import fitz
from PIL import Image, ImageDraw
import json
import numpy as np
import io
import base64
import time

# Configure the page with wide layout and expanded sidebar by default
st.set_page_config(
    page_title="RAG Vision - Transparent Document Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define our custom styles - includes animations and responsive design elements
st.markdown("""
<style>
/* Styling for evidence containers and scores */
.evidence-score {
    color: #4CAF50;
    font-weight: bold;
    padding: 2px 8px;
    border-radius: 4px;
    background: rgba(76, 175, 80, 0.1);
}

.evidence-container {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Layout utilities */
.centered-content {
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}

/* PDF viewer styling */
.pdf-viewer {
    border: 1px solid #2c2c2c;
    border-radius: 8px;
    overflow: hidden;
    margin: 10px 0;
}

/* Link styling */
.custom-link {
    color: #00bcd4;
    text-decoration: none;
    transition: color 0.3s ease;
}

.custom-link:hover {
    color: #80deea;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)




# Initialize session state for message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to display PDF documents
def display_pdf(file_path: str):
    """Creates an embedded PDF viewer with navigation controls."""
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f"""
                            <iframe 
                            src="data:application/pdf;base64,{base64_pdf}" 
                            width="100%" 
                            height="600px" 
                            type="application/pdf"
                            >
                            </iframe>
                            """
        st.markdown(f"### Preview of {file_path}")
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

# Helper function to process search results and create evidence
def process_evidence(matches, narratives, pdf_path):
    """Processes search matches and creates evidence items with annotations."""
    evidence_items = []
    context = ""
    
    for score, match in matches:
        # Create annotated image
        annotated_img = annotate_pdf_chunk(pdf_path, match, narratives['layout_dimensions'])
        
        # Convert to bytes for storage
        img_byte_arr = io.BytesIO()
        annotated_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Store evidence item
        evidence_items.append({
            "score": score,
            "page": match["page_number"],
            "image": img_byte_arr
        })
        
        # Append to context
        context += match["narrative_text"] + "\n\n"
    
    return evidence_items, context

# Sidebar configuration
with st.sidebar:
    
    # Navigation using selectbox for better accessibility
    page = st.selectbox(
        "Jump to... 👇",
        ["About", "Demo"],
        key="navigation"
    )
    
    st.divider()
    
    # Document section for demo page
    if page == "Demo":
        st.markdown("### 📄 Source Document")
        st.markdown("**Current PDF**: Letter from Birmingham Jail")
        st.markdown("""
        *A powerful letter written by Dr. Martin Luther King Jr., expressing his thoughts 
        on civil rights and justice from Birmingham City Jail.*
        """)
        
        # Display PDF preview
        try:
            display_pdf("susi-letter-from-birmingham-jail.pdf")
        except Exception as e:
            st.error("Could not load PDF preview")
        
        st.divider()
        
        # Clear history button with confirmation
        if st.button("🗑️ Clear Chat History", key="clear_history"):
            st.session_state.messages = []
            st.rerun()

# Core helper functions for RAG functionality
def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
    
    Returns:
        Similarity score between 0 and 1
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def search_narratives(query: str, narratives: dict, model, top_k: int = 3):
    """
    Search for the most relevant narrative chunks based on semantic similarity.
    
    Args:
        query: User's question
        narratives: Dictionary containing narrative entries and embeddings
        model: Sentence transformer model for encoding
        top_k: Number of top matches to return
        
    Returns:
        List of tuples containing (similarity_score, narrative_entry)
    """
    query_embedding = model.encode(query)
    similarities = []
    
    for narrative in narratives['entries']:
        similarity = cosine_similarity(query_embedding, narrative['embedding'])
        similarities.append((similarity, narrative))
    
    return sorted(similarities, key=lambda x: x[0], reverse=True)[:top_k]

def annotate_pdf_chunk(pdf_path: str, match: dict, layout_dims: dict) -> Image:
    """
    Create an annotated image of a PDF page with highlighted text chunk.
    
    Args:
        pdf_path: Path to the PDF file
        match: Dictionary containing chunk information and coordinates
        layout_dims: Dictionary with layout dimensions
    
    Returns:
        PIL Image with highlighted annotation
    """
    page_num = match['page_number']
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[page_num - 1]
    
    # Create high-resolution image for better annotation
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Calculate scaling factors
    scale_x = img.width / layout_dims['layout_width']
    scale_y = img.height / layout_dims['layout_height']
    
    # Scale coordinates and draw highlight
    bbox = match['bbox_values']
    scaled_points = [(p[0] * scale_x, p[1] * scale_y) for p in bbox]
    
    # Draw semi-transparent highlight
    draw.polygon(scaled_points, fill=(236, 255, 229, 120))
    # Draw border for emphasis
    draw.polygon(scaled_points, outline=(255, 225, 0), width=3)
    
    pdf_doc.close()
    return img

# AI Component Initialization
@st.cache_resource
def initialize_ai_components():
    """Initialize and cache AI models.
    
    This function handles the initialization of both the Gemini and sentence transformer models.
    It's cached to prevent reloading on each interaction.
    
    Returns:
        tuple: (gemini_model, embedding_model) - The initialized AI models
    """
    # Initialize Gemini
    genai.configure(api_key=st.secrets['api_key'])  # Replace with your actual API key
    model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    
    return model, embedding_model

def get_gemini_response(query: str, context: str, top_image: Image) -> str:
    """
    Generate response using Gemini model with both text and image context.
    
    Args:
        query: User's question
        context: Text context from relevant chunks
        top_image: Image of the most relevant chunk with highlighting
        
    Returns:
        Generated response from Gemini
    """
    system_prompt = """Please analyze the provided image and question carefully.
    Pay special attention to the highlighted/annotated sections in the image (marked in yellow),
    as these represent the most relevant passages for answering the question.
    Focus particularly on the content within these highlighted areas when formulating your response.
    Ensure your answer is well-structured and directly addresses the question."""
    
    user_prompt = f"""Question: {query}
    
Context: {context}

The highlighted sections in the image are particularly relevant to this question."""

    response = model.generate_content([system_prompt, user_prompt, top_image])
    return response.text

# Load narratives data
@st.cache_data
def get_narratives():
    """Loads and caches the processed narratives data."""
    with open("processed_narratives_jailer_v1.json", 'r') as f:
        return json.load(f)

if page == "About":
    # Title with rotating effect
 
    st.markdown('<h1 style="text-align: center;">🔮 ThoughtScope AI</h3>', unsafe_allow_html=True)

    # Initial description
    st.markdown("""
    ThoughtScope AI is an innovative tool that demonstrates transparent AI reasoning by showing you 
    exactly where information comes from in documents. It combines state-of-the-art language 
    models with visual understanding to provide verifiable answers.
    """)


  
    
    st.image("architecture.png", use_container_width=True)
    
    st.divider()
    
    # Getting Started section
    st.markdown("### 🚀 Getting Started")
    st.markdown("""
    1. Navigate to the **Demo** page to try out the system
    2. Ask questions about the "Letter from Birmingham Jail"
    3. See both the AI's response and the supporting evidence
    
    ### 📋 Want to replicate this?
    * GitHub repository coming soon!
    * Full documentation and setup guide included
    * Step-by-step tutorial for creating your own version
    """)


elif page == "Demo":
    st.markdown('<h1 style="text-align: center;">🎥 Demo</h3>', unsafe_allow_html=True)

    # Welcome message for new chat
    if not st.session_state.messages:
        st.info("""
        👋 Welcome to the RAG Vision Demo!
        
        I'm ready to answer questions about the "Letter from Birmingham Jail".
        The source document is loaded and ready for exploration.
        
        Try asking something like:
        - What is King's critique of the white moderate?
        - How does King justify civil disobedience?
        - What is the difference between just and unjust laws?
        """)
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown("### 🤖 AI Response")
                st.markdown(message["content"])
                
                if "evidence" in message:
                    st.markdown("### 📑 Supporting Evidence")
                    for i, evidence in enumerate(message["evidence"], 1):
                        st.markdown(f"""
                        <div class='evidence-container centered-content'>
                            <h4>Evidence #{i} <span class='evidence-score'>Score: {evidence['score']:.2f}</span></h4>
                        </div>
                        """, unsafe_allow_html=True)
                        st.image(evidence["image"], caption=f"Page {evidence['page']}")
            else:
                st.markdown(message["content"])
    
    # Chat input and processing
    if prompt := st.chat_input("Ask your question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🔍 Processing your query...", expanded=True) as status:
                # Load components
                model, embedding_model = initialize_ai_components()
                narratives = get_narratives()
                
                # Search for relevant content
                status.write("Finding relevant passages...")
                matches = search_narratives(prompt, narratives, embedding_model)
                
                # Process evidence
                evidence_items, context = process_evidence(matches, narratives, "susi-letter-from-birmingham-jail.pdf")
                
                # Generate response
                status.write("Generating answer...")
                if evidence_items:
                    top_image = Image.open(io.BytesIO(evidence_items[0]["image"]))
                    response = get_gemini_response(prompt, context, top_image)
                else:
                    response = "I apologize, but I couldn't find relevant content to answer your question."
                
                # Update status and display response
                status.update(label="✅ Response ready!", state="complete")
                
                st.markdown("### 🤖 AI Response")
                st.markdown(response)
                st.markdown("### 📑 Supporting Evidence")
                
                # Display evidence
                for i, evidence in enumerate(evidence_items, 1):
                    st.markdown(f"""
                    <div class='evidence-container centered-content'>
                        <h4>Evidence #{i} <span class='evidence-score'>Score: {evidence['score']:.2f}</span></h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(evidence["image"], caption=f"Page {evidence['page']}")
                
                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "evidence": evidence_items
                })
