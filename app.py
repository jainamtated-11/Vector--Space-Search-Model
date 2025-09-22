import streamlit as st
from backend import build_index, search_query

# Set page configuration for a wider layout
st.set_page_config(
    page_title="Vector Space Model Search Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Header Section ---
st.markdown("<h1 style='text-align: center;'>🔍 Vector Space Model Search Engine</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7393B3;'>Ranked Retrieval using TF-IDF and Cosine Similarity</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- Indexing Status and Search Bar Section ---
# Use a container to group the indexing status and search input
with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Enter your query")
        query = st.text_input(" ", label_visibility="collapsed")
    
    with col2:
        st.subheader(" ") # Spacer to align button
        if st.button("Search", use_container_width=True, type="primary"):
            if query.strip():
                st.session_state["search_triggered"] = True
                st.session_state["query"] = query
            else:
                st.warning("Please enter a query to search.")

# Display indexing status
if "index_built" not in st.session_state:
    st.session_state["index_built"] = False
    
try:
    if not st.session_state["index_built"]:
        st.info("Building index... Please wait.")
        build_index(zip_path="corpus-1.zip")
        st.session_state["index_built"] = True
        st.success("✅ Index built successfully!")
except Exception as e:
    st.error(f"Error while pre-building index: {e}")

# --- Results Section ---
st.markdown("---")
st.subheader("Search Results")

if st.session_state.get("search_triggered") and st.session_state.get("index_built"):
    try:
        results = search_query(st.session_state["query"])
        
        if results:
            for doc_id, score, doc_content in results:
                # Use a markdown expander for each result
                with st.expander(f"**DocID: {doc_id}** (Relevance Score: {score:.4f})"):
                    st.write(doc_content)
        else:
            st.info("No matching documents found. Try a different query.")
    except Exception as e:
        st.error(f"Search failed: {e}")