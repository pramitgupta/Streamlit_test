import streamlit as st
from markitdown import MarkItDown
import os

st.title("Universal File to Markdown Converter")

# File uploader for various formats
uploaded_file = st.file_uploader(
    "Upload a file (e.g., HTML, PPTX, PDF, DOCX, XLSX)",
    type=["html", "htm", "pptx", "pdf", "docx", "xlsx", "txt", "md", "rtf", "odt", "epub", "zip", "jpg", "png", "mp3", "wav"]
)

if uploaded_file is not None:
    try:
        # Initialize MarkItDown (disable plugins/AI for basics; enable if needed)
        md_converter = MarkItDown(enable_plugins=False)  # Set True for media descriptions
        
        # Convert file (MarkItDown handles bytes/path/URL)
        result = md_converter.convert(uploaded_file)
        markdown = result.text_content
        
        # Display the converted Markdown
        st.subheader("Converted Markdown")
        st.text_area("Output", markdown, height=300)
        
        # Allow user to choose file format for download
        file_format = st.selectbox("Download as", ["Markdown (.md)", "Text (.txt)"])
        file_extension = ".md" if file_format == "Markdown (.md)" else ".txt"
        
        # Use uploaded file's base name
        base_name = os.path.splitext(uploaded_file.name)[0]
        file_name = f"{base_name}_{'md' if file_extension == '.md' else 'txt'}{file_extension}"
        
        # Provide download button
        st.download_button(
            label=f"Download as {file_extension}",
            data=markdown,
            file_name=file_name,
            mime="text/markdown" if file_extension == ".md" else "text/plain"
        )
        
        # Inform user about download location
        st.info(
            f"The file will download to your browser's default download folder (e.g., Downloads). "
            f"To save it in the same directory as your uploaded file ({uploaded_file.name}), "
            f"move the downloaded file ({file_name}) manually to that location."
        )
        
    except Exception as e:
        st.error(f"Conversion failed: {str(e)}. Check file format or install extras (e.g., for PDFs).")
