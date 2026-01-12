import pdfplumber
import re 

# --- 1. Helper Function to Parse PDF ---
def parse_grades_pdf(file_storage):
    """
    Extracts text from the uploaded PDF and finds completed course IDs.
    Assumes Technion format (8-digit course codes).
    """
    completed_courses = set()
    
    try:
        with pdfplumber.open(file_storage) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Regex to find 8-digit numbers (Technion course IDs)
                    found_ids = re.findall(r'\b\d{8}\b', text)
                    completed_courses.update(found_ids)
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    
    print("Parsed from the PDF:", list(completed_courses))
    return list(completed_courses)
