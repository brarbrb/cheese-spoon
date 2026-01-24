import pdfplumber
import re 
import pandas as pd
from src.knowledgebase import get_knowledgebase

# SEMESTER_NAME = "WINTER_2025_2026"
# KB = get_knowledgebase(SEMESTER_NAME,user_query="",only_ids_titles=True)
# print(KB)

# --- 1. Helper Function to Parse PDF ---
# def parse_grades_pdf(file_storage):
#     """
#     Extracts text from the uploaded PDF and finds completed course IDs.
#     Assumes Technion format (8-digit course codes).
#     """
#     completed_courses = set()
#     try:
#         with pdfplumber.open(file_storage) as pdf:
#             for page in pdf.pages:
#                 text = page.extract_text()
#                 if text:
#                     # Regex to find 8-digit numbers (Technion course IDs)
#                     found_ids = re.findall(r'\b\d{8}\b', text)
#                     completed_courses.update(found_ids)
#     except Exception as e:
#         print(f"Error parsing PDF: {e}")
    
#     print("Parsed from the PDF:", list(completed_courses))
#     return list(completed_courses)

def parse_grades_pdf(file_storage):
    """
    Extracts course IDs and Names from a text-based Technion transcript.
    """
    completed_courses = []
    seen_ids = set()

    try:
        with pdfplumber.open(file_storage) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                for line in text.split('\n'):
                    # Skip student info/header lines
                    if any(x in line for x in ['ת.ז', 'ז.ת', 'תעודת', 'תדועת']):
                        continue

                    # Find Course ID (6 or 8 digits)
                    id_match = re.search(r'\b(\d{6}|\d{8})\b', line)
                    if id_match:
                        # Safety: Skip if it's actually part of a 9-digit number
                        if re.search(r'\d{9}', line):
                            continue
                            
                        course_id = id_match.group(1)
                        if course_id in seen_ids:
                            continue

                        # Remove ID to isolate name/metadata
                        temp_line = line.replace(course_id, '')

                        # Flip text if it contains reversed Hebrew keywords
                        if re.search(r'(ביבא|ףרוח|ץיק)', temp_line):
                            temp_line = temp_line[::-1]

                        # Clean metadata
                        # 1. Semesters and Status (Added 'עובר' and 'פטור')
                        temp_line = re.sub(r'(אביב|חורף|קיץ|תשפ"?[א-ת]|סמסטר|עובר|פטור|Winter|Spring|Summer)', '', temp_line)
                        # 2. Year ranges (2020-2021 or reversed)
                        temp_line = re.sub(r'\d{4}-\d{4}', '', temp_line)
                        # 3. Credits (floats)
                        temp_line = re.sub(r'\b\d+\.\d+\b', '', temp_line)
                        # 4. Grades (2-3 digit integers)
                        temp_line = re.sub(r'\b\d{2,3}\b', '', temp_line)
                        # 5. Artifacts like (E)
                        temp_line = re.sub(r'[\(\)]E[\(\)]', '', temp_line)
                        # 6. Trailing single digits
                        temp_line = re.sub(r'\s\d\s*$', '', temp_line)

                        course_name = temp_line.strip(' ,.-"\'')
                        
                        # Add if valid name remains
                        if course_name and not course_name.replace(" ", "").isdigit():
                            completed_courses.append({
                                'id': course_id,
                                'name': course_name
                            })
                            seen_ids.add(course_id)

    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return []
    
    return completed_courses


# --- 2. Helper function to parse the specific Hebrew format ---
def parse_review_summary(text):
    if not isinstance(text, str) or not text:
        return {
            'interest': '',
            'workload': '',
            'bottom_line': ''
        }
    
    parsed = {}
    
    # Extract "Interest" (everything between 'עניין -' and the next section)
    # using re.DOTALL so . matches newlines
    match_interest = re.search(r'עניין\s*-(.*?)(?=מטלות|שורה|$)', text, re.DOTALL)
    parsed['interest'] = match_interest.group(1).strip() if match_interest else ""

    # Extract "Workload" (handling the slash in 'מטלות\מבחן')
    match_workload = re.search(r'מטלות[\\/]?מבחן\s*-(.*?)(?=שורה|$)', text, re.DOTALL)
    parsed['workload'] = match_workload.group(1).strip() if match_workload else ""

    # Extract "Bottom Line" (everything after 'שורה תחתונה -')
    match_bottom = re.search(r'שורה תחתונה\s*-(.*)', text, re.DOTALL)
    parsed['bottom_line'] = match_bottom.group(1).strip() if match_bottom else ""
    
    return parsed