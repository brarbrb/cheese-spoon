import pdfplumber
import re 
import pandas as pd

KB = pd.read_csv('data/preprocessing/courses_data_before_llm.csv')

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



# --- 2. Logic to determine eligibility ---
def check_eligibility(prereqs, finished_set):
    if len(prereqs) == 0: # courses that have no prequisites
        return True
    
    # 'isdisjoint' returns True if NO elements are shared. 
    # We want the opposite (False), meaning at least one element IS shared.
    return not set(prereqs).isdisjoint(finished_set)


def check_eligibility_full_logic(completed_ids):
    """
    Compares completed courses against a mock catalog to determine 
    eligibility for NEXT semester courses.
    """
    # eligible_courses = KB[KB['prerequisites'].apply(lambda prereqs: set(prereqs).issubset(completed_ids))]
    
    finished_set = set(completed_ids)
    not_taken_courses = KB[~KB['course_id'].isin(finished_set)]
    eligible_courses = not_taken_courses[
        not_taken_courses['prerequisites'].apply(lambda x: check_eligibility(x, finished_set))
    ]
    
    print(f"Found {len(eligible_courses)} eligible courses.")
    print(f"List of names: {eligible_courses['title']}")

# --- 3. Creates easy to work with dictionary to get titles ---
def load_course_titles():
    # Create a dictionary: {'094101': '00940101 - מבוא להנדסת תעשיה...'}
    # print(dict(zip(KB['course_id'], KB['title']))) # course ids are incorrect!
    # return dict(zip(KB['course_id'], KB['title']))
    df = KB.copy()
    course_map = {}
    
    for title in df['title']: 
        full_id_str = title.split('-')[0].strip()
        clean_id = full_id_str[-6:] # Takes "094101" from "00940101"
        
        course_map[clean_id] = title
    # print(course_map)
    return course_map

COURSE_TITLES = load_course_titles()