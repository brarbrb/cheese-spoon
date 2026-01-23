import pdfplumber
import re 
import pandas as pd

KB = pd.read_csv('../data/preprocessing/courses_data_before_llm.csv')


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
    # Mock Catalog of "Next Semester Options"
    catalog = [
        {"id": "094123", "name": "Algorithms 1", "points": 4.0, "prereq": "104013"}, # Req: Calculus
        {"id": "094210", "name": "Operating Systems", "points": 4.0, "prereq": "094134"}, # Req: Data Structures
        {"id": "096345", "name": "Machine Learning", "points": 3.0, "prereq": "094123"}, # Req: Algo 1
        {"id": "094332", "name": "Databases", "points": 3.0, "prereq": None},
    ]

    # results = []
    # stats = {"eligible": 0, "missing_prereq": 0, "not_offered": 0}

    # for course in catalog:
    #     # Simple Logic: Do we have the prerequisite in our completed list?
    #     if course['prereq'] and course['prereq'] not in completed_ids:
    #         course['status'] = "missing_prereq"
    #         course['reason'] = f"Missing Prereq: {course['prereq']}"
    #         stats["missing_prereq"] += 1
    #     else:
    #         course['status'] = "eligible"
    #         course['reason'] = "Prerequisites met"
    #         stats["eligible"] += 1
        
    #     results.append(course)
    
    
    # eligible_courses = KB[KB['prerequisites'].apply(lambda prereqs: set(prereqs).issubset(completed_ids))]
    
    finished_set = set(completed_ids)
    not_taken_courses = KB[~KB['course_id'].isin(finished_set)]
    eligible_courses = not_taken_courses[
        not_taken_courses['prerequisites'].apply(lambda x: check_eligibility(x, finished_set))
    ]
    
    print(f"Found {len(eligible_courses)} eligible courses.")
    print(f"List of names: {eligible_courses['title']}")
