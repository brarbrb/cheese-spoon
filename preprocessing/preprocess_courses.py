import json
import os
import re
import pandas as pd
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(filepath):
    """
    Load JSON file with UTF-8 encoding.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with file contents or None if error
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def get_all_json_files(folder_path):
    """
    Scan folder for all .json files.

    Args:
        folder_path: Path to folder containing JSON files

    Returns:
        Sorted list of file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return []

    json_files = sorted(folder.glob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files in {folder_path}")
    return json_files


def clean_text(text):
    """
    Clean text by removing extra whitespace and newlines.

    Args:
        text: Text to clean

    Returns:
        Cleaned string or empty string
    """
    if text is None:
        return ""

    # Replace multiple newlines with single newline
    text = re.sub(r'\n\n+', '\n\n', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    # Normalize spaces
    text = re.sub(r' +', ' ', text)

    return text


def extract_course_id(data):
    """
    Extract course ID from data.

    Args:
        data: Dictionary with course data

    Returns:
        Course ID as string or None
    """
    try:
        return data.get('course_id')
    except Exception as e:
        logger.error(f"Error extracting course_id: {e}")
        return None


def extract_description(raw_text):
    """
    Extract course description from raw_text.

    Args:
        raw_text: Raw text from course information

    Returns:
        Description string or empty string
    """
    if not raw_text:
        return ""

    try:
        # Find the course title line (first line with course code)
        lines = raw_text.split('\n')

        # Skip first few lines (title, faculty, etc.) to find description
        start_idx = 0
        for i, line in enumerate(lines):
            if 'קטלוג מקצועות מקוון' in line:
                start_idx = i + 1
                break

        # Find where description ends
        end_markers = [
            'מקצועות קדם:',
            'מקצועות ללא זיכוי נוסף',
            'מועד א\':',
            'הערות:'
        ]

        description_lines = []
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()

            # Check if we've hit an end marker
            if any(marker in line for marker in end_markers):
                break

            # Skip empty lines at the start
            if not description_lines and not line:
                continue

            description_lines.append(line)

        description = '\n'.join(description_lines)
        return clean_text(description)

    except Exception as e:
        logger.error(f"Error extracting description: {e}")
        return ""


def extract_credits(raw_text):
    """
    Extract credit points from raw_text.

    Args:
        raw_text: Raw text from course information

    Returns:
        Credits as float or None
    """
    if not raw_text:
        return None

    try:
        pattern = r'נקודות:\s*(\d+\.?\d*)'
        match = re.search(pattern, raw_text)

        if match:
            return float(match.group(1))
        return None

    except Exception as e:
        logger.error(f"Error extracting credits: {e}")
        return None


def parse_prerequisite_group(group_text):
    """
    Parse a single prerequisite group and extract course codes.

    Args:
        group_text: Text of a single prerequisite group

    Returns:
        List of course codes (strings)
    """
    # Extract all 8-digit course codes
    course_codes = re.findall(r'\d{8}', group_text)
    return course_codes


def extract_prerequisites(raw_text):
    """
    Extract prerequisites from raw_text and parse into array of arrays.

    Args:
        raw_text: Raw text from course information

    Returns:
        List of lists containing prerequisite course codes
    """
    if not raw_text:
        return []

    try:
        # Find the prerequisite line
        # Pattern matches "מקצועות קדם:" followed by content until double newline or next Hebrew section
        pattern = r'מקצועות קדם:\s*(.+?)(?=\n\n|\n(?:מקצועות ללא|מועד|הערות:|$))'
        match = re.search(pattern, raw_text, re.DOTALL)

        if not match:
            return []

        prereq_text = match.group(1).strip()

        # Split by "או" to get alternative groups
        alternatives = re.split(r'\s+או\s+', prereq_text)

        result = []
        for alt in alternatives:
            # Parse each alternative group
            courses = parse_prerequisite_group(alt)
            if courses:
                result.append(courses)

        return result

    except Exception as e:
        logger.error(f"Error extracting prerequisites: {e}")
        return []


def extract_exam_date(raw_text, moed_type):
    """
    Extract exam date for specified moed.

    Args:
        raw_text: Raw text from course information
        moed_type: "א" or "ב"

    Returns:
        Date string in DD-MM-YYYY format or None
    """
    if not raw_text:
        return None

    try:
        # Pattern allows for optional apostrophe after moed letter
        pattern = rf'מועד\s+{moed_type}\'?:\s*(\d{{2}}-\d{{2}}-\d{{4}})'
        match = re.search(pattern, raw_text)

        if match:
            return match.group(1)
        return None

    except Exception as e:
        logger.error(f"Error extracting exam date for moed {moed_type}: {e}")
        return None


def extract_general_rating(data):
    """
    Extract general rating from feedback data.

    Args:
        data: Dictionary with course data

    Returns:
        General rating as float or None
    """
    try:
        ratings = data.get('feedback', {}).get('summary', {}).get('ratings', {})
        return ratings.get('כללי')
    except Exception as e:
        logger.error(f"Error extracting general rating: {e}")
        return None


def extract_workload_rating(data):
    """
    Extract workload rating from feedback data.

    Args:
        data: Dictionary with course data

    Returns:
        Workload rating as float or None
    """
    try:
        ratings = data.get('feedback', {}).get('summary', {}).get('ratings', {})
        return ratings.get('עומס')
    except Exception as e:
        logger.error(f"Error extracting workload rating: {e}")
        return None


def extract_all_reviews(data):
    """
    Extract and concatenate all review contents.

    Args:
        data: Dictionary with course data

    Returns:
        Concatenated review string or empty string
    """
    try:
        reviews = data.get('feedback', {}).get('reviews', [])

        review_contents = []
        for review in reviews:
            content = review.get('content', '').strip()
            if content:  # Only include non-empty reviews
                review_contents.append(content)

        if review_contents:
            return '\n---\n'.join(review_contents)
        return ""

    except Exception as e:
        logger.error(f"Error extracting reviews: {e}")
        return ""


def extract_avg_grades(histogram_raw_text):
    """
    Extract average final grades per semester from histogram data.

    Args:
        histogram_raw_text: Raw text from histograms section

    Returns:
        Dictionary with semester names as keys and final grades as values
    """
    if not histogram_raw_text:
        return {}

    try:
        # Check if histograms are not available
        if 'לא קיימות היסטוגרמות' in histogram_raw_text:
            return {}

        # Dictionary to store results
        grades_dict = {}

        # Split by lines
        lines = histogram_raw_text.split('\n')

        for line in lines:
            # Check if line contains a semester
            semester_match = re.search(r'((?:אביב|חורף)\s+\d{4}(?:-\d{4})?)', line)

            if semester_match:
                semester = semester_match.group(1).strip()

                # Look for "סופי" grade in the same line
                # Pattern: סופי followed by optional space and a number
                grade_match = re.search(r'סופי\s+(\d+)', line)

                if grade_match:
                    grade = int(grade_match.group(1))
                    grades_dict[semester] = grade

        return grades_dict

    except Exception as e:
        logger.error(f"Error extracting avg grades: {e}")
        return {}


def safe_extract(func, *args, default=None):
    """
    Safely execute an extraction function with error handling.

    Args:
        func: Function to execute
        *args: Arguments to pass to function
        default: Default value to return on error

    Returns:
        Result of function or default value
    """
    try:
        return func(*args)
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {e}")
        return default


def extract_course_data(filepath, debug=False):
    """
    Extract all data from a single course JSON file.

    Args:
        filepath: Path to JSON file
        debug: If True, print detailed extraction info

    Returns:
        Dictionary with extracted data or None
    """
    data = load_json_file(filepath)
    if data is None:
        return None

    raw_text = data.get('information', {}).get('raw_text', '')
    histogram_raw_text = data.get('histograms', {}).get('raw_text', '')

    course_row = {
        'course_id': safe_extract(extract_course_id, data),
        'title': safe_extract(extract_title, data, default=""),
        'description': safe_extract(extract_description, raw_text, default=""),
        'credits': safe_extract(extract_credits, raw_text),
        'prerequisites': safe_extract(extract_prerequisites, raw_text, default=[]),
        'moed_a': safe_extract(extract_exam_date, raw_text, 'א'),
        'moed_b': safe_extract(extract_exam_date, raw_text, 'ב'),
        'general_rating': safe_extract(extract_general_rating, data),
        'workload_rating': safe_extract(extract_workload_rating, data),
        'all_reviews': safe_extract(extract_all_reviews, data, default=""),
        'avg_grades': safe_extract(extract_avg_grades, histogram_raw_text, default={})
    }

    if debug:
        print("\n" + "=" * 80)
        print(f"📚 COURSE: {course_row['course_id']}")
        print(f"📖 Title: {course_row['title']}")  # Add this line
        print("=" * 80)

        print(f"📝 Description: {course_row['description'][:100]}..." if len(
            course_row['description']) > 100 else f"📝 Description: {course_row['description']}")
        print(f"💳 Credits: {course_row['credits']}")
        print(f"📋 Prerequisites: {course_row['prerequisites']}")
        print(f"📅 Moed A: {course_row['moed_a']}")
        print(f"📅 Moed B: {course_row['moed_b']}")
        print(f"⭐ General Rating: {course_row['general_rating']}")
        print(f"⚖️  Workload Rating: {course_row['workload_rating']}")
        print(f"💬 Reviews: {len(course_row['all_reviews'])} characters")
        if course_row['all_reviews']:
            print(f"   Preview: {course_row['all_reviews'][:150]}...")
        print(f"📊 Average Grades: {course_row['avg_grades']}")
        if course_row['avg_grades']:
            for semester, grade in course_row['avg_grades'].items():
                print(f"   {semester}: {grade}")
        print("=" * 80)

    return course_row

def extract_title(data):
    """
    Extract course title from data.

    Args:
        data: Dictionary with course data

    Returns:
        Course title as string or empty string
    """
    try:
        return data.get('information', {}).get('title', '').strip()
    except Exception as e:
        logger.error(f"Error extracting title: {e}")
        return ""
def main(folder_path, output_csv='courses_data_before_llm.csv', max_courses=None, debug=False):
    """
    Main function to process all course JSON files and create dataset.

    Args:
        folder_path: Path to folder containing JSON files
        output_csv: Output CSV filename
        max_courses: Maximum number of courses to process (None for all)
        debug: If True, print detailed extraction info for each course
    """
    logger.info(f"Starting course data extraction from {folder_path}")
    if max_courses:
        logger.info(f"⚠️  DEBUG MODE: Processing only {max_courses} courses")
    if debug:
        logger.info(f"🐛 DEBUG MODE: Detailed output enabled")

    # Get all JSON files
    json_files = get_all_json_files(folder_path)

    if not json_files:
        logger.error("No JSON files found. Exiting.")
        return

    # Limit number of files if max_courses specified
    if max_courses:
        json_files = json_files[:max_courses]
        logger.info(f"Processing {len(json_files)} out of total available files")

    # Extract data from all files
    all_courses = []
    for idx, filepath in enumerate(json_files, 1):
        logger.info(f"[{idx}/{len(json_files)}] Processing {filepath.name}")
        course_data = extract_course_data(filepath, debug=debug)

        if course_data:
            all_courses.append(course_data)

    # Create DataFrame
    df = pd.DataFrame(all_courses)

    # Convert prerequisites list to JSON string for CSV storage
    df['prerequisites'] = df['prerequisites'].apply(json.dumps)

    # Convert avg_grades dict to JSON string for CSV storage
    df['avg_grades'] = df['avg_grades'].apply(lambda x: json.dumps(x, ensure_ascii=False))


    # Print summary statistics
    logger.info("\n=== Extraction Summary ===")
    logger.info(f"Total courses processed: {len(df)}")
    logger.info(f"\nNon-null counts per column:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        if col == 'prerequisites':
            # Count non-empty arrays
            non_empty = sum(1 for x in df[col] if x != '[]')
            logger.info(f"  {col}: {non_empty} courses have prerequisites")
        elif col == 'avg_grades':
            # Count non-empty dicts
            non_empty = sum(1 for x in df[col] if x != '{}')
            logger.info(f"  {col}: {non_empty} courses have grade history")
        elif col in ['description', 'all_reviews']:
            # Count non-empty strings
            non_empty = sum(1 for x in df[col] if x)
            logger.info(f"  {col}: {non_empty} non-empty")
        else:
            logger.info(f"  {col}: {non_null}")

    # Save to CSV
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    logger.info(f"\nDataset saved to {output_csv}")

    # Display first few rows
    logger.info("\n=== Sample Data ===")
    print(df.head())

    return df


if __name__ == "__main__":
    # Example usage - replace with your folder path



    folder_path = r"C:\Users\razbi\PycharmProjects\cheesespoon\cheese-spoon\scraping\scraped_courses" # Current directory, change as needed

    # Debug mode: process only 3 courses with detailed output
    # df = main(folder_path, max_courses=30, debug=True)

    # Production mode: process all courses
    df = main(folder_path)
    print(f'Length of df {df.info}, cols {df.columns}')