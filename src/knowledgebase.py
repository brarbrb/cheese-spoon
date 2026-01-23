import os
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import json

DEFAULT_AVG_GRADE = 60

def get_pinecone():
    # 1. Load variables from the .env file
    load_dotenv()

    # 2. Get the key securely
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError("No API key found. Please check your .env file.")

    # 3. Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    return pc
def get_index_by_semester(semester_name):
    pc = get_pinecone()
    kb_name = os.getenv(semester_name)
    if not kb_name:
        raise ValueError("No API key found. Please check your .env file.")
    index = pc.Index(host=kb_name)
    return index

def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected. Consider enabling GPU in Runtime -> Change runtime type")
    return device
def get_embedding_model():
    device = get_device()
    model_name = os.getenv("EMBEDDING_MODEL")
    model = SentenceTransformer(model_name, device=device)
    return model
def embed_query(query):
    device= get_device()
    model = get_embedding_model()
    embedd_query = model.encode(
        "query: " + query,
        convert_to_numpy=True,
        show_progress_bar=False,
        device=device,
        normalize_embeddings=True
    )
    return embedd_query

def check_prerequisites(courses_list,prerequisites):
    if len(prerequisites) == 0:
        return True
    # Convert to set for O(1) lookup
    completed_set = set(courses_list)

    # Check each prerequisite combination
    for combo in prerequisites:
        # Check if ALL courses in this combo are completed
        if all(course in completed_set for course in combo):
            return True  # Found a satisfied combo!

    return False  # No combo was satisfied
def filter_according_to_requirements_and_untaken_and_prereq(response,courses_list,no_exam,min_credits):
    # 3. Filter Locally in Python
    # Convert list to set for super-fast lookups (O(1) speed)
    excluded_set = set()
    for course_id in courses_list:
        # Normalize: remove leading zeros and store all variants
        print(f'Course id is {str(course_id)}')
        normalized_id = str(course_id).lstrip('0') or '0'  # Keep at least one '0' if all zeros
        print(f'normalized id is {str(normalized_id)}')
        excluded_set.add(normalized_id)
        excluded_set.add('0' + normalized_id)
        excluded_set.add('00' + normalized_id)
        excluded_set.add(str(course_id))  # Also add original format

    print(f"[DEBUG] Excluded set: {excluded_set}")
    filtered_courses = []

    for match in response.matches:
        # print(f'ID: {match.id}, title: {match["metadata"]["title"]} | pre : {json.loads(match["metadata"]["prerequisites"])} len pre {len(json.loads(match["metadata"]["prerequisites"]))}')
        prereq= json.loads(match["metadata"]["prerequisites"])

        can_take_it = check_prerequisites(courses_list,prereq)
        if can_take_it:
            # match.id is the Course ID (assuming you used it as the vector ID)
            if no_exam == True and match["metadata"]["moed_a"] != "":
                continue
            if min_credits > match["metadata"]["credits"]:
                continue



            # Check if already taken
            if str(match.id) in excluded_set:
                print(f'[DEBUG] EXCLUDED: {match.id} (normalized: {match.id})')
                continue

            grades_dict=json.loads(match["metadata"]["avg_grades"])

            if len(grades_dict)>0:
                average = sum(grades_dict.values()) / len(grades_dict)
            else:
                average=DEFAULT_AVG_GRADE

            # Prepare the data object

            course_data = match.metadata
            course_data['ID'] = match.id
            course_data["semantic_score"] = match.score
            course_data["avg_grade_all_sem"] = average

            filtered_courses.append(course_data)
        else:
            print(f'Cannot take {match["id"]}')
    return filtered_courses


def get_all_untaken_courses_with_requirements(semester_name="WINTER_2025_2026", courses_list=[], no_exam=False,
                                              min_credits=0, user_query=""):
    print(f"[DEBUG] Starting query with: user_query='{user_query}'")

    try:
        # Get index
        print(f"[DEBUG] Getting index for semester: {semester_name}")
        index = get_index_by_semester(semester_name=semester_name)
        print(f"[DEBUG] Index obtained successfully")

        # Get index dimensions
        print(f"[DEBUG] Describing index stats...")
        stats = index.describe_index_stats()
        print(f"[DEBUG] Index stats: {stats}")

        # Create query vector
        if user_query == "":
            print(f"[DEBUG] Creating dummy vector of dimension {stats['dimension']}")
            embedded_query = [0.0] * stats['dimension']
        else:
            print(f"[DEBUG] Embedding query: '{user_query}'")
            embedded_query = embed_query(user_query)
            print(
                f"[DEBUG] Embedded query shape: {embedded_query.shape if hasattr(embedded_query, 'shape') else len(embedded_query)}")
            # Convert numpy array to list if needed
            if hasattr(embedded_query, 'tolist'):
                embedded_query = embedded_query.tolist()

        print(f"[DEBUG] Embedded query dimension: {len(embedded_query)}")

        # Query Pinecone
        print(f"[DEBUG] Querying Pinecone with top_k=10000...")
        try:
            response = index.query(
                vector=embedded_query,
                top_k=10000,
                include_metadata=True,
                timeout=30  # Add timeout
            )
            print(f"[DEBUG] Query successful! Got {len(response.matches)} matches")
        except Exception as query_error:
            print(f"[ERROR] Pinecone query failed: {query_error}")
            print(f"[ERROR] Query error type: {type(query_error)}")
            raise

        # Filter results
        print(f"[DEBUG] Filtering results...")
        filtered_courses = filter_according_to_requirements_and_untaken_and_prereq(
            response, courses_list, no_exam, min_credits
        )
        print(f"[DEBUG] Filtered to {len(filtered_courses)} courses")

        # Convert to DataFrame
        filtered_courses = pd.DataFrame(filtered_courses)
        print(f"[DEBUG] Returning DataFrame with shape: {filtered_courses.shape}")

        return filtered_courses

    except Exception as e:
        print(f"[ERROR] Exception in get_all_untaken_courses_with_requirements: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise

def rerank(df,semantic_weight=0.2,credits_weight=0.2,avg_grade_weight=0.2,workload_rating_weight=0.2,general_rating_weight=0.2):

    if df.empty:
        return df

    max_credits = df["credits"].max()
    # Create a copy to avoid modifying original
    df_ranked = df.copy()

    # Normalize semantic score (assuming it's already 0-1 from Pinecone)
    # If score doesn't exist (no semantic search), default to 0
    semantic_score = df_ranked["semantic_score"]

    # Normalize credits (divide by max)
    max_credits = df_ranked['credits'].max()
    if max_credits > 0:
        credits_normalized = df_ranked['credits'] / max_credits
    else:
        credits_normalized = 0

    # Normalize avg_grade (divide by 100)
    avg_grade_normalized = df_ranked['avg_grade_all_sem'] / 100

    # Normalize workload_rating (divide by 5)
    workload_normalized = df_ranked['workload_rating'] / 5

    # Normalize general_rating (divide by 5)
    general_rating_normalized = df_ranked['general_rating'] / 5

    # Calculate combined score
    df_ranked['combined_score'] = (
            semantic_weight * semantic_score +
            credits_weight * credits_normalized +
            avg_grade_weight * avg_grade_normalized +
            workload_rating_weight * workload_normalized +
            general_rating_weight * general_rating_normalized
    )

    # Sort by combined score (descending - higher is better)
    df_ranked = df_ranked.sort_values('combined_score', ascending=False).reset_index(drop=True)

    return df_ranked
def recommend_courses(semester_name="WINTER_2025_2026",courses_list=[],no_exam=False,min_credits=0,user_query="",semantic_weight=0.2,credits_weight=0.2,avg_grade_weight=0.2,workload_rating_weight=0.2,general_rating_weight=0.2):
    print(f'User query {user_query}')
    print(f'Before rerank')
    print(f'Courses: {len(courses_list)}')
    print(f'courses {courses_list}')
    filtered_courses = get_all_untaken_courses_with_requirements(semester_name,courses_list,no_exam,min_credits,user_query)

    # print(filtered_courses.head(10)[['title','avg_grade_all_sem',"prerequisites"]])

    reranked_courses = rerank(filtered_courses,semantic_weight,credits_weight,avg_grade_weight,workload_rating_weight,general_rating_weight)

    # print(reranked_courses.head(10)[['title','avg_grade_all_sem',"prerequisites"]])
    return reranked_courses
# recommend_courses(courses_list=['02340221'])