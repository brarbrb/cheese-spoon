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
    embedd_query = model.encode(query,
                              convert_to_numpy=True,
                              show_progress_bar=False,
                              device=device,
                              normalize_embeddings=True)
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
    excluded_set = set(courses_list)
    filtered_courses = []

    for match in response.matches:
        print(f'title: {match["metadata"]["title"]} | pre : {json.loads(match["metadata"]["prerequisites"])} len pre {len(json.loads(match["metadata"]["prerequisites"]))}')
        prereq= json.loads(match["metadata"]["prerequisites"])

        can_take_it = check_prerequisites(courses_list,prereq)
        if can_take_it:
            # match.id is the Course ID (assuming you used it as the vector ID)
            if no_exam == True and match["metadata"]["moed_a"] != "":
                continue
            if min_credits > match["metadata"]["credits"]:
                continue
            if match.ID not in excluded_set:
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
            print(f'Cannot take {match["metadata"]["title"]}')
    return filtered_courses

def get_all_untaken_courses_with_requirements(semester_name="WINTER_2025_2026",courses_list=[],no_exam=False,min_credits=0,user_query=""):
    # Get all untaken courses with filters and semantic search
    # Retrive all courses that are not in the student's completed courses list

    index = get_index_by_semester(semester_name=semester_name)
    # 1. Get index dimensions (needed to create the dummy vector)
    stats = index.describe_index_stats()
    if user_query == "":
        embedded_query = [0.0] * stats['dimension'] # dummy vector (no user query)
    else:
        embedded_query = embed_query(user_query)

    # 2. Get EVERYTHING (The "Dump")
    # We set top_k high (max 10,000) to ensure we grab the whole semester catalog but according to semantic search
    response = index.query(
        vector=embedded_query,
        top_k=10000,
        include_metadata=True
    )

    filtered_courses = filter_according_to_requirements_and_untaken_and_prereq(response,courses_list,no_exam,min_credits)
    filtered_courses = pd.DataFrame(filtered_courses)

    return filtered_courses

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
    print(f'Before rerank')
    filtered_courses = get_all_untaken_courses_with_requirements(semester_name,courses_list,no_exam,min_credits,user_query)

    print(filtered_courses.head(10)[['title','avg_grade_all_sem',"prerequisites"]])
    print('*'*100)
    print('after rerank')
    reranked_courses = rerank(filtered_courses,semantic_weight,credits_weight,avg_grade_weight,workload_rating_weight,general_rating_weight)

    print(reranked_courses.head(10)[['title','avg_grade_all_sem',"prerequisites"]])

# recommend_courses(courses_list=['02340221'])