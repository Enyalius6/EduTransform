# mcqgenerator.py

import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
import time
import pandas as pd

def generate_mcqs():
    # ----------------------------
    # Load the FAISS index and document metadata
    # ----------------------------
    with open('faiss_index.pkl', 'rb') as f:
        faiss_index = pickle.load(f)
    with open('documents.pkl', 'rb') as f:
        documents_metadata = pickle.load(f)  # Each entry is assumed to be a dict with keys like "text" and "page"

    # ----------------------------
    # Initialize the Sentence-BERT model for embedding queries (if needed)
    # ----------------------------
    sbert_model = SentenceTransformer('all-mpnet-base-v2')

    # ----------------------------
    # Initialize the Gemini Client with your API key
    # ----------------------------
    gemini_api_key = "AIzaSyAbObOCEEZslqSCZohXX4tMGqsZP_grPz4"
    client = genai.Client(api_key=gemini_api_key)

    # ----------------------------
    # Function to generate MCQs from a given context
    # ----------------------------
    def generate_mcqs_from_topic(context):
        prompt = (
            "You are an educator specialized in creating multiple-choice questions (MCQs). "
            "Generate three MCQs based solely on the following context. "
            "The first question should be easy, the second question should be of medium difficulty, "
            "and the third question should be hard. "
            "For each question, in addition to the difficulty, question text, and options, also provide a topic that best represents the key subject area of the question. "
            "Each question should include four answer options labeled A, B, C, and D. ONLY 1 option MUST BE correct, The other options should be decieving/plausible but should be incorrect. "
            "FOCUS on both the quality of the questions and the options that are being generated. "
            "Do not include any explanations or reference the context in the questions themselves. Create questions that are general and concept-based. "
            "Format the output as follows for each question:\n\n"
            "Difficulty: <Easy/Medium/Hard>\n"
            "Topic: <topic text>\n"
            "Question: <question text>\n"
            "A) <option text>\n"
            "B) <option text>\n"
            "C) <option text>\n"
            "D) <option text>\n"
            "Answer: <the option label that is correct>\n\n"
            "Context:\n" + context
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        answer = response.text.strip() if response.text else "No questions generated."
        return answer

    # ----------------------------
    # Parser for Gemini Output
    # ----------------------------
    def parse_mcqs(gemini_output, chunk_id, page_info):
        mcq_list = []
        # Split output into individual questions by double newlines
        questions = gemini_output.strip().split("\n\n")
        for q in questions:
            lines = [line.strip() for line in q.splitlines() if line.strip()]
            if len(lines) < 7:
                continue  # Skip if not enough lines for a full MCQ
            mcq = {"chunk": chunk_id, "page": page_info}
            for line in lines:
                # If the line starts with an asterisk, remove it and mark that option as correct.
                star_present = False
                if line.startswith("*"):
                    star_present = True
                    line = line[1:].strip()
                if line.startswith("Difficulty:"):
                    mcq["difficulty"] = line.split("Difficulty:")[-1].strip()
                elif line.startswith("Topic:"):
                    mcq["topic"] = line.split("Topic:")[-1].strip()
                elif line.startswith("Question:"):
                    mcq["question"] = line.split("Question:")[-1].strip()
                elif line.startswith("A)"):
                    text = line.split("A)")[1].strip()
                    mcq["option_A"] = text
                    if star_present:
                        mcq["correct_answer"] = "A"
                elif line.startswith("B)"):
                    text = line.split("B)")[1].strip()
                    mcq["option_B"] = text
                    if star_present:
                        mcq["correct_answer"] = "B"
                elif line.startswith("C)"):
                    text = line.split("C)")[1].strip()
                    mcq["option_C"] = text
                    if star_present:
                        mcq["correct_answer"] = "C"
                elif line.startswith("D)"):
                    text = line.split("D)")[1].strip()
                    mcq["option_D"] = text
                    if star_present:
                        mcq["correct_answer"] = "D"
                elif line.startswith("Answer:"):
                    # This will override any previous setting if present.
                    mcq["correct_answer"] = line.split("Answer:")[-1].strip()
            mcq_list.append(mcq)
        return mcq_list

    # ----------------------------
    # Main loop: Generate MCQs for each document chunk and store in a CSV file
    # ----------------------------
    all_mcqs = []  # This will hold dictionaries for every MCQ generated
    print("Generating MCQs from all topics present in the documents...\n")
    for idx, doc_entry in enumerate(documents_metadata):
        context = doc_entry.get("text", "")
        page_info = doc_entry.get("page", "Unknown")
        print(f"--- Generating MCQs for Document Chunk {idx+1} (Page: {page_info}) ---\n")
        gemini_output = generate_mcqs_from_topic(context)
        mcqs_for_chunk = parse_mcqs(gemini_output, chunk_id=idx+1, page_info=page_info)
        all_mcqs.extend(mcqs_for_chunk)
        print(gemini_output)
        print("\n" + "="*80 + "\n")
        time.sleep(2)  # Adjust sleep to avoid rate-limiting if needed

    # Save the structured MCQs into a CSV file
    df_mcqs = pd.DataFrame(all_mcqs)
    df_mcqs.to_csv("mcqs.csv", index=False)
    print("MCQs successfully saved to mcqs.csv")

if __name__ == "__main__":
    generate_mcqs()
