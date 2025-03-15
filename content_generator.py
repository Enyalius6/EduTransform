import pickle
import faiss
import numpy as np
import time
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from google import genai
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
)
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# ------------------------------
# Helper: Markdown to HTML Conversion
# ------------------------------
def convert_markdown_to_html(text):
    """
    Convert markdown-style bold markers **text** into HTML <b>text</b> tags,
    and replace newline characters with HTML <br /> tags for proper paragraphing.
    """
    pattern = re.compile(r"\*\*(.*?)\*\*")
    text = pattern.sub(r"<b>\1</b>", text)
    # Replace newline characters with <br /> tags
    text = text.replace("\n", "<br />")
    return text

# ------------------------------
# Load Models, FAISS Index & Data
# ------------------------------

# Load the FAISS index and document metadata
with open('faiss_index.pkl', 'rb') as f:
    faiss_index = pickle.load(f)
with open('documents.pkl', 'rb') as f:
    documents_metadata = pickle.load(f)

# Initialize SBERT and Gemini
sbert_model = SentenceTransformer('all-mpnet-base-v2')
gemini_api_key = "AIzaSyAbObOCEEZslqSCZohXX4tMGqsZP_grPz4"
client = genai.Client(api_key=gemini_api_key)

# ------------------------------
# Core Functions
# ------------------------------

def get_context_for_question(question_text, documents_metadata):
    """
    Find the most relevant document context for a given question using SBERT embeddings.
    Returns a tuple of (context_text, page_number).
    """
    question_embedding = sbert_model.encode([question_text])[0]
    question_embedding = np.float32(question_embedding)
    question_embedding = np.expand_dims(question_embedding, axis=0)
    
    D, I = faiss_index.search(question_embedding, k=1)
    relevant_doc = documents_metadata[I[0][0]]
    return relevant_doc['text'], relevant_doc['page']

def analyze_student_response(question, student_response, correct_answer, context, preferences):
    """
    Generate a personalized explanation based on the student's response, context,
    and user preferences.
    """
    # Define mappings for clarity in the prompt
    complexity_mapping = {
        1: "very simple",
        2: "simple",
        3: "moderately complex",
        4: "complex",
        5: "highly complex"
    }
    length_mapping = {
        1: "very short",
        2: "short",
        3: "moderate",
        4: "long",
        5: "very long"
    }
    examples_mapping = {
        1: "no real-life examples",
        2: "a small real-life example",
        3: "one real-life example",
        4: "one real-life example",
        5: "two detailed real-life examples"
    }

    # Use preferences provided by the user (defaults are set to 3)
    complexity = complexity_mapping.get(preferences.get("complexity", 3))
    length = length_mapping.get(preferences.get("length", 3))
    examples = examples_mapping.get(preferences.get("real_life_examples", 3))

    # Build the prompt with user preferences
    prompt = f"""
As an educational assistant, analyze this student's response and provide a clean, direct, and concise explanation.
Please ensure the explanation is {complexity} in complexity/depth of knowledge, {length} in length, and includes {examples}.

Question: {question}
Student's Response: {student_response}
Correct Answer: {correct_answer}
Context from Learning Material: {context}

Explain the concept related to the question and context, balancing detail and brevity.
Provide only the essential explanation without prefatory comments.
Focus on the quality of the explanation.
If there is an exceptionally good fun fact related to the concept, include it.
Start the real-life example and fun fact on a new line for clarity.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text.strip()

def process_student_responses(responses_df, user_preferences):
    """
    Process all student responses and generate personalized feedback.
    Expects responses_df to have columns: 'question', 'user_response', 'correct_answer'.
    Returns a DataFrame with detailed analysis.
    """
    analysis_results = []

    for index, row in responses_df.iterrows():
        # Retrieve context for the question
        context, page_num = get_context_for_question(row['question'], documents_metadata)

        # Generate personalized explanation
        explanation = analyze_student_response(
            row['question'],
            row['user_response'],
            row['correct_answer'],
            context,
            user_preferences  # Use user-specified preferences
        )

        analysis_results.append({
            'question': row['question'],
            'student_response': row['user_response'],
            'correct_answer': row['correct_answer'],
            'page_number': page_num,
            'personalized_explanation': explanation,
            'is_correct': row['user_response'].strip() == row['correct_answer'].strip()
        })

        # Pause briefly to avoid rate limiting with the Gemini API
        time.sleep(2)

    return pd.DataFrame(analysis_results)

def generate_summary_report(analysis_df):
    """
    Generate a summary report of the student's performance.
    Returns a dictionary with summary statistics.
    """
    total_questions = len(analysis_df)
    correct_answers = sum(analysis_df['is_correct'])
    performance = (correct_answers / total_questions) * 100 if total_questions > 0 else 0

    topics_needing_review = analysis_df[~analysis_df['is_correct']]['page_number'].tolist()

    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'performance_percentage': performance,
        'pages_needing_review': sorted(set(topics_needing_review))
    }

def generate_beautiful_pdf_report(analysis_df, summary, output_filename="refined_student_analysis.pdf"):
    """
    Generate a refined PDF report that is well-structured, visually appealing,
    and easy to read. The report includes a summary and detailed analysis.
    """
    doc = SimpleDocTemplate(output_filename,
                            pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()

    # Define custom styles for title, headings, and normal text
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=24,
        leading=28,
        alignment=1,  # center-aligned
        spaceAfter=24,
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=18,
        leading=22,
        spaceAfter=12,
    )
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['BodyText'],
        fontSize=12,
        leading=16,
        spaceAfter=10,
    )

    story = []

    # Title Page
    story.append(Paragraph("Student Performance Analysis", title_style))
    story.append(Spacer(1, 0.5 * inch))

    # Summary Section
    story.append(Paragraph("Summary", heading_style))
    summary_data = [
        ["Total Questions:", summary['total_questions']],
        ["Correct Answers:", summary['correct_answers']],
        ["Performance:", f"{summary['performance_percentage']:.2f}%"],
        ["Pages Needing Review:", ", ".join(map(str, summary['pages_needing_review'])) if summary['pages_needing_review'] else "None"]
    ]
    table = Table(summary_data, colWidths=[200, 250])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    story.append(table)
    story.append(PageBreak())

    # Detailed Analysis Section
    story.append(Paragraph("Detailed Analysis", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # For each question, add a clearly formatted block with explanation and correctness
    for idx, row in analysis_df.iterrows():
        story.append(Paragraph(f"<b>Question:</b> {row['question']}", normal_style))
        story.append(Paragraph(f"<b>Your Response:</b> {row['student_response']}", normal_style))
        story.append(Paragraph(f"<b>Correct Answer:</b> {row['correct_answer']}", normal_style))
        story.append(Paragraph(f"<b>Page Number:</b> {row['page_number']}", normal_style))
        # Convert markdown-style bold markers in the explanation to HTML tags
        personalized_explanation_html = convert_markdown_to_html(row['personalized_explanation'])
        story.append(Paragraph(f"<b>Personalized Explanation:</b> {personalized_explanation_html}", normal_style))
        story.append(Paragraph(f"<b>Is Correct:</b> {row['is_correct']}", normal_style))
        story.append(Spacer(1, 0.3 * inch))

    doc.build(story)
    print(f"Beautiful PDF generated: {output_filename}")

# ------------------------------
# Optional: Standalone Testing
# ------------------------------
if __name__ == "__main__":
    # For standalone testing, you can simulate reading a CSV of student responses.
    # This block will only run if you execute this file directly.
    responses_df = pd.read_csv('Responses.csv')  # Make sure this file exists in the current directory.
    
    # Instead of interactive terminal input, here you could hard-code preferences for testing.
    user_preferences = {
        "complexity": 3,
        "length": 3,
        "real_life_examples": 3
    }
    
    print("Processing student responses...")
    analysis_df = process_student_responses(responses_df, user_preferences)
    summary = generate_summary_report(analysis_df)
    generate_beautiful_pdf_report(analysis_df, summary, output_filename="refined_student_analysis.pdf")
    print("PDF report generated successfully.")
