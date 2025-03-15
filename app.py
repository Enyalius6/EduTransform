from flask import Flask, render_template, request, redirect, url_for, flash, session
import os, time, random
import pandas as pd
from datatovectors import process_pdf
from mcqgenerator import generate_mcqs
from content_generator import process_student_responses , generate_beautiful_pdf_report , generate_summary_report

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash and session

# Configure an upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ------------------- Existing Routes -------------------

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Clear previous quiz session data when a new file is being uploaded.
        keys_to_clear = [
            "quiz_started", "current_difficulty", "recent_topics", "score",
            "total_questions", "question_history", "current_question", "start_time", "quiz_available"
        ]
        for key in keys_to_clear:
            session.pop(key, None)
        
        if 'pdf_file' not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)
        if file and file.filename.lower().endswith('.pdf'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'test2.pdf')
            file.save(file_path)
            try:
                process_pdf(file_path)  # Process PDF and generate new pickle files.
                flash("PDF processed successfully!")
                generate_mcqs()         # Generate new MCQs (creates a new mcqs.csv).
                flash("MCQs generated successfully!")
                session["quiz_available"] = True
            except Exception as e:
                flash(f"An error occurred: {e}")
            return redirect(url_for('upload'))
        else:
            flash("Invalid file format. Please upload a PDF.")
            return redirect(request.url)
    else:
        # For a GET request, do NOT clear the quiz_available flag.
        quiz_available = session.get("quiz_available", False)
    return render_template("upload.html", quiz_available=quiz_available)

# ------------------- Adaptive Quiz Routes -------------------

# Helper functions for quiz logic
def load_quiz_dataframe():
    return pd.read_csv("mcqs.csv")

def update_difficulty_in_session():
    # Look at the last 3 questions
    history = session['question_history'][-3:]
    if len(history) > 0:
        avg_correct = sum(1 for q in history if q.get("was_correct", False)) / len(history)
        mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
        reverse_mapping = {1: "Easy", 2: "Medium", 3: "Hard"}
        current_level = mapping[session['current_difficulty']]
        if avg_correct > 0.8 and current_level < 3:
            new_level = current_level + 1
        elif avg_correct < 0.5 and current_level > 1:
            new_level = current_level - 1
        else:
            new_level = current_level
        session['current_difficulty'] = reverse_mapping[new_level]

def choose_next_question(df):
    # Select a random question based on the current difficulty and not in recent topics
    candidates = df[
        (df["difficulty"] == session['current_difficulty']) &
        (~df["topic"].isin(session['recent_topics']))
    ]
    if candidates.empty:
        candidates = df[df["difficulty"] == session['current_difficulty']]
    if candidates.empty:
        candidates = df.copy()
    chosen = candidates.sample(n=1).iloc[0]
    return chosen.to_dict()  # Convert row to dict

@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    df = load_quiz_dataframe()
    # Initialize session state for the quiz if not already done
    if "quiz_started" not in session:
        session["quiz_started"] = True
        session["current_difficulty"] = "Medium"
        session["recent_topics"] = []  # to avoid repeating topics
        session["score"] = 0
        session["total_questions"] = 0
        session["question_history"] = []
        session["current_question"] = None
        session["start_time"] = time.time()

    # POST request: Process submitted answer
    if request.method == "POST":
        user_answer = request.form.get("answer", "")
        elapsed = time.time() - session["start_time"]
        if elapsed > 60:
            user_answer = "No Answer (Timed Out)"
        
        # Safely retrieve correct answer information
        correct_label = session["current_question"].get("correct_answer")
        if not correct_label or str(correct_label).strip().lower() == "nan":
            correct_option = "No correct answer provided"
            was_correct = False
        else:
            correct_option = session["current_question"].get(f"option_{correct_label}")
            was_correct = (user_answer.strip() == correct_option.strip())
        
        # Record the response in the question history
        session["question_history"].append({
            "question": session["current_question"]["question"],
            "user_response": user_answer,
            "correct_answer": correct_option,
            "was_correct": was_correct,
            "topic": session["current_question"]["topic"]
        })
        session["total_questions"] += 1
        if was_correct:
            session["score"] += 1

        # Update recent topics and adjust difficulty (see your update_difficulty_in_session function)
        session["recent_topics"].append(session["current_question"]["topic"])
        if len(session["recent_topics"]) > 3:
            session["recent_topics"].pop(0)
        update_difficulty_in_session()

        if session["total_questions"] >= 10:
            return redirect(url_for("quiz_result"))
        else:
            session["current_question"] = choose_next_question(df)
            session["start_time"] = time.time()
            return redirect(url_for("quiz"))
    else:
        # GET request: if no current question, choose one.
        if session["current_question"] is None:
            session["current_question"] = choose_next_question(df)
            session["start_time"] = time.time()
        elapsed = time.time() - session["start_time"]
        remaining_time = max(0, 60 - int(elapsed))
        return render_template("quiz.html",
                               current_question=session["current_question"],
                               current_difficulty=session["current_difficulty"],
                               total_questions=session["total_questions"],
                               remaining_time=remaining_time,
                               score=session["score"])

@app.route("/quiz_result")
def quiz_result():
    question_history = session.get("question_history", [])
    score = session.get("score", 0)
    total = session.get("total_questions", 0)
    return render_template("quiz_result.html", score=score, total=total, question_history=question_history)
@app.route("/new_quiz")
def new_quiz():
    # Clear quiz-related keys from session.
    keys_to_clear = ["quiz_started", "current_difficulty", "recent_topics", "score",
                     "total_questions", "question_history", "current_question", "start_time"]
    for key in keys_to_clear:
        session.pop(key, None)
    return redirect(url_for("quiz"))

@app.route("/study_materials", methods=["POST"])
def study_materials():
    # Get user inputs from the form
    try:
        complexity = int(request.form.get("complexity"))
        length = int(request.form.get("length"))
        real_life_examples = int(request.form.get("real_life_examples"))
    except (ValueError, TypeError):
        flash("Please enter valid integers between 1 and 5 for all preferences.")
        return redirect(url_for("quiz_result"))
    
    user_preferences = {
        "complexity": complexity,
        "length": length,
        "real_life_examples": real_life_examples
    }
    
    # Retrieve quiz responses from session (stored in question_history)
    responses = session.get("question_history", [])
    if not responses:
        flash("No quiz responses found. Please take the quiz first.")
        return redirect(url_for("quiz"))
    
    responses_df = pd.DataFrame(responses)
    
    # Process responses and generate personalized feedback
    analysis_df = process_student_responses(responses_df, user_preferences)
    summary = generate_summary_report(analysis_df)
    
    # Ensure the static/reports folder exists
    reports_dir = os.path.join(app.root_path, "static", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    output_filename = os.path.join("static", "reports", "refined_student_analysis.pdf")
    generate_beautiful_pdf_report(analysis_df, summary, output_filename=output_filename)
    
    # Render the study materials result page with a correct download link:
    return render_template("study_materials_result.html",
                           output_filename="reports/refined_student_analysis.pdf",
                           summary=summary)
if __name__ == "__main__":
    app.run(debug=True)
