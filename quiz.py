import streamlit as st
import pandas as pd
import random
import time
import base64
from streamlit_autorefresh import st_autorefresh

df = pd.read_csv("mcqs.csv")

if "current_difficulty" not in st.session_state:
    st.session_state.current_difficulty = "Medium"
if "recent_topics" not in st.session_state:
    st.session_state.recent_topics = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None

st_autorefresh(interval=1000, key="timer_autorefresh")

def update_difficulty():
    history = st.session_state.question_history[-3:]
    if len(history) > 0:
        avg_correct = sum(1 for q in history if q.get("was_correct", False)) / len(history)
        mapping = {"Easy": 1, "Medium": 2, "Hard": 3}
        reverse_mapping = {1: "Easy", 2: "Medium", 3: "Hard"}
        current_level = mapping[st.session_state.current_difficulty]
        if avg_correct > 0.8 and current_level < 3:
            new_level = current_level + 1
        elif avg_correct < 0.5 and current_level > 1:
            new_level = current_level - 1
        else:
            new_level = current_level
        st.session_state.current_difficulty = reverse_mapping[new_level]

def choose_next_question():
    candidates = df[
        (df["difficulty"] == st.session_state.current_difficulty) &
        (~df["topic"].isin(st.session_state.recent_topics))
    ]
    if candidates.empty:
        candidates = df[df["difficulty"] == st.session_state.current_difficulty]
    if candidates.empty:
        candidates = df.copy()
    chosen = candidates.sample(n=1).iloc[0]
    return chosen

if st.session_state.total_questions >= 10:
    st.header("Quiz Complete!")
    st.write(f"Your final score is: **{st.session_state.score} / {st.session_state.total_questions}**")

    history_df = pd.DataFrame(st.session_state.question_history)[["question", "user_response", "correct_answer"]]
    st.write("Review of your responses:")
    st.write(history_df)

    csv = history_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    download_link = (
        f'<a id="download_csv" href="data:file/csv;base64,{b64}" download="quiz_responses.csv"></a>'
        f'<script>document.getElementById("download_csv").click();</script>'
    )
    st.markdown(download_link, unsafe_allow_html=True)
    st.stop()

if st.session_state.current_question is None:
    st.session_state.current_question = choose_next_question()
    st.session_state.start_time = time.time()

elapsed_time = time.time() - st.session_state.start_time
remaining_time = max(0, 60 - int(elapsed_time))

if remaining_time <= 0:
    st.warning("Time's up! Moving to the next question.")
    correct_label = st.session_state.current_question["correct_answer"]
    correct_option = st.session_state.current_question[f"option_{correct_label}"]

    st.session_state.question_history.append({
        "question": st.session_state.current_question["question"],
        "user_response": "No Answer (Timed Out)",
        "correct_answer": correct_option,
        "was_correct": False,
        "topic": st.session_state.current_question["topic"]
    })

    st.session_state.total_questions += 1
    update_difficulty()

    st.session_state.recent_topics.append(st.session_state.current_question["topic"])
    if len(st.session_state.recent_topics) > 3:
        st.session_state.recent_topics.pop(0)

    st.session_state.current_question = choose_next_question()
    st.session_state.start_time = time.time()
    st.experimental_rerun()

st.header("Dynamic Quiz")
st.write(f"**Question {st.session_state.total_questions + 1} of 10**")
st.write(f"**Topic:** {st.session_state.current_question['topic']}")
st.write(st.session_state.current_question["question"])
st.write(f"**Time Remaining:** {remaining_time} seconds")

options = [
    st.session_state.current_question["option_A"],
    st.session_state.current_question["option_B"],
    st.session_state.current_question["option_C"],
    st.session_state.current_question["option_D"],
]
selected = st.radio("Select your answer:", options)

if st.button("Submit Answer"):
    st.session_state.total_questions += 1
    correct_label = st.session_state.current_question["correct_answer"]
    correct_option = st.session_state.current_question[f"option_{correct_label}"]

    if selected.strip() == correct_option.strip():
        st.success("Correct!")
        st.session_state.score += 1
        was_correct = True
    else:
        st.error(f"Incorrect! The correct answer was: {correct_option}")
        was_correct = False

    st.session_state.question_history.append({
        "question": st.session_state.current_question["question"],
        "user_response": selected,
        "correct_answer": correct_option,
        "was_correct": was_correct,
        "topic": st.session_state.current_question["topic"]
    })

    st.session_state.recent_topics.append(st.session_state.current_question["topic"])
    if len(st.session_state.recent_topics) > 3:
        st.session_state.recent_topics.pop(0)

    update_difficulty()

    st.session_state.current_question = choose_next_question()
    st.session_state.start_time = time.time()
    st.experimental_rerun()