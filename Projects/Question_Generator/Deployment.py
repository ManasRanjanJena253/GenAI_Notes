# Importing dependencies
import streamlit as st
import pandas as pd
import random
from Question_Generator import QuestionGenerator
import os

# Creating a class to handle quiz functionality
class QuizManager:
    def __init__(self):
        # Initializing empty list to store questions, user answers and results
        self.questions = []
        self.user_answers = []
        self.results = []

    def generate_questions(self, generator, topic, question_type, difficulty, num_questions):
        try:
            # Generating specified no. of questions
            for _ in range(num_questions):
                if question_type == "Multiple Choice":
                    question = generator.generate_mcq(topic, difficulty.lower())
                    self.questions.append({
                        'type' : 'MCQ',
                        'question' : question.question,
                        'options' : question.options,
                        'correct_answer' : question.correct_answer
                    })
                else :
                    question = generator.generate_fill_blank(topic, difficulty.lower())
                    self.questions.append({
                        'type' : 'Fill in the Blank',
                        'question' : question.question,
                        'correct_answer' : question.answer
                    })
        except Exception as e:
            # Displaying error if question generation fails.
            st.error(f"Error generating questions : {e}")
            return False
        return True

    def attempt_quiz(self):
        # Displaying questions and collecting user answers
        for i, q in enumerate(self.questions):
            st.markdown(f"**Question {i+1}: {q['question']}**")

            # Handle MCQ input usign radio buttolns
            if q['type'] == 'MCQ':
                user_answer = st.radio(
                    f"Select an answer for Question {i + 1}",
                    q['options'],
                    key = f"mcq_{i}"
                )
                self.user_answers.append(user_answer)
            # Handling fill in the blanks
            else :
                user_answer = st.text_input(
                    f"Fill in the blank for the question {i + 1}",
                    key = f"fill_blank{i}"
                )
                self.user_answers.append(user_answer)

    def evaluate(self):
        self.results = []   # Resetting the previous results if stored any
        for i, (q, user_ans) in enumerate(zip(self.questions, self.user_answers)):
            # Creating a base result dictionary
            result_dict = {
                "question_number" : i + 1,
                "question" : q['question'],
                "question_type" : q['type'],
                "user_answer" : q['correct_answer'],
                "is_correct" : False
            }
            # Evaluating mcq answers
            if q['type'] == 'MCQ':
                result_dict['options'] = q['options']
                result_dict['is_correct'] = user_ans == q['correct_answer']
            # Evaluating fill in the blanks
            else :
                result_dict['options'] = []
                result_dict['is_correct'] = user_ans.strip().lower() == q['correct_answer']

            self.results.append(result_dict)

    def generate_result_df(self):
        """Function to convert the results into pandas dataframe."""
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(self.results)


def main():
    # Configuring streamlit
    st.set_page_config(page_title = "Question Generator")

    # Initializing the session state variables
    if 'quiz_manager' not in st.session_state:
        st.session_state.quiz_manager = QuizManager()
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False

    # Setting page title
    st.title("Question Generator")

    # Creating sidebar for quiz settings
    st.sidebar.header("Quiz Settings")

    # Question type Selection dropdown
    question_type = st.sidebar.selectbox(
        "Select Question Type",
        ['Multiple Choice', "Fill in the blanks"],
        index = 0
    )

    # Topic input
    topic = st.text_input("Enter the topic")

    # Difficulty level selection
    difficulty = st.selectbox("Select the difficulty level",
                              ['Easy', 'Medium', 'Hard'],
                              index = 1)

    # Selecting the number of questions
    num_ques = st.number_input("Enter the number of questions",
                               min_value = 1,
                               max_value = 15,
                               value = 5)

    # Generate quiz button handler
    if st.sidebar.button("Generate Quiz"):
        st.session_state.quiz_submitted = False
        generator = QuestionGenerator()
        st.session_state.quiz_generated = st.session_state.quiz_manager.generate_questions(generator, topic, question_type, difficulty, num_ques)
        st.rerun()

    # Displaying quiz if generated
    if st.session_state.quiz_generated and st.session_state.quiz_manager.questions:
        st.header("Quiz")
        st.session_state.quiz_manager.attempt_quiz()

        # Submit quiz button
        if st.button("Submit Quiz"):
            st.session_state.quiz_manager.evaluate()
            st.session_state.quiz_submitted = True
            st.rerun()

    if st.session_state.quiz_submitted:
        st.header("Quiz Results")
        results_df = st.session_state.quiz_manager.generate_result_df()

main()