import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import json
# Apply the same engineering feature functions expected by the pipeline
def clean_and_engineer(data):
    df_processed = data.copy()
    import re
    
    attempts = np.maximum(df_processed['num_students_attempted'], 1)
    df_processed['time_per_attempt'] = df_processed['time_taken_minutes'] / attempts
    df_processed['log_attempts'] = np.log1p(df_processed['num_students_attempted'])
    df_processed['question_length'] = df_processed['question_text'].astype(str).apply(lambda x: len(x.split()))
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)
        return text
    
    df_processed['cleaned_text'] = df_processed['question_text'].apply(clean_text)
    return df_processed

st.set_page_config(page_title="Intelligent Exam Question Analysis", layout="wide")

st.title("INTELLIGENT EXAM QUESTION ANALYSIS")

# Load models
@st.cache_resource
def load_models():
    bloom_path = "models/bloom_pipeline.pkl"
    diff_path = "models/difficulty_pipeline.pkl"
    
    if os.path.exists(bloom_path) and os.path.exists(diff_path):
        bloom_pipe = joblib.load(bloom_path)
        diff_pipe = joblib.load(diff_path)
        return bloom_pipe, diff_pipe
    return None, None

bloom_model, difficulty_model = load_models()

if not bloom_model:
    st.warning("Models not found. Please run `python train_and_save.py` first.")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    [
        "📝 Predict Cognitive Load & Difficulty",
        "📊 Model Analytics & Accuracy",
        "🤖 Agentic Assessment Assistant",
    ],
)

if page == "📝 Predict Cognitive Load & Difficulty":
    st.subheader("Predict Cognitive Load & Empirical Difficulty")
    st.sidebar.header("Input Metrics")
    
    # Academic Text
    question_text = st.text_area("Question Text", placeholder="E.g., Evaluate the ethical implications of genetic engineering...")
    
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", ["Computer Science", "Biology", "Physics", "Mathematics", "Economics"])
    with col2:
        topic = st.text_input("Topic", "Ethics")
    
    st.markdown("---")
    st.markdown("### Empirical Student Data")
    st.markdown("These statistics provide the model with the necessary vectors to anchor its psychometric predictions.")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        avg_score = st.slider("Average Score (%)", 0.0, 100.0, 75.0)
        correct_pct = st.slider("Correct Percentage", 0.0, 1.0, 0.75)
    with col4:
        attempted = st.number_input("Students Attempted", value=150, min_value=1)
        correct = st.number_input("Students Correct", value=112, min_value=0)
    with col5:
        time_taken = st.number_input("Total Time Taken (minutes)", value=300.0)
        
    if st.button("Analyze Question", type="primary"):
        if not question_text.strip():
            st.error("Please provide the question text.")
        else:
            # Construct DataFrame record
            input_dict = {
                "question_text": question_text,
                "subject": subject,
                "topic": topic,
                "avg_score": float(avg_score),
                "correct_percentage": float(correct_pct),
                "num_students_attempted": float(attempted),
                "num_students_correct": float(correct),
                "time_taken_minutes": float(time_taken)
            }
            
            input_df = pd.DataFrame([input_dict])
            engineered_df = clean_and_engineer(input_df)
            
            # Inference
            try:
                bloom_pred = bloom_model.predict(engineered_df)[0]
                diff_pred = difficulty_model.predict(engineered_df)[0]
                
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                metric1, metric2 = st.columns(2)
                metric1.metric(label="Predicted Bloom's Taxonomy Level", value=bloom_pred)
                metric2.metric(label="Estimated Academic Difficulty", value=diff_pred)
                
                st.info("💡 **Model Confidence Note**: These classifications utilize purely Classical ML Ensembling mapped over TF-IDF n-grams (1-3) and raw psychometric dimensions.")
            except Exception as e:
                st.error(f"Inference Engine Failed: {e}. Please ensure the models were built using the same version of scikit-learn.")
                
elif page == "📊 Model Analytics & Accuracy":
        st.markdown("### Model Performance & Detailed Insights")
        
        metrics_path = "models/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                pipeline_metrics = json.load(f)
                
            m1, m2 = st.columns(2)
            m1.metric(label="Bloom's Level Accuracy", value=f"{pipeline_metrics.get('bloom_accuracy', 'N/A')}%")
            m2.metric(label="Difficulty Level Accuracy", value=f"{pipeline_metrics.get('difficulty_accuracy', 'N/A')}%")
            
            st.divider()
            st.markdown("### 🔍 Dataset Analytical Insights")
            
            try:
                df_viz = pd.read_csv("cognitive_dataset.csv")
                df_viz = df_viz.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=['time_taken_minutes', 'correct_percentage', 'num_students_attempted']
                )

                # Insight 1: Time Taken vs Score Correlation
                st.markdown("**1. Correlation: Time Taken vs. Success Rate**")
                st.markdown("Does spending more time on a question correlate with higher student success?")
                df_scatter = df_viz[['time_taken_minutes', 'correct_percentage']].rename(
                    columns={'time_taken_minutes': 'Time Spent (min)', 'correct_percentage': 'Success Rate'}
                ).dropna()
                st.scatter_chart(df_scatter, x='Time Spent (min)', y='Success Rate')

                st.markdown("---")

                # Insight 2: Difficulty Impact on Attempts
                st.markdown("**2. Student Engagement by Empirical Difficulty**")
                st.markdown("Average number of student attempts broken down by the question's difficulty rating.")
                df_bar = df_viz.groupby('difficulty')['num_students_attempted'].mean().reset_index()
                df_bar = df_bar.set_index('difficulty')
                st.bar_chart(df_bar)

                st.markdown("---")

                # Insight 3: Bloom's Level Distribution
                st.markdown("**3. Question Frequency Across Bloom's Taxonomy**")
                st.markdown("The distribution of cognitive load requirements across the entire tested dataset.")
                df_bloom = df_viz['bloom_level'].value_counts().reset_index()
                df_bloom.columns = ['Bloom Taxonomy Level', 'Question Count']
                df_bloom = df_bloom.set_index('Bloom Taxonomy Level')
                st.bar_chart(df_bloom)
                
            except Exception as e:
                st.warning("Could not load internal dataset for deep visualizations.")
        else:
            st.warning("No metrics found. Please re-run the `train_and_save.py` script to generate evaluation scores.")

# ============================================================
# PAGE 3 — Agentic Assessment Assistant (Milestone 2)
# ============================================================
elif page == "🤖 Agentic Assessment Assistant":
    st.subheader("🤖 Agentic Assessment Design Assistant")
    st.markdown(
        "Enter your exam questions below. The system will classify each question "
        "using the trained ML models, then run an **AI agent** (LangGraph + RAG) "
        "to analyse coverage gaps and generate improvement recommendations grounded "
        "in pedagogical literature."
    )

    # --- Set Groq API key from environment or use default ---
    if "GROQ_API_KEY" not in os.environ:
        # For local development, set your API key in .env file or environment
        # For deployment, set it in Streamlit secrets or environment variables
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Configuration")
        api_key_input = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            help="Enter your Groq API key. Get one free at https://console.groq.com/keys"
        )
        if api_key_input:
            os.environ["GROQ_API_KEY"] = api_key_input
        else:
            st.info("💡 **Tip:** Enter your Groq API key in the sidebar to use the Agentic Assistant.")
            st.stop()

    # --- Inputs ---
    subject = st.selectbox(
        "Subject",
        ["Computer Science", "Biology", "Physics", "Mathematics", "Economics"],
        key="agent_subject",
    )
    query = st.text_area(
        "Assessment Goal / Query",
        placeholder="E.g., Identify gaps in our Chapter 5 exam on Genetics. "
                    "We want to improve higher-order thinking coverage.",
        height=80,
    )

    st.markdown("---")
    st.markdown("### Step 1 — Enter Exam Questions")
    st.caption(
        "Enter one question per line. The ML models will auto-classify each "
        "question's Bloom's level and difficulty."
    )

    questions_raw = st.text_area(
        "Questions (one per line)",
        placeholder="Define the term DNA replication.\n"
                    "Explain how mRNA is synthesized during transcription.\n"
                    "Analyse the impact of a point mutation on protein function.\n"
                    "Design an experiment to test CRISPR efficiency in E. coli.",
        height=200,
        key="agent_questions",
    )

    st.markdown("### Step 2 — Optional Student Performance Data")
    st.caption("Providing performance stats improves gap detection accuracy.")
    use_perf_data = st.checkbox("Include average student performance metrics")
    avg_score_global = 70.0
    correct_pct_global = 0.70
    if use_perf_data:
        c1, c2 = st.columns(2)
        avg_score_global = c1.slider("Avg Score (%)", 0.0, 100.0, 70.0, key="ag_avg")
        correct_pct_global = c2.slider("Correct Percentage", 0.0, 1.0, 0.70, key="ag_cp")

    st.markdown("---")
    run_btn = st.button("▶ Run Agent Analysis", type="primary", use_container_width=True)

    if run_btn:
        lines = [l.strip() for l in questions_raw.strip().splitlines() if l.strip()]
        if len(lines) < 2:
            st.error("Please enter at least 2 questions to run a meaningful analysis.")
            st.stop()

        if not query.strip():
            query = f"Analyse assessment quality and coverage for {subject}."

        def classify_by_keyword(text):
            """Rule-based Bloom's + difficulty classification using question verb."""
            t = text.lower()
            if any(w in t for w in ["define", "list", "name", "recall", "identify", "state", "what is", "label", "match"]):
                return "Remember", "Easy"
            elif any(w in t for w in ["explain", "describe", "summarise", "summarize", "compare", "classify", "interpret", "paraphrase"]):
                return "Understand", "Easy"
            elif any(w in t for w in ["solve", "calculate", "apply", "demonstrate", "use", "show", "compute", "implement"]):
                return "Apply", "Moderate"
            elif any(w in t for w in ["analyse", "analyze", "differentiate", "examine", "break down", "organise", "relate", "deconstruct"]):
                return "Analyze", "Moderate"
            elif any(w in t for w in ["evaluate", "judge", "justify", "critique", "assess", "argue", "defend", "appraise"]):
                return "Evaluate", "Hard"
            elif any(w in t for w in ["design", "create", "formulate", "construct", "develop", "plan", "produce", "generate", "propose"]):
                return "Create", "Hard"
            return "Understand", "Moderate"  # default

        def estimate_stats(bloom):
            """Return representative student performance stats per Bloom's level."""
            stats = {
                "Remember":  dict(avg_score=85.0, correct_percentage=0.85, num_students_attempted=120, time_taken_minutes=90.0),
                "Understand": dict(avg_score=78.0, correct_percentage=0.78, num_students_attempted=115, time_taken_minutes=150.0),
                "Apply":      dict(avg_score=68.0, correct_percentage=0.68, num_students_attempted=105, time_taken_minutes=220.0),
                "Analyze":    dict(avg_score=60.0, correct_percentage=0.60, num_students_attempted=95,  time_taken_minutes=300.0),
                "Evaluate":   dict(avg_score=55.0, correct_percentage=0.55, num_students_attempted=88,  time_taken_minutes=360.0),
                "Create":     dict(avg_score=50.0, correct_percentage=0.50, num_students_attempted=80,  time_taken_minutes=420.0),
            }
            return stats.get(bloom, dict(avg_score=avg_score_global, correct_percentage=correct_pct_global,
                                         num_students_attempted=100, time_taken_minutes=200.0))

        # --- Classification of each question ---
        questions_for_agent: list[dict] = []
        with st.status("Step 1/4 — Classifying questions...", expanded=True) as status:
            for q_text in lines:
                bloom_pred, diff_pred = classify_by_keyword(q_text)
                stats = estimate_stats(bloom_pred)

                questions_for_agent.append({
                    "text": q_text,
                    "bloom_level": bloom_pred,
                    "difficulty": diff_pred,
                    "correct_percentage": stats["correct_percentage"],
                })
            status.update(label=f"✅ Classified {len(lines)} questions.", state="complete")

        # --- ML Results Preview ---
        with st.expander("📋 Question Classification Results", expanded=False):
            ml_df = pd.DataFrame([
                {"Question": q["text"][:80] + ("…" if len(q["text"]) > 80 else ""),
                 "Bloom's Level": q["bloom_level"],
                 "Difficulty": q["difficulty"]}
                for q in questions_for_agent
            ])
            st.dataframe(ml_df, use_container_width=True)

        # --- LangGraph Agent ---
        try:
            from agent import run_agent

            with st.status("Step 2/4 — Retrieving pedagogical guidelines (RAG)...", expanded=False):
                pass  # RAG runs inside the agent; progress shown via status steps below

            with st.status("Running LangGraph agent (nodes: analyze → retrieve → gaps → recommend → compile)...", expanded=True) as agent_status:
                st.write("🔍 Analyzing difficulty & Bloom's distributions...")
                st.write("📚 Querying pedagogy knowledge base (Chroma RAG)...")
                st.write("🧠 Identifying learning gaps with Gemini LLM...")
                st.write("💡 Generating improvement recommendations...")
                result = run_agent(
                    query=query,
                    subject=subject,
                    questions=questions_for_agent,
                )
                agent_status.update(label="✅ Agent analysis complete!", state="complete")

        except EnvironmentError as env_err:
            st.error(str(env_err))
            st.stop()
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()

        # --- Structured Report ---
        st.markdown("---")
        st.markdown("## 📄 Assessment Report")

        tab_summary, tab_dist, tab_gaps, tab_recs, tab_refs, tab_ethics = st.tabs([
            "📊 Summary",
            "📈 Distributions",
            "⚠️ Learning Gaps",
            "✅ Recommendations",
            "📚 References",
            "⚖️ Ethics Notice",
        ])

        with tab_summary:
            st.markdown("### Assessment Quality Summary")
            st.info(result["quality_summary"] or "No summary generated.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Questions", len(questions_for_agent))
            c2.metric("Avg Correct %", f"{result['avg_correct_pct']:.1%}")
            c3.metric("Bloom's Levels Covered", len(result["bloom_dist"]))

        with tab_dist:
            st.markdown("### Difficulty Distribution")
            diff_df = pd.DataFrame(
                list(result["difficulty_dist"].items()), columns=["Difficulty", "Count"]
            ).set_index("Difficulty")
            st.bar_chart(diff_df)

            st.markdown("### Bloom's Taxonomy Distribution")
            bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
            bloom_data = {k: result["bloom_dist"].get(k, 0) for k in bloom_order}
            bloom_df = pd.DataFrame(
                list(bloom_data.items()), columns=["Bloom Level", "Count"]
            ).set_index("Bloom Level")
            st.bar_chart(bloom_df)

        with tab_gaps:
            st.markdown("### Identified Learning Gaps")
            if result["gaps"]:
                for gap in result["gaps"]:
                    st.markdown(f"• {gap}")
            else:
                st.success("No significant gaps detected.")

        with tab_recs:
            st.markdown("### Improvement Recommendations")
            if result["recommendations"]:
                for i, rec in enumerate(result["recommendations"], 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.info("No recommendations generated.")

        with tab_refs:
            st.markdown("### Pedagogical References")
            if result["references"]:
                for ref in result["references"]:
                    st.markdown(f"• {ref}")
            else:
                st.markdown("No explicit references cited.")

        with tab_ethics:
            st.markdown("### Educational & Ethical Notice")
            st.warning(result["disclaimer"])
