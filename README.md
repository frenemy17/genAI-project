# 🎓 Intelligent Exam Question Analysis & Agentic Assessment Design

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)

> An AI-powered educational analytics system that analyzes exam questions and provides autonomous assessment design recommendations using Machine Learning and Agentic AI.

## 🌟 Features

### 📝 Milestone 1: Classical ML Analytics
- **Bloom's Taxonomy Classification** - Predict cognitive load levels (Remember, Understand, Apply, Analyze, Evaluate, Create)
- **Difficulty Prediction** - Classify questions as Easy, Moderate, or Hard
- **High Accuracy** - 91.19% accuracy for Bloom's classification, 73.56% for difficulty
- **Advanced ML Pipeline** - Voting Ensembles (Logistic Regression + Random Forest)
- **NLP Features** - TF-IDF vectorization with 1-5 grams, student performance metrics

### 🤖 Milestone 2: Agentic AI Assistant
- **LangGraph Workflow** - 5-node autonomous agent pipeline
- **RAG System** - Retrieval-Augmented Generation with ChromaDB
- **Pedagogical Knowledge Base** - 10+ educational frameworks and best practices
- **Gap Analysis** - Identifies missing Bloom's levels, difficulty imbalances, coverage gaps
- **Actionable Recommendations** - Evidence-based improvement suggestions
- **Groq AI Integration** - Powered by Llama 3.3 70B model

## 🚀 Live Demo

**Local URL:** `http://localhost:8501`

## 📸 Screenshots

### Page 1: Question Classification
Predict Bloom's taxonomy level and difficulty for individual questions with student performance metrics.

### Page 2: Model Analytics
View model accuracy, dataset insights, and performance visualizations.

### Page 3: Agentic Assessment Assistant
AI-powered exam analysis with gap detection and pedagogical recommendations.

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **ML Models** | Scikit-learn (Logistic Regression, Random Forest, Voting Ensembles) |
| **NLP** | TF-IDF Vectorization, N-Grams (1-5) |
| **Agent Framework** | LangGraph |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **LLM** | Groq AI (Llama 3.3 70B) |
| **Language** | Python 3.14 |

## 📦 Installation

### Prerequisites
- Python 3.14+
- pip or conda
- Groq API Key (free from https://console.groq.com/keys)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/frenemy17/genAI-project.git
cd genAI-project
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API Key**

**Option A: Environment Variable (Recommended)**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Option B: .env File**
```bash
cp .env.example .env
# Edit .env and add your API key
```

**Option C: Streamlit Secrets (For Deployment)**
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and add your API key
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Access the app**
Open your browser and navigate to `http://localhost:8501`

> **Note:** If you don't set the API key in environment, you can enter it in the sidebar when using the Agentic Assistant.

## 📊 Model Training

The pre-trained models are included in the `models/` directory. To retrain:

```bash
python train_and_save.py
```

This will:
- Load `cognitive_dataset.csv`
- Engineer features (TF-IDF, student metrics)
- Train Voting Classifier ensembles
- Save models to `models/` directory
- Generate `metrics.json` with accuracy scores

## 🎯 Usage

### 1. Single Question Analysis
- Navigate to **"📝 Predict Cognitive Load & Difficulty"**
- Enter question text and student performance data
- Get instant Bloom's level and difficulty predictions

### 2. Model Performance Review
- Navigate to **"📊 Model Analytics & Accuracy"**
- View model accuracy metrics
- Explore dataset visualizations and insights

### 3. Agentic Assessment Analysis
- Navigate to **"🤖 Agentic Assessment Assistant"**
- Select subject and enter assessment goal
- Paste exam questions (one per line)
- Click **"▶ Run Agent Analysis"**
- Review comprehensive report with:
  - Quality summary
  - Bloom's and difficulty distributions
  - Identified learning gaps
  - Actionable recommendations
  - Pedagogical references

## 🧪 Test Examples

### Test Case 1: Unbalanced Exam (Too Easy)
```
Define what an algorithm is.
What does CPU stand for?
List three programming languages.
Name the founder of Python programming language.
What is a variable in programming?
```
**Expected:** Identifies lack of higher-order thinking, too many "Remember" level questions.

### Test Case 2: Well-Balanced Exam
```
Define photosynthesis.
Explain how enzymes work as catalysts.
Apply the concept of natural selection to explain antibiotic resistance.
Analyze the impact of deforestation on the carbon cycle.
Evaluate the effectiveness of renewable energy sources.
Design an experiment to test plant growth under different light conditions.
```
**Expected:** Shows good distribution, minimal gaps, positive feedback.

## 🏗️ Project Structure

```
genAI-project/
├── app.py                              # Main Streamlit application
├── agent.py                            # LangGraph agent implementation
├── pedagogy_kb.py                      # RAG knowledge base
├── train_and_save.py                   # Model training script
├── cognitive_dataset.csv               # Training dataset
├── requirements.txt                    # Python dependencies
├── models/
│   ├── bloom_pipeline.pkl              # Bloom's taxonomy classifier
│   ├── difficulty_pipeline.pkl         # Difficulty classifier
│   └── metrics.json                    # Model performance metrics
└── README.md                           # Project documentation
```

## 🎓 Educational Frameworks

The system is grounded in established pedagogical research:

- **Bloom's Taxonomy** - Cognitive learning objectives framework
- **30-40-30 Difficulty Rule** - Balanced assessment design
- **Cognitive Load Theory** - Sweller (1988)
- **Assessment Quality Standards** - Item analysis and discrimination
- **Curriculum-Assessment Alignment** - Content and construct validity

## 📈 Model Performance

| Model | Accuracy | Method |
|-------|----------|--------|
| **Bloom's Taxonomy** | 91.19% | Voting Ensemble (LR + RF) |
| **Difficulty Level** | 73.56% | Voting Ensemble (LR + RF) |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Team Members** - GenAI Project Team

## 🙏 Acknowledgments

- Bloom's Taxonomy Framework
- Educational Measurement & Assessment Principles
- LangChain & LangGraph communities
- Groq AI for free LLM access

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ Star this repository if you find it helpful!**
