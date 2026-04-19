"""
Pedagogy Knowledge Base — RAG source for Milestone 2.
Builds a Chroma vector store from hardcoded pedagogical documents.
"""
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PEDAGOGY_DOCS = [
    Document(
        page_content="""Bloom's Taxonomy Framework for Assessment Design:
Bloom's Taxonomy classifies learning objectives into six hierarchical levels:
1. Remember: Recall facts, definitions. Verbs: define, list, identify, name.
2. Understand: Interpret and explain. Verbs: explain, summarize, classify, compare.
3. Apply: Use knowledge in new situations. Verbs: solve, demonstrate, implement.
4. Analyze: Break down into components. Verbs: differentiate, organize, deconstruct.
5. Evaluate: Make judgments based on criteria. Verbs: judge, critique, justify, defend.
6. Create: Produce original work. Verbs: design, construct, plan, formulate.

Best Practice: A balanced summative assessment should include questions from all levels.
Higher-order thinking (Analyze, Evaluate, Create) should comprise at least 40% of the exam.
Assessments heavy in Remember/Understand only test surface knowledge and fail to measure
deep learning or transfer of knowledge to novel situations.""",
        metadata={"source": "Bloom's Taxonomy Educational Framework", "category": "cognitive_framework"},
    ),
    Document(
        page_content="""Difficulty Distribution Best Practices — The 30-40-30 Rule:
A well-balanced exam follows the 30-40-30 difficulty guideline:
  - Easy (30%): Build confidence, test foundational/recall knowledge.
  - Medium (40%): Core learning objectives, majority of curriculum coverage.
  - Hard (30%): Challenge advanced learners, differentiate top performers.

Warning signs of poor difficulty distribution:
  - >60% easy: Assessment lacks rigour; all students score similarly — poor discrimination.
  - >60% hard: Demoralising; does not reflect instructional coverage.
  - Bimodal (all easy + all hard, no medium): Structural design failure.

Discrimination Index (Point Biserial Correlation):
  - < 0.20: Poor — item should be revised or removed.
  - 0.20–0.29: Fair — consider revision.
  - 0.30–0.39: Good.
  - ≥ 0.40: Excellent discriminator.

Facility Index (p-value, proportion correct):
  - > 0.85: Too easy — consider removing or raising complexity.
  - < 0.15: Too hard — consider scaffolding or removing.
  - Ideal range: 0.30–0.70.""",
        metadata={"source": "Educational Measurement & Assessment Principles", "category": "difficulty_design"},
    ),
    Document(
        page_content="""Curriculum-Assessment Alignment (CAA) Theory:
Assessment questions must align with declared learning objectives and teaching coverage.

Content Validity: An assessment has content validity when it adequately samples
the domain of knowledge being tested. Ensure:
  - No single topic exceeds 30% of all questions (over-representation).
  - No major topic is completely absent (under-coverage gap).
  - Questions span multiple subtopics proportionally.

Construct Validity: Questions must measure what they claim to measure.
Avoid construct-irrelevant variance (e.g., reading difficulty unrelated to subject).

Instructional Alignment: Cognitive level of questions must match the instructional
approach. If concepts were taught at the Apply level, testing only at Remember level
is a misalignment that undervalues student preparation.""",
        metadata={"source": "Standards for Educational and Psychological Testing", "category": "curriculum_alignment"},
    ),
    Document(
        page_content="""Learning Gap Identification Framework:
A learning gap exists when student performance reveals systematic failure to achieve
a learning objective. Key quantitative indicators:
  1. Low Correct Percentage (<40%): Students systematically failing a concept.
  2. High Time per Attempt (>2× average): Struggle with complexity or clarity.
  3. Cluster of failures at specific Bloom's levels: Indicates pedagogical gap.
  4. Topic-specific failure patterns: Reveals curriculum coverage gaps.

Gap Analysis by Bloom's Level:
  - Foundation Gaps (Remember/Understand fail): Revisit basic instruction, worked examples.
  - Application Gaps (Apply fails but Understand passes): Add practice problems, real-world scenarios.
  - Higher-Order Gaps (Analyze/Evaluate fail): Scaffolded critical thinking, case studies, debates.

Remediation Strategy: Identify the lowest Bloom's level where gaps appear,
address it fully before moving to higher levels. Bottom-up remediation is more
effective than parallel remediation across all levels.""",
        metadata={"source": "Diagnostic Assessment & Learning Gap Framework", "category": "gap_analysis"},
    ),
    Document(
        page_content="""Question Quality Criteria for Effective Assessment Items:
High-quality exam questions meet these standards:
  1. Clarity: Unambiguous language; single correct interpretation possible.
  2. Relevance: Directly tests a stated learning objective.
  3. Cognitive Appropriateness: Cognitive demand matches intended Bloom's level.
  4. Plausible Distractors (MCQ): Wrong options are educationally meaningful.
  5. Discrimination: Differentiates between high and low performers.

Item Revision Triggers:
  - Ambiguous phrasing flagged by multiple students.
  - Discrimination index < 0.20 in post-exam analysis.
  - Facility index outside 0.15–0.85 range.
  - Question tests two separate concepts simultaneously (double-barreled).

Regular Item Bank Maintenance:
  - Review every item after each administration using item analysis statistics.
  - Retire items with persistent poor discrimination.
  - Update items that reference outdated content or context.""",
        metadata={"source": "Item Analysis & Assessment Quality Standards", "category": "question_quality"},
    ),
    Document(
        page_content="""Formative vs Summative Assessment Design Principles:

Formative Assessment (Assessment FOR Learning):
  - Purpose: Monitor learning progress, provide ongoing feedback.
  - Frequency: Continuous, low-stakes.
  - Design: Wide coverage, emphasis on lower-to-mid Bloom's levels, diagnostic focus.

Summative Assessment (Assessment OF Learning):
  - Purpose: Evaluate competency at end of unit or course.
  - Frequency: Periodic, high-stakes.
  - Design: Comprehensive, includes higher-order thinking, marks-weighted.

Summative Exam Design Checklist:
  ✓ Covers all major learning objectives.
  ✓ Questions at Remember, Understand, Apply AND at least one higher level.
  ✓ Difficulty follows 30-40-30 guideline.
  ✓ Questions are unambiguous.
  ✓ Time allocation appropriate (typically 1–2 min per mark).
  ✓ Marking scheme is objective and reproducible.""",
        metadata={"source": "Assessment for Learning: Beyond the Black Box (Black & Wiliam, 1998)", "category": "assessment_types"},
    ),
    Document(
        page_content="""Cognitive Load Theory in Assessment Design (Sweller, 1988):
Three types of cognitive load relevant to assessment:
  1. Intrinsic Load: Inherent complexity of subject matter.
     → Match question complexity to student expertise level.
  2. Extraneous Load: Unnecessary demand from poor question design.
     → Use clear, concise wording; avoid irrelevant information in stems.
  3. Germane Load: Cognitive effort that promotes schema formation.
     → Include questions requiring students to connect concepts.

Practical Guidelines:
  - Progress from simple to complex within each section.
  - Avoid embedding multiple distinct concepts in a single question.
  - Long question stems with multiple clauses increase extraneous load.
  - Working memory constraint: students hold 7±2 information chunks.
  - Provide formula sheets for STEM subjects to reduce recall burden.""",
        metadata={"source": "Cognitive Load Theory — Sweller (1988, 2011)", "category": "cognitive_science"},
    ),
    Document(
        page_content="""Assessment Improvement Priority Framework:
Four dimensions for improving an existing assessment:

1. COVERAGE IMPROVEMENTS:
   - Add questions for uncovered learning objectives.
   - Ensure all Bloom's levels are represented.
   - Balance representation across topics.

2. DIFFICULTY IMPROVEMENTS:
   - Adjust p-values for flagged items (too easy / too hard).
   - Add scaffold questions before high-difficulty items.
   - Revise or remove items with negative discrimination.

3. COGNITIVE QUALITY IMPROVEMENTS:
   - Replace recall-only questions with applied thinking questions.
   - Add scenario-based questions for higher Bloom's levels.
   - Introduce case studies or data interpretation tasks.

4. ALIGNMENT IMPROVEMENTS:
   - Map each question to a specific learning objective.
   - Align time allocation with question complexity.
   - Balance marks distribution across topics proportionally.

Priority Matrix (act in this order):
  High Impact + Low Effort → Fix immediately.
  High Impact + High Effort → Plan and schedule.
  Low Impact + Low Effort → Fix if time allows.
  Low Impact + High Effort → Defer.""",
        metadata={"source": "Assessment Quality Assurance Framework", "category": "improvement_framework"},
    ),
    Document(
        page_content="""Ethical & Responsible AI Practices in Educational Assessment:
Ethical Principles for Assessment Design:
  1. Fairness: Questions must not disadvantage students based on cultural background
     or language proficiency unrelated to subject knowledge.
  2. Validity: Measure only what you intend to measure.
  3. Reliability: Consistent scoring regardless of who evaluates.
  4. Transparency: Students should know assessment criteria in advance.
  5. Accessibility: Questions must be accessible to students with disabilities.

Responsible AI in Education:
  - AI recommendations are advisory, not prescriptive.
  - Human educators retain final judgment on all assessment decisions.
  - AI analysis should supplement, not replace, expert pedagogical judgment.
  - Student data used for AI analysis must be anonymised and protected.
  - Communicate AI confidence levels to avoid uncritical over-reliance.
  - Bias check: Review AI-generated recommendations for potential cultural or
    demographic bias before implementation.

Data Privacy: All student performance data must comply with applicable privacy
regulations (e.g., FERPA, GDPR). Individual-level data should never be exposed
in aggregate reports.""",
        metadata={"source": "Ethical Framework for Educational Technology (ISTE, UNESCO)", "category": "ethics"},
    ),
    Document(
        page_content="""Subject-Specific Assessment Guidelines:

STEM Subjects (Mathematics, Physics, Computer Science):
  - Include both procedural (Apply) and conceptual (Understand/Analyze) questions.
  - Provide formula sheets to avoid over-testing memorisation.
  - Emphasise data interpretation and multi-step problem solving.
  - Partial credit for multi-step problems prevents penalising early errors.

Life Sciences (Biology):
  - Balance factual recall with application to novel scenarios.
  - Include diagram interpretation and experimental design questions.
  - Higher-order questions: hypothesis testing, data analysis, ethics of science.

Social Sciences (Economics):
  - Case study analysis alongside theoretical questions.
  - Evaluate/Create level via policy analysis essays.
  - Data analysis (graphs, tables) as a significant component.

General Best Practices (all subjects):
  - At least one real-world application question per major topic.
  - Update question banks annually to reflect curriculum changes.
  - Pilot new questions as unscored items before counting them.""",
        metadata={"source": "Subject-Specific Assessment Design Guidelines", "category": "subject_guidelines"},
    ),
]

_vectorstore = None


def get_vectorstore() -> Chroma:
    """Return a cached in-memory Chroma vector store loaded with pedagogy docs."""
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma.from_documents(PEDAGOGY_DOCS, embeddings)
    return _vectorstore


def retrieve_relevant_docs(query: str, k: int = 4) -> list[str]:
    """Retrieve top-k relevant pedagogy passages for a query."""
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    return [d.page_content for d in docs]
