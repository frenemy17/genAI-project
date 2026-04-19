"""
Agentic Assessment Design Assistant — Milestone 2
LangGraph workflow with RAG for pedagogical guidelines.

Nodes (in order):
  1. analyze_patterns    — compute difficulty & Bloom's distributions from question data
  2. retrieve_guidelines — RAG retrieval from the pedagogy Chroma knowledge base
  3. identify_gaps       — LLM identifies learning gaps from distributions + context
  4. generate_recs       — LLM generates structured improvement recommendations
  5. compile_report      — assemble final structured report with disclaimer
"""
import os
from typing import TypedDict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from pedagogy_kb import retrieve_relevant_docs


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AssessmentState(TypedDict):
    query: str
    subject: str
    questions: List[dict]          # [{text, bloom_level, difficulty}, ...]
    difficulty_dist: dict          # {Easy: int, Moderate: int, Hard: int}
    bloom_dist: dict               # {Remember: int, ...}
    avg_correct_pct: float         # average correct_percentage across questions
    retrieved_contexts: List[str]  # top-k RAG passages
    quality_summary: str           # LLM-generated quality assessment
    gaps: List[str]                # identified learning gaps
    recommendations: List[str]     # improvement recommendations
    references: List[str]          # pedagogical references cited
    disclaimer: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# LLM helper — lazy init so the API key is read at runtime
# ---------------------------------------------------------------------------

def _get_llm() -> ChatGroq:
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Get a free key at https://console.groq.com/keys "
            "and set it in the sidebar before running the agent."
        )
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Node 1: Analyze difficulty & Bloom's distributions
# ---------------------------------------------------------------------------

def analyze_patterns(state: AssessmentState) -> AssessmentState:
    questions = state["questions"]

    diff_dist: dict[str, int] = {}
    bloom_dist: dict[str, int] = {}
    correct_pcts: list[float] = []

    for q in questions:
        d = q.get("difficulty", "Unknown")
        b = q.get("bloom_level", "Unknown")
        diff_dist[d] = diff_dist.get(d, 0) + 1
        bloom_dist[b] = bloom_dist.get(b, 0) + 1
        if "correct_percentage" in q:
            correct_pcts.append(float(q["correct_percentage"]))

    avg_correct = sum(correct_pcts) / len(correct_pcts) if correct_pcts else 0.0

    return {
        **state,
        "difficulty_dist": diff_dist,
        "bloom_dist": bloom_dist,
        "avg_correct_pct": round(avg_correct, 4),
    }


# ---------------------------------------------------------------------------
# Node 2: Retrieve pedagogical guidelines (RAG)
# ---------------------------------------------------------------------------

def retrieve_guidelines(state: AssessmentState) -> AssessmentState:
    # Build a targeted query from distribution stats + subject
    total = len(state["questions"])
    query = (
        f"Assessment design for {state['subject']}. "
        f"Difficulty distribution: {state['difficulty_dist']}. "
        f"Bloom's distribution: {state['bloom_dist']}. "
        f"Total questions: {total}. "
        f"Query: {state['query']}"
    )
    contexts = retrieve_relevant_docs(query, k=5)
    return {**state, "retrieved_contexts": contexts}


# ---------------------------------------------------------------------------
# Node 3: Identify learning gaps
# ---------------------------------------------------------------------------

_GAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert educational psychologist and assessment specialist. "
        "Analyse the provided assessment statistics and pedagogical guidelines, "
        "then identify concrete learning gaps. "
        "Be specific, evidence-based, and concise. "
        "Do NOT hallucinate data. Only reference what is given.",
    ),
    (
        "human",
        """Subject: {subject}
Total Questions: {total}
Difficulty Distribution: {difficulty_dist}
Bloom's Taxonomy Distribution: {bloom_dist}
Average Correct Percentage: {avg_correct_pct:.1%}

Pedagogical Context (retrieved):
{context}

Task: List the top 4–6 learning gaps present in this assessment.
Format each gap as a single clear sentence starting with a bullet point (•).
Focus on: missing Bloom's levels, difficulty imbalances, topic coverage gaps,
and student performance patterns.""",
    ),
])


def identify_gaps(state: AssessmentState) -> AssessmentState:
    llm = _get_llm()
    context = "\n\n---\n\n".join(state["retrieved_contexts"][:3])
    chain = _GAP_PROMPT | llm

    response = chain.invoke({
        "subject": state["subject"],
        "total": len(state["questions"]),
        "difficulty_dist": state["difficulty_dist"],
        "bloom_dist": state["bloom_dist"],
        "avg_correct_pct": state["avg_correct_pct"],
        "context": context,
    })

    raw = response.content.strip()
    gaps = [
        line.lstrip("•-– ").strip()
        for line in raw.splitlines()
        if line.strip() and line.strip()[0] in ("•", "-", "–", "*")
    ]
    if not gaps:
        gaps = [s.strip() for s in raw.split("\n") if s.strip()]

    return {**state, "gaps": gaps}


# ---------------------------------------------------------------------------
# Node 4: Generate recommendations + quality summary
# ---------------------------------------------------------------------------

_REC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior curriculum designer providing actionable, evidence-based "
        "recommendations to improve an exam. Ground every recommendation in "
        "the pedagogical guidelines provided. Be specific and practical.",
    ),
    (
        "human",
        """Subject: {subject}
Difficulty Distribution: {difficulty_dist}
Bloom's Taxonomy Distribution: {bloom_dist}
Average Correct Percentage: {avg_correct_pct:.1%}

Identified Learning Gaps:
{gaps}

Pedagogical Guidelines (retrieved):
{context}

Task: Provide:
1. QUALITY SUMMARY (2–3 sentences): Overall assessment quality and its key strength/weakness.
2. RECOMMENDATIONS (6–8 bullet points): Specific, prioritised improvements.
   Start each with an action verb. Reference Bloom's levels or difficulty as appropriate.
3. REFERENCES (3–5 items): Name the specific pedagogical frameworks/sources you applied.

Use this exact format:
QUALITY SUMMARY:
<text>

RECOMMENDATIONS:
• <rec 1>
• <rec 2>
...

REFERENCES:
• <ref 1>
• <ref 2>
...""",
    ),
])


def generate_recs(state: AssessmentState) -> AssessmentState:
    llm = _get_llm()
    context = "\n\n---\n\n".join(state["retrieved_contexts"])
    chain = _REC_PROMPT | llm

    response = chain.invoke({
        "subject": state["subject"],
        "difficulty_dist": state["difficulty_dist"],
        "bloom_dist": state["bloom_dist"],
        "avg_correct_pct": state["avg_correct_pct"],
        "gaps": "\n".join(f"• {g}" for g in state["gaps"]),
        "context": context,
    })

    raw = response.content.strip()

    # Parse sections
    quality_summary = ""
    recommendations: list[str] = []
    references: list[str] = []

    current_section = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().startswith("QUALITY SUMMARY"):
            current_section = "summary"
            continue
        if stripped.upper().startswith("RECOMMENDATIONS"):
            current_section = "recs"
            continue
        if stripped.upper().startswith("REFERENCES"):
            current_section = "refs"
            continue

        if current_section == "summary":
            quality_summary += " " + stripped
        elif current_section == "recs" and stripped[0] in ("•", "-", "*", "–"):
            recommendations.append(stripped.lstrip("•-*– ").strip())
        elif current_section == "refs" and stripped[0] in ("•", "-", "*", "–"):
            references.append(stripped.lstrip("•-*– ").strip())

    return {
        **state,
        "quality_summary": quality_summary.strip(),
        "recommendations": recommendations,
        "references": references,
    }


# ---------------------------------------------------------------------------
# Node 5: Compile final report (adds disclaimer)
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "EDUCATIONAL & ETHICAL NOTICE: This report is generated by an AI system "
    "using statistical analysis and retrieved pedagogical literature. "
    "All recommendations are advisory only and must be reviewed by a qualified "
    "educator before implementation. Student performance data used in this analysis "
    "should be anonymised and handled in compliance with applicable privacy "
    "regulations (FERPA, GDPR, or applicable local law). "
    "The AI does not replace professional pedagogical judgment."
)


def compile_report(state: AssessmentState) -> AssessmentState:
    return {**state, "disclaimer": DISCLAIMER}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_agent():
    graph = StateGraph(AssessmentState)

    graph.add_node("analyze_patterns", analyze_patterns)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("identify_gaps", identify_gaps)
    graph.add_node("generate_recs", generate_recs)
    graph.add_node("compile_report", compile_report)

    graph.set_entry_point("analyze_patterns")
    graph.add_edge("analyze_patterns", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "identify_gaps")
    graph.add_edge("identify_gaps", "generate_recs")
    graph.add_edge("generate_recs", "compile_report")
    graph.add_edge("compile_report", END)

    return graph.compile()


def run_agent(
    query: str,
    subject: str,
    questions: list[dict],
) -> AssessmentState:
    """
    Run the agentic assessment pipeline.

    Parameters
    ----------
    query     : Free-text question or goal from the educator.
    subject   : Subject name (e.g. 'Biology').
    questions : List of dicts with keys: text, bloom_level, difficulty,
                and optionally correct_percentage.

    Returns
    -------
    Final AssessmentState after all nodes have executed.
    """
    agent = build_agent()
    initial_state: AssessmentState = {
        "query": query,
        "subject": subject,
        "questions": questions,
        "difficulty_dist": {},
        "bloom_dist": {},
        "avg_correct_pct": 0.0,
        "retrieved_contexts": [],
        "quality_summary": "",
        "gaps": [],
        "recommendations": [],
        "references": [],
        "disclaimer": "",
        "error": None,
    }
    return agent.invoke(initial_state)
