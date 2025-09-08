import os, json, pathlib, importlib.util, sys, hashlib, pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from typing import cast, List, Literal
from test_utils.prompt import LLM_AS_A_JUDGE_PROMPT, USER_TASK, EXPERT_CODE, BASE_CODE
from test_utils.format_code import folder_to_prompt_string
from test_utils.git_branch import get_git_branch

CANDIDATE_NAME = get_git_branch()
LLM_AS_JUDGE_MODEL = "claude-sonnet-4-20250514"
CODE_FOLDER = [pathlib.Path("../src/memory_agent")]

HUMAN_NOTES = """
1. Local/Deployed testing:
llm = init_chat_model() should be llm = init_chat_model("anthropic:claude-3-5-sonnet-latest") otherwise we get a positional argument error.
2. Local/Deployed testing:
    - context=utils.split_model_and_provider(model), should be removed.
"""

class LlmAsJudgeEvidence(BaseModel):
    issue: str
    severity: Literal["minor", "major", "critical"]

class BasicRequirements(BaseModel):
    presence_of_interrupt: bool # 2 points
    user_id_and_category_wise_storage: bool # 2 points
    correct_categories: bool # 1 points
    category_retrieval: bool # 1 points

class GoodPractices(BaseModel):
    functional_interrupt: bool # 2 points
    llm_based_categorization: bool # 2 points
    no_test_files: bool # 1 points

class LlmAsJudgeOutput(BaseModel):
    basic_requirements: BasicRequirements
    good_practices: GoodPractices
    code_quality_check: bool
    code_quality_evidence: List[LlmAsJudgeEvidence]
    code_correctness_check: bool
    code_correctness_evidence: List[LlmAsJudgeEvidence]

def _write_score(score):
    out = pathlib.Path("results"); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"code_quality_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)

def _add(score, awarded_pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": awarded_pts, "passed": ok, "msg": msg})
    score["points"] += awarded_pts

def _load_judge():
    """
    Returns a (invoke, model_name) tuple.
    """
    llm = ChatAnthropic(model=LLM_AS_JUDGE_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(LlmAsJudgeOutput)

    return (lambda msgs: structured_llm.invoke(msgs)), f"anthropic:{LLM_AS_JUDGE_MODEL}"

def _calculate_score(evidence_list: List[LlmAsJudgeEvidence], max_points: int) -> float:
    """Calculates a score based on a list of evidence items and their severity."""
    points_deducted = 0
    for evidence in evidence_list:
        if evidence.severity == "critical":
            points_deducted += 2
        elif evidence.severity == "major":
            points_deducted += 1
        elif evidence.severity == "minor":
            points_deducted += 0.5
        else:
            print(f"Warning: Unknown severity level '{evidence.severity}'")
            points_deducted += 1
    
    return max(0, max_points - points_deducted)

def test_best_practices_llm_judge():
    score = {"candidate": CANDIDATE_NAME, "bucket": "code_quality", "points": 0, "max_points": 23, "details": []}
    user_code = folder_to_prompt_string(CODE_FOLDER)

    with open('txt_dump/user_code_llm_as_judge.txt', 'w') as f:
        f.write(user_code)

    # Prompt the judge with task-specific guidelines
    system = LLM_AS_A_JUDGE_PROMPT.format(user_task=USER_TASK, expert_code=EXPERT_CODE, user_code=user_code, human_notes=HUMAN_NOTES, base_code=BASE_CODE)
    user = {
        "role": "user",
        "content": "Return the JSON object evaluating the codebase."
    }

    try:
        invoke, model_name = _load_judge()
        resp = invoke([SystemMessage(content=system), HumanMessage(content=user["content"])])
        judge = cast(LlmAsJudgeOutput, resp)
    except Exception as e:
        _add(score, 0, "judge_error", False, f"Judge error: {type(e).__name__}: {e}")
        _write_score(score)
        pytest.fail(f"LLM judge failed: {e}")
        
    # Code Quality check
    code_quality_points = _calculate_score(judge.code_quality_evidence, 4)
    _add(score, code_quality_points, "code_quality_check", judge.code_quality_check, str(judge.code_quality_evidence))

    # Code Correctness check (for general bugs)
    code_correctness_points = _calculate_score(judge.code_correctness_evidence, 8)
    _add(score, code_correctness_points, "code_correctness_check", judge.code_correctness_check, str(judge.code_correctness_evidence))

    # Basic Requirements check
    _add(score, 2 if judge.basic_requirements.presence_of_interrupt else 0, "presence_of_interrupt", judge.basic_requirements.presence_of_interrupt, "Presence of interrupt is present" if judge.basic_requirements.presence_of_interrupt else "Presence of interrupt is not present")
    _add(score, 2 if judge.basic_requirements.user_id_and_category_wise_storage else 0, "user_id_and_category_wise_storage", judge.basic_requirements.user_id_and_category_wise_storage, "User id and category wise storage is present" if judge.basic_requirements.user_id_and_category_wise_storage else "User id and category wise storage is not present")
    _add(score, 1 if judge.basic_requirements.correct_categories else 0, "correct_categories", judge.basic_requirements.correct_categories, "Correct categories is present" if judge.basic_requirements.correct_categories else "Correct categories is not present")
    _add(score, 1 if judge.basic_requirements.category_retrieval else 0, "category_retrieval", judge.basic_requirements.category_retrieval, "Category retrieval is present" if judge.basic_requirements.category_retrieval else "Category retrieval is not present")

    # Good Practices check
    _add(score, 2 if judge.good_practices.functional_interrupt else 0, "functional_interrupt", judge.good_practices.functional_interrupt, "Functional interrupt is present" if judge.good_practices.functional_interrupt else "Functional interrupt is not present")
    _add(score, 2 if judge.good_practices.llm_based_categorization else 0, "llm_based_categorization", judge.good_practices.llm_based_categorization, "LLM based categorization is present" if judge.good_practices.llm_based_categorization else "LLM based categorization is not present")
    _add(score, 1 if judge.good_practices.no_test_files else 0, "no_test_files", judge.good_practices.no_test_files, "No test files is present" if judge.good_practices.no_test_files else "No test files is not present")

    _write_score(score)