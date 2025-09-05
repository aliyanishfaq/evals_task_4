import importlib.util
import sys
import os
import json
import hashlib
import pathlib
import pytest
from typing import List
import langsmith as ls
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from test_utils.git_branch import get_git_branch


DEFAULT_AGENT_FILENAME = os.getenv("DEFAULT_AGENT_FILENAME", "main.py")
DEFAULT_AGENT_PATH = pathlib.Path.cwd() / f"../{DEFAULT_AGENT_FILENAME}"
CANDIDATE_NAME = get_git_branch()

def _load_module(agent_py_path: pathlib.Path):
    """Import agent.py as a temporary module; success == compiles/imports."""
    module_name = "candidate_agent_" + hashlib.md5(str(agent_py_path).encode()).hexdigest()[:8]
    spec = importlib.util.spec_from_file_location(module_name, str(agent_py_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, f"agent.py must compile/import: {e}"


def _get_builder(mod):
    """Return global `builder` from the module or raise a clear error."""
    if not hasattr(mod, "builder") and not hasattr(mod, "agent"):
        raise AssertionError("agent.py must export a global variable `builder` or `agent`")
    return mod.builder or mod.agent


def _write_score(score):
    out_dir = pathlib.Path("results")
    out_dir.mkdir(exist_ok=True, parents=True)
    with open(out_dir / f"memory_{score['candidate']}.json", "w") as f:
        json.dump(score, f, indent=2)


def _add(score, pts, key, ok, msg=""):
    score["details"].append({"key": key, "points": (pts if ok else 0), "passed": bool(ok), "msg": msg})
    if ok:
        score["points"] += pts

@pytest.mark.asyncio
async def test_memory_storage_with_correct_user_and_category():
    """Test that memories are stored with the correct user and category, and NOT stored when rejected."""
    score = {"candidate": CANDIDATE_NAME, "bucket": "memory_category", "points": 0, "max_points": 9, "details": []}
    failures = []
    try:
        # Determine which source directory to use based on the agent filename
        if "expert_src" in str(DEFAULT_AGENT_FILENAME):
            src_path = os.path.join(os.path.dirname(__file__), "..", "expert_src")
        else:
            src_path = os.path.join(os.path.dirname(__file__), "..", "..", "src")

        if src_path not in sys.path:
            sys.path.insert(0, src_path)  # Insert at beginning to take precedence

        from memory_agent.context import Context
        from memory_agent.graph import builder

        mod, err = _load_module(DEFAULT_AGENT_PATH)
        if mod is None:
            raise Exception(f"Failed to load module: {err}")
        
        builder = _get_builder(mod)
        if builder is None:
            raise Exception(f"Failed to get builder: {err}")
        
        mem_store = InMemoryStore()
        checkpointer = MemorySaver()
        graph = builder.compile(store=mem_store, checkpointer=checkpointer)

        # Test 1: Memory storage when accepted (Professional category)
        user_input_accept_1 = "I work as a data scientist at Google and love my job. Remember this."
        expected_category_1 = "professional"
        user_id_accept_1 = "test-user-accept-1"
        config_accept_1 = {"thread_id": f"{get_git_branch()}_accept_1"}

        res = await graph.ainvoke(
            {"messages": [("user", user_input_accept_1)]},
            config_accept_1,
            context=Context(user_id=user_id_accept_1),
        )
        
        if "__interrupt__" not in res:
            failures.append(f"No interrupt detected for professional acceptance test")
            _add(score, 0, "accept_professional_interrupt", False, "No interrupt detected for professional acceptance test")
        else:
            _add(score, 2, "accept_professional_interrupt", True, "Interrupt detected for professional acceptance test")
            
            while "__interrupt__" in res:                
                res = await graph.ainvoke(
                    Command(resume="accept"),
                    config_accept_1,
                    context=Context(user_id=user_id_accept_1),
                )
            
            with open("accept_professional_post_interrupt.txt", "w") as f:
                f.write(str(res))
        
        # Check that memory was stored in professional category
        namespace = ("memories", user_id_accept_1, expected_category_1)
        memories = mem_store.search(namespace)
        with open("accept_professional_memories.txt", "w") as f:
            f.write(str(memories))
            
        if len(memories) > 0:
            _add(score, 2, "memory_stored_professional", True, f"Memory correctly stored in {expected_category_1} category when accepted")
        else:
            namespace = ("memories", user_id_accept_1)
            memories = mem_store.search(namespace)
            with open("accept_professional_memories.txt", "w") as f:
                f.write(str(memories))
            if len(memories) > 0:
                _add(score, 1, "memory_stored_professional", True, f"Memory found but in wrong category(ies): {len(memories)} items found (expected: {expected_category_1})")
            else:
                failures.append(f"No memory found in {expected_category_1} category when accepted")
                _add(score, 0, "memory_stored_professional", False, f"No memory found in {expected_category_1} category when accepted")

        # Test 2: Memory storage when accepted (Personal category)
        user_input_accept_2 = "My favorite hobby is playing guitar and I practice every evening. Remember this."
        expected_category_2 = "personal"
        user_id_accept_2 = "test-user-accept-2"
        config_accept_2 = {"thread_id": f"{get_git_branch()}_accept_2"}

        res = await graph.ainvoke(
            {"messages": [("user", user_input_accept_2)]},
            config_accept_2,
            context=Context(user_id=user_id_accept_2),
        )
        
        if "__interrupt__" not in res:
            failures.append(f"No interrupt detected for personal acceptance test")
            _add(score, 0, "accept_personal_interrupt", False, "No interrupt detected for personal acceptance test")
        else:
            _add(score, 2, "accept_personal_interrupt", True, "Interrupt detected for personal acceptance test")
            
            while "__interrupt__" in res:                
                res = await graph.ainvoke(
                    Command(resume="accept"),
                    config_accept_2,
                    context=Context(user_id=user_id_accept_2),
                )
            
            with open("accept_personal_post_interrupt.txt", "w") as f:
                f.write(str(res))
            
        # Check that memory was stored in personal category
        namespace = ("memories", user_id_accept_2, expected_category_2)
        memories = mem_store.search(namespace)
        with open("accept_personal_memories.txt", "w") as f:
            f.write(str(memories))
            
        if len(memories) > 0:
            _add(score, 2, "memory_stored_personal", True, f"Memory correctly stored in {expected_category_2} category when accepted")
        else:
            namespace = ("memories", user_id_accept_2)
            memories = mem_store.search(namespace)
            with open("accept_personal_memories.txt", "w") as f:
                f.write(str(memories))
            if len(memories) > 0:
                _add(score, 1, "memory_stored_personal", True, f"Memory found but in wrong category(ies): {len(memories)} items found (expected: {expected_category_2})")
            else:
                failures.append(f"No memory found in {expected_category_2} category when accepted")
                _add(score, 0, "memory_stored_personal", False, f"No memory found in {expected_category_2} category when accepted")

        # Test 3: Memory NOT stored when rejected
        user_input_reject = "I work as a software engineer at Microsoft and enjoy coding. Remember this."
        expected_category_reject = "professional"
        user_id_reject = "test-user-reject"
        config_reject = {"thread_id": f"{get_git_branch()}_reject"}

        res = await graph.ainvoke(
            {"messages": [("user", user_input_reject)]},
            config_reject,
            context=Context(user_id=user_id_reject),
        )
        
        if "__interrupt__" not in res:
            failures.append(f"No interrupt detected for rejection test")
            _add(score, 0, "reject_interrupt_detected", False, "No interrupt detected for rejection test")
        else:
            _add(score, 2, "reject_interrupt_detected", True, "Interrupt detected for rejection test")
            
            while "__interrupt__" in res:                
                res = await graph.ainvoke(
                    Command(resume="reject"),
                    config_reject,
                    context=Context(user_id=user_id_reject),
                )
            
            with open("reject_test_post_interrupt.txt", "w") as f:
                f.write(str(res))
            
        # Check that NO memory was stored when rejected
        namespace = ("memories", user_id_reject)
        memories = mem_store.search(namespace)
        with open("reject_test_memories.txt", "w") as f:
            f.write(str(memories))
            
        if len(memories) == 0:
            _add(score, 1, "memory_not_stored_on_reject", True, f"Memory correctly NOT stored in {expected_category_reject} category when rejected")
        else:
            failures.append(f"Memory was incorrectly stored in {expected_category_reject} category despite rejection")
            _add(score, 0, "memory_not_stored_on_reject", False, f"Memory was incorrectly stored in {expected_category_reject} category despite rejection")

        
        

    except Exception as e:
        failures.append(f"Failed to execute test: {e}")
        _add(score, 0, "test_execution_error", False, str(e))
    
    _write_score(score)
    if failures:
        pytest.fail(" | ".join(failures))
