# flow_state_manager.py
from typing import Dict

class FlowStateManager:
    """
    Tracks the current flow step for each session and advances using law_firm.json.
    """

    def __init__(self, flow_json: dict):
        self.flow = flow_json
        self.step_index = self._index_steps()
        # in-memory store: session_id -> {"current_step": str, "answers": dict}
        self.sessions: Dict[str, Dict] = {}

    def _index_steps(self) -> dict:
        index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                index[step["id"]] = step
        return index

    # -------- session helpers --------
    def ensure_session(self, session_id: str):
        """
        Ensure a session with this ID exists; if not, create it with default state.
        This lets the API supply its own session_id instead of letting
        FlowStateManager generate one.
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "current_step": "start",
                "answers": {}
            }

    def get_current_step(self, session_id: str) -> str:
        self.ensure_session(session_id)
        return self.sessions[session_id]["current_step"]

    def set_current_step(self, session_id: str, step_id: str):
        self.ensure_session(session_id)
        self.sessions[session_id]["current_step"] = step_id

    def get_step(self, step_id: str) -> dict:
        return self.step_index[step_id]

    def get_prompt(self, step_id: str) -> str:
        return self.step_index[step_id]["prompt"]

    # -------- answer detection --------
    def did_user_answer_step(self, step_id: str, user_message: str) -> bool:
        step = self.get_step(step_id)
        msg = (user_message or "").strip().lower()
        itype = step.get("input_type")

        if itype == "none":
            return True

        if itype == "text":
            return len(msg) > 0

        if itype in ("choice", "yes_no"):
            # Special handling: date/time-like answers for pi_intro
            if step_id == "pi_intro":
                if any(c in msg for c in ["/", "-", ":"]) or any(
                    m in msg for m in [
                        "yesterday", "today", "ago", "last week", "last month",
                        "january", "february", "march", "april", "may", "june",
                        "july", "august", "september", "october", "november", "december"
                    ]
                ):
                    return True

            for opt in step.get("options", []):
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return True

        # file / other types can be added later
        return False

    # -------- next step logic --------
    def determine_next_step(self, current_step_id: str, user_message: str) -> str | None:
        step = self.get_step(current_step_id)
        msg = (user_message or "").lower()

        # personal-injury “other” routing
        if current_step_id == "pi_injury_type":
            for opt in step.get("options", []):
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return opt["next_step"]
            # fallback: long free-text description → go to treatment
            if len(msg.split()) > 3:
                return "pi_medical_treatment"

        # pi_injury_details always moves to pi_medical_treatment
        if current_step_id == "pi_injury_details":
            return step.get("next_step", "pi_medical_treatment")

        if "options" in step:
            for opt in step["options"]:
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return opt.get("next_step")

        return step.get("next_step")

    def advance_if_answered(self, session_id: str, user_message: str) -> str:
        """
        If user answered the current step, move to the next one.
        Otherwise leave current_step alone.
        """
        self.ensure_session(session_id)
        current = self.sessions[session_id]["current_step"]

        if not self.did_user_answer_step(current, user_message):
            return current

        nxt = self.determine_next_step(current, user_message)
        if nxt:
            self.sessions[session_id]["current_step"] = nxt
        return self.sessions[session_id]["current_step"]
