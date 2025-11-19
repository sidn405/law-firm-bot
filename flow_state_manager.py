import uuid

class FlowStateManager:
    """
    Universal reusable engine to run JSON-based scripted flows.
    Designed for multi-business chatbots (law firm, medical, restaurant, etc.)
    """

    def __init__(self, flow_json):
        """
        flow_json = your parsed law_firm.json
        """
        self.flow = flow_json
        self.step_index = self._index_steps()

        # { session_id: { "current_step": "pi_intro", "answers": {...}} }
        self.sessions = {}

    # ------------------------------
    # Build step index
    # ------------------------------
    def _index_steps(self):
        index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                index[step["id"]] = step
        return index

    # ------------------------------
    # Session Handling
    # ------------------------------
    def start_session(self):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "current_step": "start",   # default
            "answers": {}
        }
        return session_id

    def get_state(self, session_id):
        return self.sessions.get(session_id)

    def set_state(self, session_id, state):
        self.sessions[session_id] = state

    # ------------------------------
    # Step Logic
    # ------------------------------
    def get_current_step(self, session_id):
        state = self.get_state(session_id)
        return state["current_step"]

    def get_step_prompt(self, step_id):
        return self.step_index[step_id]["prompt"]

    def get_step(self, step_id):
        return self.step_index[step_id]

    # ------------------------------
    # Determine next step
    # ------------------------------
    def determine_next_step(self, current_step_id, user_message):
        step = self.get_step(current_step_id)

        # If step has multiple options (choice / yes_no)
        if "options" in step:
            msg = user_message.lower()

            for opt in step["options"]:
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return opt["next_step"]

        # Text-based (just go to next_step)
        if "next_step" in step:
            return step["next_step"]

        return None

    # ------------------------------
    # Check if user answered current step
    # ------------------------------
    def did_user_answer_step(self, step_id, user_message):
        step = self.get_step(step_id)

        if step["input_type"] == "none":
            return True  # automatically continue

        if step["input_type"] == "text":
            return len(user_message.strip()) > 0

        if step["input_type"] == "file":
            # You can customize this later
            return False

        if step["input_type"] in ["choice", "yes_no"]:
            msg = user_message.lower()
            for opt in step.get("options", []):
                if opt["label"].lower() in msg or opt["value"].lower() in msg:
                    return True

        return False

    # ------------------------------
    # Advance flow (only when answered)
    # ------------------------------
    def advance_step(self, session_id, user_message):
        state = self.get_state(session_id)
        current_step = state["current_step"]

        answered = self.did_user_answer_step(current_step, user_message)

        if not answered:
            return current_step  # do not advance

        # Move to next step
        next_step = self.determine_next_step(current_step, user_message)

        if next_step:
            state["current_step"] = next_step
            self.set_state(session_id, state)

        return next_step or current_step
