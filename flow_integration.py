"""
LAW FIRM CHATBOT - JSON FLOW INTEGRATION
Integrates structured intake flows from law_firm.json with OpenAI conversations
"""
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from flow_state_manager import FlowStateManager
import asyncio

class FlowManager:
    """Manages structured conversation flows from JSON"""
    
    def __init__(self, flow_file: str = "law_firm.json"):
        self.flow_file = Path(flow_file)
        self.flows = self.load_flows()
        self.active_sessions = {}  # session_id -> flow state
        
    def load_flows(self) -> Dict:
        """Load flows from JSON file"""
        if not self.flow_file.exists():
            print(f"Warning: Flow file {self.flow_file} not found")
            return {}
        
        with open(self.flow_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def start_flow(self, session_id: str, flow_id: str) -> Dict:
        """Start a new flow for a session"""
        if flow_id not in [flow['id'] for flow in self.flows.get('flows', [])]:
            return {"error": f"Flow {flow_id} not found"}
        
        # Get flow
        flow = next(f for f in self.flows['flows'] if f['id'] == flow_id)
        
        # Initialize session state
        self.active_sessions[session_id] = {
            'flow_id': flow_id,
            'current_step_id': flow['steps'][0]['id'],
            'flow_data': {},
            'step_index': 0
        }
        
        # Get first step
        return self.get_current_step(session_id)
    
    def get_current_step(self, session_id: str) -> Dict:
        """Get current step for a session"""
        if session_id not in self.active_sessions:
            # Return main menu
            return self.get_main_menu()
        
        state = self.active_sessions[session_id]
        flow = next(f for f in self.flows['flows'] if f['id'] == state['flow_id'])
        step = next(s for s in flow['steps'] if s['id'] == state['current_step_id'])
        
        return {
            'prompt': self.interpolate_variables(step['prompt'], state['flow_data']),
            'input_type': step.get('input_type', 'text'),
            'options': step.get('options', []),
            'step_id': step['id'],
            'flow_id': state['flow_id']
        }
    
    def get_main_menu(self) -> Dict:
        """Get main menu flow"""
        main_menu = next((f for f in self.flows['flows'] if f['id'] == 'main_menu'), None)
        if not main_menu:
            return {'prompt': 'How can I help you?', 'input_type': 'text', 'options': []}
        
        start_step = main_menu['steps'][0]
        return {
            'prompt': self.interpolate_variables(start_step['prompt'], {}),
            'input_type': start_step.get('input_type', 'choice'),
            'options': start_step.get('options', []),
            'step_id': start_step['id'],
            'flow_id': 'main_menu'
        }
    
    def process_response(self, session_id: str, user_input: str, selected_option: Optional[str] = None) -> Dict:
        """Process user response and advance flow"""
        
        if session_id not in self.active_sessions:
            # Handle main menu selection
            if selected_option:
                # Find next step from main menu
                main_menu = next(f for f in self.flows['flows'] if f['id'] == 'main_menu')
                start_step = main_menu['steps'][0]
                
                option = next((o for o in start_step['options'] if o['value'] == selected_option), None)
                if option and 'next_step' in option:
                    # Determine which flow contains this step
                    for flow in self.flows['flows']:
                        if any(s['id'] == option['next_step'] for s in flow['steps']):
                            return self.start_flow(session_id, flow['id'])
            
            return self.get_main_menu()
        
        state = self.active_sessions[session_id]
        current_flow = next(f for f in self.flows['flows'] if f['id'] == state['flow_id'])
        current_step = next(s for s in current_flow['steps'] if s['id'] == state['current_step_id'])
        
        # Store user input
        state['flow_data'][state['current_step_id']] = user_input
        
        # Determine next step
        next_step_id = None
        
        if selected_option and current_step.get('input_type') == 'choice':
            # User selected an option
            option = next((o for o in current_step.get('options', []) if o['value'] == selected_option), None)
            if option:
                next_step_id = option.get('next_step')
        elif 'next_step' in current_step:
            # Fixed next step
            next_step_id = current_step['next_step']
        
        if next_step_id:
            # Check if next step exists in current flow
            next_step = next((s for s in current_flow['steps'] if s['id'] == next_step_id), None)
            if next_step:
                state['current_step_id'] = next_step_id
                return self.get_current_step(session_id)
            else:
                # Flow complete
                return self.complete_flow(session_id)
        else:
            # End of flow
            return self.complete_flow(session_id)
    
    def complete_flow(self, session_id: str) -> Dict:
        """Complete a flow and return collected data"""
        if session_id not in self.active_sessions:
            return {'completed': False}
        
        state = self.active_sessions[session_id]
        flow_data = state['flow_data']
        
        # Clean up session
        del self.active_sessions[session_id]
        
        return {
            'completed': True,
            'flow_id': state['flow_id'],
            'data': flow_data,
            'message': 'Thank you for providing this information. Our team will review and contact you shortly.'
        }
    
    def interpolate_variables(self, text: str, data: Dict) -> str:
        """Replace {{variables}} in text with actual values"""
        import re
        
        # Replace {{variable}} with values from data
        def replace_var(match):
            var_name = match.group(1)
            return str(data.get(var_name, match.group(0)))
        
        return re.sub(r'\{\{(\w+)\}\}', replace_var, text)
    
    def get_intake_summary(self, session_id: str) -> Dict:
        """Get summary of collected data"""
        if session_id not in self.active_sessions:
            return {}
        
        return self.active_sessions[session_id]['flow_data']
    
    def is_in_flow(self, session_id: str) -> bool:
        """Check if session is in an active flow"""
        return session_id in self.active_sessions


# Integration with OpenAI chatbot
class HybridChatbotService:
    """
    Hybrid approach: Rules control flow, LLM only generates responses
    """
    
    def __init__(self, openai_client=None):
        self.openai_client = openai_client
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, "law_firm", "law_firm.json")

        with open(script_path, "r", encoding="utf-8") as f:
            self.flow = json.load(f)

        self.flow_manager = FlowStateManager(self.flow)

        # Build step index
        self.step_index = {}
        for flow in self.flow["flows"]:
            for step in flow["steps"]:
                self.step_index[step["id"]] = step

    async def chat(
        self,
        message: str,
        conversation_history: list,
        current_step_id: str,
        session_data: dict
    ):
        """
        Main chat method - decides whether user answered the question
        """
        
        step_data = self.step_index.get(current_step_id, {})
        
        # STEP 1: Did user answer the current question?
        answered, extracted_value = self._check_if_answered(message, step_data, session_data)
        
        # STEP 2: Generate appropriate response
        if answered:
            # User answered - acknowledge and ask next question
            response = await self._generate_acknowledgment_and_next(
                user_message=message,
                current_step_id=current_step_id,
                extracted_value=extracted_value,
                session_data=session_data
            )
            should_advance = True
        else:
            # User didn't answer - handle off-topic or clarify
            response = await self._handle_off_topic(
                user_message=message,
                current_step_id=current_step_id,
                session_data=session_data
            )
            should_advance = False
        
        return {
            "response": response,
            "should_advance": should_advance,
            "extracted_value": extracted_value
        }
    
    def _check_if_answered(self, message: str, step_data: dict, session_data: dict) -> tuple[bool, any]:
        """
        Rule-based checking if user answered the question
        Returns: (answered: bool, extracted_value: any)
        """
        
        msg_lower = message.lower().strip()
        input_type = step_data.get("input_type", "text")
        step_id = step_data.get("id")
        
        # Handle different input types
        
        if input_type == "none":
            return True, None
        
        if input_type == "choice" or input_type == "yes_no":
            # Check if message matches any option
            for option in step_data.get("options", []):
                label = option["label"].lower()
                value = option["value"].lower()
                
                if label in msg_lower or value in msg_lower:
                    return True, option["value"]
            
            # Special handling for pi_intro (date/time question)
            if step_id == "pi_intro":
                # Check for date/time patterns
                if any(pattern in msg_lower for pattern in [
                    "today", "yesterday", "last week", "last month", "ago",
                    "monday", "tuesday", "wednesday", "thursday", "friday",
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"
                ]) or "/" in message or ":" in message:
                    return True, message
            
            return False, None
        
        if input_type == "text":
            # Must be at least 3 words for meaningful answer
            if len(msg_lower.split()) >= 3:
                return True, message
            return False, None
        
        if input_type == "date":
            # Check for date patterns
            if "/" in message or "-" in message or any(
                month in msg_lower for month in [
                    "january", "february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"
                ]
            ):
                return True, message
            return False, None
        
        # Default: any non-empty response counts
        return len(msg_lower) > 0, message
    
    async def _generate_acknowledgment_and_next(
        self,
        user_message: str,
        current_step_id: str,
        extracted_value: any,
        session_data: dict
    ) -> str:
        """
        User answered - acknowledge briefly and ask next question
        LLM only generates the acknowledgment, we append the next question
        """
        
        current_step = self.step_index[current_step_id]
        next_step_id = self._determine_next_step(current_step_id, extracted_value)
        
        if not next_step_id or next_step_id == "end":
            # Flow complete
            return current_step.get("prompt", "Thank you for providing that information.")
        
        next_step = self.step_index[next_step_id]
        next_question = next_step.get("prompt", "")
        
        # Generate brief acknowledgment with LLM
        acknowledgment = await self._generate_brief_ack(user_message, current_step)
        
        # CRITICAL: Always append the next scripted question
        return f"{acknowledgment}\n\n{next_question}"
    
    async def _generate_brief_ack(
        self,
        user_message: str,
        current_step: dict
    ) -> str:
        """Generate a 1-sentence acknowledgment"""
        
        if not self.openai_client:
            return "Got it."
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're a legal intake assistant. Generate a brief "
                            "1-sentence acknowledgment (under 10 words). "
                            "Be empathetic but concise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"User answered: '{user_message}'\n\n"
                            "Generate brief acknowledgment:"
                        ),
                    },
                ],
                max_tokens=30,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "Thank you."

    
    async def _handle_off_topic(
        self,
        user_message: str,
        current_step_id: str,
        session_data: dict
    ) -> str:
        """
        User didn't answer - either off-topic or unclear
        Answer briefly, then re-ask the scripted question
        """
        
        current_step = self.step_index[current_step_id]
        scripted_question = current_step.get("prompt", "")
        
        # Generate brief response to their off-topic question
        async def _handle_off_topic(
            self,
            user_message: str,
            current_step_id: str,
            session_data: dict,
        ) -> str:
            """
            User didn't answer - either off-topic or unclear
            Answer briefly, then re-ask the scripted question
            """
            current_step = self.step_index[current_step_id]
            scripted_question = current_step.get("prompt", "")
            
            # Generate brief response to their off-topic question
            if self.openai_client:
                try:
                    response = await asyncio.to_thread(
                        self.openai_client.chat.completions.create,
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You're a legal intake assistant. Answer this "
                                    "question in 1 sentence (under 20 words), then stop. "
                                    "If you don't know, say 'I don't have that information.'"
                                ),
                            },
                            {
                                "role": "user",
                                "content": user_message,
                            },
                        ],
                        max_tokens=50,
                        temperature=0.5,
                    )
                    brief_answer = response.choices[0].message.content.strip()
                    return f"{brief_answer}\n\n{scripted_question}"
                except Exception:
                    # fall through to plain re-ask
                    pass

            # Fallback: just re-ask the question
            return f"Let me ask you this: {scripted_question}"

    
    def _determine_next_step(self, current_step_id: str, user_value: any) -> str:
        """Determine next step based on rules in JSON"""
        
        step = self.step_index[current_step_id]
        
        # Check if there are conditional options
        if "options" in step:
            for option in step["options"]:
                if option.get("value") == user_value:
                    return option.get("next_step")
        
        # Otherwise use default next_step
        return step.get("next_step")