"""
LAW FIRM CHATBOT - JSON FLOW INTEGRATION
Integrates structured intake flows from law_firm.json with OpenAI conversations
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path

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
class HybridChatbot:
    """
    Combines OpenAI conversational AI with structured JSON flows
    Uses OpenAI for natural conversation, JSON flows for intake
    """
    
    def __init__(self, flow_manager: FlowManager, openai_chatbot):
        self.flow_manager = flow_manager
        self.openai_chatbot = openai_chatbot
        
    async def process_message(self, session_id: str, message: str, conversation_history: List[Dict], knowledge_base: str = "") -> Dict:
        """
        Process message using either structured flow or OpenAI conversation
        """
        
        # Check if user is in a structured flow
        if self.flow_manager.is_in_flow(session_id):
            return self.handle_flow_message(session_id, message)
        
        # Check if message should trigger a flow
        flow_triggers = {
            'personal injury': 'personal_injury_intake',
            'car accident': 'personal_injury_intake',
            'slip and fall': 'personal_injury_intake',
            'family law': 'family_law_intake',
            'immigration': 'immigration_intake',
            'schedule': 'scheduling_flow',
            'consultation': 'scheduling_flow',
        }
        
        message_lower = message.lower()
        for trigger, flow_id in flow_triggers.items():
            if trigger in message_lower:
                # Start structured flow
                flow_response = self.flow_manager.start_flow(session_id, flow_id)
                return {
                    'response': flow_response['prompt'],
                    'flow_active': True,
                    'input_type': flow_response['input_type'],
                    'options': flow_response.get('options', [])
                }
        
        # Use OpenAI for general conversation
        response = await self.openai_chatbot.chat(
            message=message,
            conversation_history=conversation_history,
            knowledge_base=knowledge_base
        )
        
        return {
            'response': response,
            'flow_active': False
        }
    
    def handle_flow_message(self, session_id: str, message: str) -> Dict:
        """Handle message during active flow"""
        
        current_step = self.flow_manager.get_current_step(session_id)
        
        # Process based on input type
        if current_step['input_type'] == 'choice':
            # User should select an option
            # Try to match message to option value or label
            for option in current_step.get('options', []):
                if message.lower() in [option['value'].lower(), option['label'].lower()]:
                    flow_response = self.flow_manager.process_response(
                        session_id, 
                        message, 
                        selected_option=option['value']
                    )
                    break
            else:
                # No match, ask again
                return {
                    'response': f"{current_step['prompt']}\n\nPlease select one of the options:",
                    'flow_active': True,
                    'input_type': current_step['input_type'],
                    'options': current_step['options']
                }
        else:
            # Text, yes_no, or other input
            flow_response = self.flow_manager.process_response(session_id, message)
        
        # Check if flow is complete
        if flow_response.get('completed'):
            return {
                'response': flow_response['message'],
                'flow_active': False,
                'flow_complete': True,
                'collected_data': flow_response['data']
            }
        
        return {
            'response': flow_response['prompt'],
            'flow_active': True,
            'input_type': flow_response['input_type'],
            'options': flow_response.get('options', [])
        }


# Example usage in main.py:
"""
# Initialize flow manager
flow_manager = FlowManager("law_firm.json")

# Create hybrid chatbot
hybrid_bot = HybridChatbot(flow_manager, chatbot)

# In chat endpoint:
@app.post("/api/chat")
async def chat_endpoint(chat: ChatMessage, db: Session = Depends(get_db)):
    # Get conversation history
    conversation = db.query(Conversation).filter(
        Conversation.session_id == session_id
    ).first()
    
    # Get knowledge base
    knowledge_base = await scraper.get_knowledge_base()
    
    # Process with hybrid bot
    response_data = await hybrid_bot.process_message(
        session_id=session_id,
        message=chat.message,
        conversation_history=conversation.messages if conversation else [],
        knowledge_base=knowledge_base
    )
    
    # If flow is complete, save collected data to case
    if response_data.get('flow_complete'):
        # Create case with collected data
        case = Case(
            client_id=chat.client_id,
            case_type=response_data['collected_data'].get('pi_injury_type', 'unknown'),
            intake_data=response_data['collected_data']
        )
        db.add(case)
        db.commit()
    
    return response_data
"""