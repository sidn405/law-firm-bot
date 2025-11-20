import React, { useState, useEffect, useRef } from 'react';
import { Send, Phone, Mail } from 'lucide-react';

const LawFirmChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [currentStep, setCurrentStep] = useState('start');
  const [isLoading, setIsLoading] = useState(false);
  const [showButtons, setShowButtons] = useState(true);
  const messagesEndRef = useRef(null);

  // Flow definition from law_firm.json
  const flowSteps = {
    start: {
      prompt: "Hello and welcome to our law firm. I can help you with a free case evaluation, scheduling, or questions about our services.\n\nWhat type of legal issue do you need help with?",
      input_type: "choice",
      options: [
        { value: "personal_injury", label: "Personal Injury", next_step: "pi_intro" },
        { value: "family_law", label: "Family Law", next_step: "family_intro" },
        { value: "immigration", label: "Immigration", next_step: "imm_intro" },
        { value: "criminal_defense", label: "Criminal Defense", next_step: "crim_intro" },
        { value: "workers_comp", label: "Workers' Compensation", next_step: "wc_intro" },
        { value: "business_law", label: "Business / Contract", next_step: "biz_intro" },
        { value: "estate_planning", label: "Estate Planning / Wills", next_step: "estate_intro" },
        { value: "other", label: "Something else", next_step: "other_intro" }
      ]
    },
    pi_intro: {
      prompt: "I'm sorry you're going through this. I'll ask a few quick questions so our team can review your situation.\n\nWhat date did the incident occur?",
      input_type: "choice",
      options: [
        { value: "11/12/2025", label: "November 12, 2025", next_step: "pi_injury_type" },
        { value: "today", label: "Today", next_step: "pi_injury_type" },
        { value: "30_days", label: "Within the last 30 days", next_step: "pi_injury_type" },
        { value: "6_months", label: "1–6 months ago", next_step: "pi_injury_type" },
        { value: "over_6_months", label: "Over 6 months ago", next_step: "pi_injury_type" }
      ]
    },
    pi_injury_type: {
      prompt: "How were you injured?",
      input_type: "choice",
      options: [
        { value: "car_accident", label: "Car accident", next_step: "pi_medical_treatment" },
        { value: "slip_fall", label: "Slip and fall", next_step: "pi_medical_treatment" },
        { value: "work_injury", label: "Workplace injury", next_step: "pi_medical_treatment" },
        { value: "med_mal", label: "Medical malpractice", next_step: "pi_medical_treatment" },
        { value: "other", label: "Other (explain)", next_step: "pi_injury_details" }
      ]
    },
    pi_injury_details: {
      prompt: "Please briefly describe what happened.",
      input_type: "text",
      next_step: "pi_medical_treatment"
    },
    pi_medical_treatment: {
      prompt: "Have you received any medical treatment for this injury?",
      input_type: "choice",
      options: [
        { value: "yes", label: "Yes", next_step: "pi_has_attorney" },
        { value: "no", label: "No", next_step: "pi_has_attorney" }
      ]
    },
    pi_has_attorney: {
      prompt: "Do you already have an attorney for this matter?",
      input_type: "choice",
      options: [
        { value: "no", label: "No", next_step: "pi_contact_info" },
        { value: "yes", label: "Yes", next_step: "pi_existing_attorney" }
      ]
    },
    pi_existing_attorney: {
      prompt: "Thanks for letting us know. Because you're already represented, we may not be able to take your case, but you're welcome to contact our office for general questions.",
      input_type: "none",
      next_step: "end"
    },
    pi_contact_info: {
      prompt: "Please share your contact details so our team can review your case and reach out.\n\n1) Full name\n2) Phone number\n3) Email address",
      input_type: "text",
      next_step: "pi_docs"
    },
    pi_docs: {
      prompt: "If you have photos, police reports, or medical documents, you can upload them here (optional).\n\nType 'skip' if you don't have any documents right now, or 'done' when you've uploaded them.",
      input_type: "text",
      next_step: "pi_consult"
    },
    pi_consult: {
      prompt: "Would you like to schedule a free consultation now?",
      input_type: "choice",
      options: [
        { value: "yes", label: "Yes, schedule a consultation", next_step: "pi_schedule" },
        { value: "no", label: "No, just have someone call me", next_step: "pi_followup" }
      ]
    },
    pi_schedule: {
      prompt: "Great. Please provide your preferred date and time (e.g., 'November 25 at 2pm'), and we'll confirm by SMS/email.",
      input_type: "text",
      next_step: "pi_end"
    },
    pi_followup: {
      prompt: "Thanks. Our team will review your information and contact you as soon as possible.",
      input_type: "none",
      next_step: "pi_end"
    },
    pi_end: {
      prompt: "Your intake has been submitted. You'll receive a confirmation shortly. If this is urgent, please call our office at (555) 123-4567.",
      input_type: "none"
    },
    other_intro: {
      prompt: "Please briefly describe the legal issue you're dealing with.",
      input_type: "text",
      next_step: "other_contact_info"
    },
    other_contact_info: {
      prompt: "Thank you. Please provide your name, phone number, and email so our team can follow up.",
      input_type: "text",
      next_step: "other_end"
    },
    other_end: {
      prompt: "Thanks. Your message has been sent to our intake team. Expect a response within one business day.",
      input_type: "none"
    },
    family_intro: {
      prompt: "I understand family matters can be difficult. Our team specializes in divorce, custody, and family law.\n\nPlease provide your contact information so we can schedule a consultation.",
      input_type: "text",
      next_step: "other_end"
    },
    imm_intro: {
      prompt: "We can help with immigration matters including visas, green cards, and citizenship.\n\nPlease provide your contact information and a brief description of your situation.",
      input_type: "text",
      next_step: "other_end"
    },
    crim_intro: {
      prompt: "We understand this is a serious matter. Our criminal defense team is here to help.\n\nPlease provide your contact information and a brief description of your situation.",
      input_type: "text",
      next_step: "other_end"
    },
    wc_intro: {
      prompt: "We can help you get the workers' compensation benefits you deserve.\n\nPlease provide your contact information and details about your workplace injury.",
      input_type: "text",
      next_step: "other_end"
    },
    biz_intro: {
      prompt: "Our business law team can help with contracts, disputes, and business formation.\n\nPlease provide your contact information and a brief description of your needs.",
      input_type: "text",
      next_step: "other_end"
    },
    estate_intro: {
      prompt: "We can help you plan for the future with wills, trusts, and estate planning.\n\nPlease provide your contact information to schedule a consultation.",
      input_type: "text",
      next_step: "other_end"
    }
  };

  // Display greeting on mount
  useEffect(() => {
    const greeting = flowSteps.start;
    setMessages([{
      role: 'assistant',
      content: greeting.prompt,
      timestamp: new Date().toISOString(),
      step: 'start',
      options: greeting.options
    }]);
  }, []);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleButtonClick = (option) => {
    setShowButtons(false);
    
    // Add user selection
    const userMessage = {
      role: 'user',
      content: option.label,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    // Get next step
    const nextStep = option.next_step;
    const nextStepData = flowSteps[nextStep];

    if (!nextStepData) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Thank you for your inquiry. Our team will contact you shortly.',
        timestamp: new Date().toISOString()
      }]);
      return;
    }

    // Add assistant response
    setTimeout(() => {
      const assistantMessage = {
        role: 'assistant',
        content: nextStepData.prompt,
        timestamp: new Date().toISOString(),
        step: nextStep,
        options: nextStepData.options,
        inputType: nextStepData.input_type
      };
      setMessages(prev => [...prev, assistantMessage]);
      setCurrentStep(nextStep);
      setShowButtons(true);
    }, 500);
  };

  const handleTextSubmit = () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const currentStepData = flowSteps[currentStep];
    const nextStepId = currentStepData.next_step;
    const nextStepData = flowSteps[nextStepId];

    setTimeout(() => {
      if (nextStepData) {
        const assistantMessage = {
          role: 'assistant',
          content: nextStepData.prompt,
          timestamp: new Date().toISOString(),
          step: nextStepId,
          options: nextStepData.options,
          inputType: nextStepData.input_type
        };
        setMessages(prev => [...prev, assistantMessage]);
        setCurrentStep(nextStepId);
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Thank you. Your information has been received.',
          timestamp: new Date().toISOString()
        }]);
      }
      setIsLoading(false);
    }, 500);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleTextSubmit();
    }
  };

  const currentStepData = flowSteps[currentStep];
  const needsTextInput = currentStepData?.input_type === 'text';

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-900 to-indigo-900 text-white p-6 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">Legal Intake Assistant</h1>
          <p className="text-blue-200 flex items-center gap-2">
            <Phone className="w-4 h-4" />
            24/7 Free Case Evaluation
          </p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-4xl mx-auto w-full">
        {messages.map((msg, idx) => (
          <div key={idx}>
            <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} mb-2`}>
              <div className={`max-w-xl rounded-2xl px-6 py-4 ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-br-none' 
                  : 'bg-white text-gray-800 shadow-md rounded-bl-none border border-gray-200'
              }`}>
                <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
              </div>
            </div>

            {/* Show buttons for choice steps */}
            {msg.role === 'assistant' && msg.options && showButtons && idx === messages.length - 1 && (
              <div className="flex flex-wrap gap-2 mt-4 ml-2">
                {msg.options.map((option, optIdx) => (
                  <button
                    key={optIdx}
                    onClick={() => handleButtonClick(option)}
                    className="px-6 py-3 bg-white hover:bg-blue-50 text-blue-900 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 border-2 border-blue-200 hover:border-blue-400 font-medium"
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-2xl px-6 py-4 shadow-md">
              <div className="flex gap-2">
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Area - Only show for text input steps */}
      {needsTextInput && (
        <div className="border-t border-gray-300 bg-white p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Type your response here..."
                className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 text-gray-800"
                disabled={isLoading}
              />
              <button
                onClick={handleTextSubmit}
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2 font-medium"
              >
                <Send className="w-4 h-4" />
                Send
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="bg-gray-100 border-t border-gray-300 p-3 text-center text-sm text-gray-600">
        <div className="flex items-center justify-center gap-4">
          <span className="flex items-center gap-1">
            <Phone className="w-4 h-4" />
            (555) 123-4567
          </span>
          <span className="flex items-center gap-1">
            <Mail className="w-4 h-4" />
            info@lawfirm.com
          </span>
        </div>
      </div>
    </div>
  );
};

export default LawFirmChatbot;