
/* =====================================================
   MULTI-STYLE CHAT WIDGET (CSP-friendly)
   - No inline scripts or inline event handlers
   - Uses external JS only (allowed by script-src 'self')
===================================================== */

const WIDGET_IDS = ["w1","w2","w3","w4","w5","w6"];
let ACTIVE_WID = "w6";

// Widget state storage
const widgetStates = {};
WIDGET_IDS.forEach(wid => {
  widgetStates[wid] = {
    sessionId: null,
    clientId: null,
    currentStep: 'start',
    intakeActive: true,
    collectedData: {},
    stripe: null,
    paymentFlowActive: false,
    paymentClientData: {},
    awaitingFarewellResponse: false,
    paymentFlow: {
      active: false,
      step: null,
      selectedType: null,
      clientInfo: {}
    }
  };
});

function setActiveWidget(wid) {
  ACTIVE_WID = wid;
}

function getEl(id) {
  return document.getElementById(`${ACTIVE_WID}-${id}`);
}

function getElFor(wid, id) {
  return document.getElementById(`${wid}-${id}`);
}

function setFrontWidget(activeWid) {
  WIDGET_IDS.forEach((wid) => {
    const container = document.getElementById(`${wid}-lawfirm-chatbot-container`);
    if (!container) return;
    if (activeWid && wid === activeWid) container.classList.add("is-front");
    else container.classList.remove("is-front");
  });
}

function closeOtherWidgets(activeWid) {
  WIDGET_IDS.forEach((wid) => {
    if (wid === activeWid) return;

    const win = getElFor(wid, "lawfirm-chat-window");
    if (win) win.classList.remove("open");

    const overlay1 = getElFor(wid, "lawfirm-overlay");
    if (overlay1) overlay1.classList.remove("open");

    const overlay2 = getElFor(wid, "chat-modal-overlay");
    if (overlay2) overlay2.classList.remove("open");
  });

  setFrontWidget(activeWid || null);
}

function bringToFront(wid) {
  // remove from all
  WIDGET_IDS.forEach((id) => {
    const c = document.getElementById(`${id}-lawfirm-chatbot-container`);
    if (c) c.classList.remove("is-front");
  });
  // add to active
  const active = document.getElementById(`${wid}-lawfirm-chatbot-container`);
  if (active) active.classList.add("is-front");
}


        // =====================================================
        // LAW FIRM CHATBOT - FRONTEND LOGIC W/ SCRIPTED INTAKE
        // =====================================================
        
        // Configuration - UPDATE THIS WITH YOUR RAILWAY URL
        const API_URL = 'https://law-firm-bot-production.up.railway.app';
        let sessionId = null;
        let clientId = null;
        let currentStep = 'start';
        let intakeActive = true;
        let collectedData = {};
        let stripe = null;
        let paymentFlowActive = false;
        let paymentClientData = {};
        let awaitingFarewellResponse = false;
        
        let paymentFlow = {
            active: false,
            step: null,
            selectedType: null,
            clientInfo: {}
        };
        
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
                    { value: "other", label: "Something else", next_step: "other_intro" },
                    { value: "make_payment", label: "💳 Make a Payment", next_step: "payment_flow" }
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
                prompt: `
                    <div class="contact-info-box" style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="font-size: 14px; color: #1e40af; font-weight: 600; margin-bottom: 6px;">📋 Contact Information</p>
                        <p style="font-size: 13px; color: #1f2937; margin-bottom: 8px; line-height: 1.4;">Please share your contact details so our team can review your case and reach out.</p>
                        <p style="font-size: 13px; color: #374151; margin-bottom: 6px;"><strong>Enter:</strong> Full name, phone number, email (separated by commas)</p>
                        <p style="font-size: 13px; color: #374151; background: white; padding: 6px 8px; border-radius: 4px; font-family: monospace; margin-bottom: 0;">
                            John Smith, 555-123-4567, john@email.com
                        </p>
                    </div>
                `,
                input_type: "text",
                next_step: "pi_docs"
            },
            pi_docs: {
                prompt: `
                    <div class="docs-info-box" style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="font-size: 14px; color: #065f46; font-weight: 600; margin-bottom: 6px;">📎 Documents (Optional)</p>
                        <p style="font-size: 13px; color: #1f2937; margin-bottom: 8px; line-height: 1.4;">If you have photos, police reports, or medical documents, you can upload them here.</p>
                        <p style="font-size: 13px; color: #374151; background: white; padding: 6px 8px; border-radius: 4px; margin-bottom: 0; line-height: 1.5;">
                            Type <strong>'skip'</strong> if you don't have any documents right now<br>
                            Type <strong>'done'</strong> when you've uploaded them
                        </p>
                    </div>
                `,
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
                prompt: "Great! Let me check our availability and book your consultation.\n\nWhat date and time works best for you? (e.g., 'November 28 at 2pm' or 'next Tuesday at 10am')",
                input_type: "text",
                next_step: "pi_schedule_process"
            },
            pi_schedule_process: {
                prompt: "Let me check our calendar...",
                input_type: "none",
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
            payment_flow: {
                prompt: "I'll help you make a payment. Let me start the payment process...",
                input_type: "none",
                next_step: null
            },

            family_intro: {
                prompt: "I understand family matters can be difficult. Our team specializes in divorce, custody, child support, and other family law matters.\n\nWhat type of family law issue do you need help with?",
                input_type: "choice",
                options: [
                    { value: "divorce", label: "Divorce / Separation", next_step: "family_divorce_status" },
                    { value: "custody", label: "Child Custody", next_step: "family_custody_status" },
                    { value: "child_support", label: "Child Support", next_step: "family_support_intro" },
                    { value: "adoption", label: "Adoption", next_step: "family_adoption_intro" },
                    { value: "domestic_violence", label: "Domestic Violence / Restraining Order", next_step: "family_dv_intro" },
                    { value: "other_family", label: "Other family law matter", next_step: "family_other_intro" }
                ]
            },
            
            // DIVORCE PATH
            family_divorce_status: {
                prompt: "Have you or your spouse filed for divorce yet?",
                input_type: "choice",
                options: [
                    { value: "not_filed", label: "No, not yet", next_step: "family_divorce_married_length" },
                    { value: "filed_me", label: "Yes, I filed", next_step: "family_divorce_married_length" },
                    { value: "filed_spouse", label: "Yes, my spouse filed", next_step: "family_divorce_married_length" }
                ]
            },
            family_divorce_married_length: {
                prompt: "How long have you been married?",
                input_type: "choice",
                options: [
                    { value: "under_1", label: "Less than 1 year", next_step: "family_divorce_children" },
                    { value: "1_5", label: "1-5 years", next_step: "family_divorce_children" },
                    { value: "5_10", label: "5-10 years", next_step: "family_divorce_children" },
                    { value: "over_10", label: "Over 10 years", next_step: "family_divorce_children" }
                ]
            },
            family_divorce_children: {
                prompt: "Do you have children together?",
                input_type: "choice",
                options: [
                    { value: "yes_minor", label: "Yes, under 18", next_step: "family_divorce_property" },
                    { value: "yes_adult", label: "Yes, over 18", next_step: "family_divorce_property" },
                    { value: "no", label: "No children", next_step: "family_divorce_property" }
                ]
            },
            family_divorce_property: {
                prompt: "Are there significant assets or property to divide? (real estate, retirement accounts, businesses, etc.)",
                input_type: "choice",
                options: [
                    { value: "yes_complex", label: "Yes, significant assets", next_step: "family_contact_info" },
                    { value: "some", label: "Some assets", next_step: "family_contact_info" },
                    { value: "minimal", label: "Minimal assets", next_step: "family_contact_info" }
                ]
            },
            
            // CUSTODY PATH
            family_custody_status: {
                prompt: "What is your current custody situation?",
                input_type: "choice",
                options: [
                    { value: "no_order", label: "No custody order yet", next_step: "family_custody_children_count" },
                    { value: "modify_existing", label: "Need to modify existing order", next_step: "family_custody_children_count" },
                    { value: "violation", label: "Other parent violating order", next_step: "family_custody_children_count" },
                    { value: "relocation", label: "Relocation issue", next_step: "family_custody_children_count" }
                ]
            },
            family_custody_children_count: {
                prompt: "How many children are involved?",
                input_type: "choice",
                options: [
                    { value: "1", label: "1 child", next_step: "family_custody_ages" },
                    { value: "2", label: "2 children", next_step: "family_custody_ages" },
                    { value: "3_plus", label: "3 or more children", next_step: "family_custody_ages" }
                ]
            },
            family_custody_ages: {
                prompt: "What are the ages of the child(ren)? (You can type a brief answer like '5 and 8')",
                input_type: "text",
                next_step: "family_contact_info"
            },
            
            // CHILD SUPPORT PATH
            family_support_intro: {
                prompt: "Are you seeking to:",
                input_type: "choice",
                options: [
                    { value: "establish", label: "Establish child support", next_step: "family_support_custody" },
                    { value: "modify", label: "Modify existing support order", next_step: "family_support_custody" },
                    { value: "enforce", label: "Enforce unpaid support", next_step: "family_support_custody" }
                ]
            },
            family_support_custody: {
                prompt: "Is there a custody order in place?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes", next_step: "family_contact_info" },
                    { value: "no", label: "No", next_step: "family_contact_info" },
                    { value: "pending", label: "Case is pending", next_step: "family_contact_info" }
                ]
            },
            
            // ADOPTION PATH
            family_adoption_intro: {
                prompt: "What type of adoption are you interested in?",
                input_type: "choice",
                options: [
                    { value: "stepparent", label: "Stepparent adoption", next_step: "family_contact_info" },
                    { value: "relative", label: "Relative/kinship adoption", next_step: "family_contact_info" },
                    { value: "domestic", label: "Domestic infant adoption", next_step: "family_contact_info" },
                    { value: "foster", label: "Foster care adoption", next_step: "family_contact_info" }
                ]
            },
            
            // DOMESTIC VIOLENCE PATH
            family_dv_intro: {
                prompt: "I'm sorry you're going through this. Your safety is the priority.\n\nDo you need an emergency protective order?",
                input_type: "choice",
                options: [
                    { value: "emergency", label: "Yes, emergency protection needed", next_step: "family_dv_emergency" },
                    { value: "have_order", label: "I have an order, need to enforce it", next_step: "family_contact_info" },
                    { value: "modify", label: "Need to modify existing order", next_step: "family_contact_info" }
                ]
            },
            family_dv_emergency: {
                prompt: "If you are in immediate danger, please call 911 now.\n\nFor emergency legal protection, call our office immediately at (555) 123-4567.\n\nWould you like to provide contact information so we can reach you as soon as possible?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes, contact me ASAP", next_step: "family_contact_info" },
                    { value: "call_now", label: "I'll call the office now", next_step: "family_dv_resources" }
                ]
            },
            family_dv_resources: {
                prompt: "Please call us at (555) 123-4567.\n\nAdditional resources:\n• National Domestic Violence Hotline: 1-800-799-7233\n• Crisis Text Line: Text START to 88788\n\nYour safety matters. We're here to help.",
                input_type: "none"
            },
            
            // OTHER FAMILY LAW
            family_other_intro: {
                prompt: "Please briefly describe your family law matter.",
                input_type: "text",
                next_step: "family_contact_info"
            },
            
            // FAMILY LAW CONTACT INFO
            family_contact_info: {
                prompt: `
                    <div class="contact-info-box" style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="font-size: 14px; color: #1e40af; font-weight: 600; margin-bottom: 6px;">📋 Contact Information</p>
                        <p style="font-size: 13px; color: #1f2937; margin-bottom: 8px; line-height: 1.4;">Please share your contact details so our family law team can reach you for a confidential consultation.</p>
                        <p style="font-size: 13px; color: #374151; margin-bottom: 6px;"><strong>Enter:</strong> Full name, phone number, email (separated by commas)</p>
                        <p style="font-size: 13px; color: #374151; background: white; padding: 6px 8px; border-radius: 4px; font-family: monospace; margin-bottom: 0;">
                            Jane Smith, 555-123-4567, jane@email.com
                        </p>
                    </div>
                `,
                input_type: "text",
                next_step: "family_consult"
            },
            family_consult: {
                prompt: "Would you like to schedule a confidential consultation?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes, schedule consultation", next_step: "family_schedule" },
                    { value: "no", label: "No, just have someone call me", next_step: "family_end" }
                ]
            },
            family_schedule: {
                prompt: "What date and time works best for your consultation? (e.g., 'Tomorrow at 2pm' or 'This Friday at 10am')",
                input_type: "text",
                next_step: "family_end"
            },
            family_end: {
                prompt: "Thank you. Your information has been sent to our family law team. All consultations are confidential.\n\nYou'll receive a response within one business day. For urgent matters, call (555) 123-4567.",
                input_type: "none"
            },
            
            // ============================================
            // IMMIGRATION LAW INTAKE FLOW
            // ============================================
            imm_intro: {
                prompt: "We help with visas, green cards, citizenship, deportation defense, and more.\n\nWhat type of immigration matter do you need help with?",
                input_type: "choice",
                options: [
                    { value: "visa", label: "Work or Student Visa", next_step: "imm_visa_type" },
                    { value: "green_card", label: "Green Card / Permanent Residence", next_step: "imm_green_card_type" },
                    { value: "citizenship", label: "Citizenship / Naturalization", next_step: "imm_citizenship_status" },
                    { value: "deportation", label: "Deportation / Removal Defense", next_step: "imm_deportation_urgent" },
                    { value: "asylum", label: "Asylum / Refugee Status", next_step: "imm_asylum_intro" },
                    { value: "family_petition", label: "Family-Based Immigration", next_step: "imm_family_petition" },
                    { value: "other_imm", label: "Other immigration matter", next_step: "imm_other_intro" }
                ]
            },
            
            // VISA PATH
            imm_visa_type: {
                prompt: "What type of visa do you need?",
                input_type: "choice",
                options: [
                    { value: "h1b", label: "H-1B (Work visa)", next_step: "imm_visa_status" },
                    { value: "f1", label: "F-1 (Student visa)", next_step: "imm_visa_status" },
                    { value: "l1", label: "L-1 (Intracompany transfer)", next_step: "imm_visa_status" },
                    { value: "o1", label: "O-1 (Extraordinary ability)", next_step: "imm_visa_status" },
                    { value: "other_work", label: "Other work visa", next_step: "imm_visa_status" }
                ]
            },
            imm_visa_status: {
                prompt: "What is your current status in the U.S.?",
                input_type: "choice",
                options: [
                    { value: "outside_us", label: "Currently outside the U.S.", next_step: "imm_contact_info" },
                    { value: "valid_status", label: "In the U.S. with valid status", next_step: "imm_contact_info" },
                    { value: "expired", label: "Status expired or expiring soon", next_step: "imm_contact_info" },
                    { value: "overstayed", label: "Overstayed visa", next_step: "imm_contact_info" }
                ]
            },
            
            // GREEN CARD PATH
            imm_green_card_type: {
                prompt: "How are you seeking your green card?",
                input_type: "choice",
                options: [
                    { value: "employment", label: "Through employment", next_step: "imm_green_card_stage" },
                    { value: "family", label: "Through family member", next_step: "imm_green_card_stage" },
                    { value: "marriage", label: "Through marriage to U.S. citizen", next_step: "imm_green_card_marriage" },
                    { value: "investor", label: "Investment visa (EB-5)", next_step: "imm_green_card_stage" },
                    { value: "other_gc", label: "Other category", next_step: "imm_green_card_stage" }
                ]
            },
            imm_green_card_marriage: {
                prompt: "How long have you been married?",
                input_type: "choice",
                options: [
                    { value: "under_2", label: "Less than 2 years", next_step: "imm_green_card_stage" },
                    { value: "over_2", label: "Over 2 years", next_step: "imm_green_card_stage" }
                ]
            },
            imm_green_card_stage: {
                prompt: "What stage are you at in the process?",
                input_type: "choice",
                options: [
                    { value: "not_started", label: "Haven't started yet", next_step: "imm_contact_info" },
                    { value: "petition_pending", label: "Petition pending", next_step: "imm_contact_info" },
                    { value: "interview_scheduled", label: "Interview scheduled", next_step: "imm_contact_info" },
                    { value: "denied", label: "Application denied", next_step: "imm_contact_info" }
                ]
            },
            
            // CITIZENSHIP PATH
            imm_citizenship_status: {
                prompt: "Are you currently a green card holder?",
                input_type: "choice",
                options: [
                    { value: "yes_5plus", label: "Yes, for 5+ years", next_step: "imm_citizenship_eligible" },
                    { value: "yes_3plus", label: "Yes, for 3+ years (married to citizen)", next_step: "imm_citizenship_eligible" },
                    { value: "yes_under3", label: "Yes, but less than 3 years", next_step: "imm_citizenship_wait" },
                    { value: "no", label: "No, not a green card holder", next_step: "imm_citizenship_no_gc" }
                ]
            },
            imm_citizenship_eligible: {
                prompt: "Have you traveled outside the U.S. for more than 6 months at a time in the past 5 years?",
                input_type: "choice",
                options: [
                    { value: "no", label: "No", next_step: "imm_contact_info" },
                    { value: "yes", label: "Yes", next_step: "imm_contact_info" }
                ]
            },
            imm_citizenship_wait: {
                prompt: "You may need to wait until you meet the 3 or 5 year requirement. However, let's discuss your situation to see if there are any special provisions that apply.",
                input_type: "none",
                next_step: "imm_contact_info"
            },
            imm_citizenship_no_gc: {
                prompt: "You'll need to obtain a green card before applying for citizenship. We can help you with that process.",
                input_type: "none",
                next_step: "imm_contact_info"
            },
            
            // DEPORTATION DEFENSE
            imm_deportation_urgent: {
                prompt: "Do you have an upcoming court date or removal order?",
                input_type: "choice",
                options: [
                    { value: "court_soon", label: "Court date within 30 days", next_step: "imm_deportation_emergency" },
                    { value: "court_later", label: "Court date more than 30 days away", next_step: "imm_contact_info" },
                    { value: "order_issued", label: "Removal order issued", next_step: "imm_deportation_emergency" },
                    { value: "detained", label: "Currently detained", next_step: "imm_deportation_emergency" },
                    { value: "no_court", label: "No court date yet", next_step: "imm_contact_info" }
                ]
            },
            imm_deportation_emergency: {
                prompt: "This requires immediate attention. Please call our office now at (555) 123-4567 for urgent consultation.\n\nWould you like to leave your contact information for our immigration attorney to call you back ASAP?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes, have attorney call me", next_step: "imm_contact_info" },
                    { value: "calling", label: "I'll call the office now", next_step: "imm_deportation_call" }
                ]
            },
            imm_deportation_call: {
                prompt: "Please call us immediately at (555) 123-4567.\n\nAsk for the immigration emergency line. We have attorneys available for urgent deportation defense matters.",
                input_type: "none"
            },
            
            // ASYLUM PATH
            imm_asylum_intro: {
                prompt: "Are you already in the United States?",
                input_type: "choice",
                options: [
                    { value: "yes_under_1yr", label: "Yes, arrived less than 1 year ago", next_step: "imm_asylum_filed" },
                    { value: "yes_over_1yr", label: "Yes, arrived over 1 year ago", next_step: "imm_asylum_filed" },
                    { value: "no", label: "No, outside the U.S.", next_step: "imm_asylum_outside" }
                ]
            },
            imm_asylum_filed: {
                prompt: "Have you filed for asylum yet?",
                input_type: "choice",
                options: [
                    { value: "not_filed", label: "No, not yet", next_step: "imm_contact_info" },
                    { value: "filed_pending", label: "Yes, application pending", next_step: "imm_contact_info" },
                    { value: "interview_scheduled", label: "Interview scheduled", next_step: "imm_contact_info" },
                    { value: "denied", label: "Application denied", next_step: "imm_contact_info" }
                ]
            },
            imm_asylum_outside: {
                prompt: "For asylum applications from outside the U.S., we recommend contacting the nearest U.S. embassy or consulting with our team about refugee resettlement programs.",
                input_type: "none",
                next_step: "imm_contact_info"
            },
            
            // FAMILY PETITION
            imm_family_petition: {
                prompt: "Who is petitioning for you (or who are you petitioning for)?",
                input_type: "choice",
                options: [
                    { value: "spouse", label: "Spouse", next_step: "imm_family_petitioner_status" },
                    { value: "parent", label: "Parent", next_step: "imm_family_petitioner_status" },
                    { value: "child", label: "Child", next_step: "imm_family_petitioner_status" },
                    { value: "sibling", label: "Sibling", next_step: "imm_family_petitioner_status" }
                ]
            },
            imm_family_petitioner_status: {
                prompt: "Is the petitioner a U.S. citizen or green card holder?",
                input_type: "choice",
                options: [
                    { value: "citizen", label: "U.S. Citizen", next_step: "imm_contact_info" },
                    { value: "green_card", label: "Green Card Holder", next_step: "imm_contact_info" },
                    { value: "neither", label: "Neither", next_step: "imm_family_not_eligible" }
                ]
            },
            imm_family_not_eligible: {
                prompt: "Unfortunately, only U.S. citizens and green card holders can petition for family members. Let's discuss other options that might be available.",
                input_type: "none",
                next_step: "imm_contact_info"
            },
            
            // OTHER IMMIGRATION
            imm_other_intro: {
                prompt: "Please briefly describe your immigration matter.",
                input_type: "text",
                next_step: "imm_contact_info"
            },
            
            // IMMIGRATION CONTACT INFO
            imm_contact_info: {
                prompt: `
                    <div class="contact-info-box" style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="font-size: 14px; color: #1e40af; font-weight: 600; margin-bottom: 6px;">📋 Contact Information</p>
                        <p style="font-size: 13px; color: #1f2937; margin-bottom: 8px; line-height: 1.4;">Please provide your contact details for a confidential immigration consultation. All information is protected by attorney-client privilege.</p>
                        <p style="font-size: 13px; color: #374151; margin-bottom: 6px;"><strong>Enter:</strong> Full name, phone number, email (separated by commas)</p>
                        <p style="font-size: 13px; color: #374151; background: white; padding: 6px 8px; border-radius: 4px; font-family: monospace; margin-bottom: 0;">
                            Maria Garcia, 555-123-4567, maria@email.com
                        </p>
                    </div>
                `,
                input_type: "text",
                next_step: "imm_consult"
            },
            imm_consult: {
                prompt: "Would you like to schedule a consultation with our immigration attorney?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes, schedule consultation", next_step: "imm_schedule" },
                    { value: "no", label: "No, just have someone call me", next_step: "imm_end" }
                ]
            },
            imm_schedule: {
                prompt: "What date and time works best for you? (e.g., 'Monday at 3pm' or 'Next week, morning')",
                input_type: "text",
                next_step: "imm_end"
            },
            imm_end: {
                prompt: "Thank you. Your information has been sent to our immigration law team. All consultations are confidential and protected.\n\nYou'll receive a response within one business day. For urgent matters, call (555) 123-4567.",
                input_type: "none"
            },
            
            // ============================================
            // CRIMINAL DEFENSE INTAKE FLOW
            // ============================================
            crim_intro: {
                prompt: "We understand this is a serious and stressful situation. Our criminal defense team is here to protect your rights.\n\nWhat type of criminal matter are you facing?",
                input_type: "choice",
                options: [
                    { value: "arrest", label: "Recently arrested / charged", next_step: "crim_arrest_when" },
                    { value: "investigation", label: "Under investigation", next_step: "crim_investigation_contacted" },
                    { value: "warrant", label: "Warrant issued", next_step: "crim_warrant_urgent" },
                    { value: "court_pending", label: "Court case pending", next_step: "crim_court_when" },
                    { value: "probation", label: "Probation / parole issue", next_step: "crim_probation_violation" },
                    { value: "expungement", label: "Record expungement / sealing", next_step: "crim_expungement_intro" },
                    { value: "other_crim", label: "Other criminal matter", next_step: "crim_other_intro" }
                ]
            },
            
            // ARREST PATH
            crim_arrest_when: {
                prompt: "When were you arrested or charged?",
                input_type: "choice",
                options: [
                    { value: "today", label: "Today or yesterday", next_step: "crim_arrest_charges" },
                    { value: "this_week", label: "Within the past week", next_step: "crim_arrest_charges" },
                    { value: "this_month", label: "Within the past month", next_step: "crim_arrest_charges" },
                    { value: "longer", label: "More than a month ago", next_step: "crim_arrest_charges" }
                ]
            },
            crim_arrest_charges: {
                prompt: "What are you charged with? (General category)",
                input_type: "choice",
                options: [
                    { value: "dui", label: "DUI / DWI", next_step: "crim_arrest_custody" },
                    { value: "drug", label: "Drug-related charges", next_step: "crim_arrest_custody" },
                    { value: "theft", label: "Theft / property crime", next_step: "crim_arrest_custody" },
                    { value: "assault", label: "Assault / violent crime", next_step: "crim_arrest_custody" },
                    { value: "domestic", label: "Domestic violence", next_step: "crim_arrest_custody" },
                    { value: "sex", label: "Sex crime", next_step: "crim_arrest_custody" },
                    { value: "white_collar", label: "White collar / fraud", next_step: "crim_arrest_custody" },
                    { value: "other_charge", label: "Other charges", next_step: "crim_arrest_custody" }
                ]
            },
            crim_arrest_custody: {
                prompt: "Are you currently in custody or have you been released?",
                input_type: "choice",
                options: [
                    { value: "in_custody", label: "Currently in custody", next_step: "crim_custody_urgent" },
                    { value: "released_bail", label: "Released on bail", next_step: "crim_arrest_attorney" },
                    { value: "released_own", label: "Released on own recognizance", next_step: "crim_arrest_attorney" }
                ]
            },
            crim_custody_urgent: {
                prompt: "If you're calling from jail, we recommend having a family member or friend contact us on your behalf.\n\nOur office number is (555) 123-4567. Ask for the criminal defense emergency line.\n\nWe can often arrange bail hearings within 24-48 hours.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            crim_arrest_attorney: {
                prompt: "Do you currently have an attorney?",
                input_type: "choice",
                options: [
                    { value: "no", label: "No", next_step: "crim_arrest_court_date" },
                    { value: "public_defender", label: "Public defender", next_step: "crim_arrest_court_date" },
                    { value: "yes", label: "Yes, private attorney", next_step: "crim_attorney_existing" }
                ]
            },
            crim_attorney_existing: {
                prompt: "If you already have a private attorney, we may not be able to take your case due to conflict of interest rules. However, you can contact us for a second opinion or if you're considering changing counsel.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            crim_arrest_court_date: {
                prompt: "Do you have a court date scheduled?",
                input_type: "choice",
                options: [
                    { value: "yes_soon", label: "Yes, within 2 weeks", next_step: "crim_court_date_urgent" },
                    { value: "yes_later", label: "Yes, more than 2 weeks away", next_step: "crim_contact_info" },
                    { value: "no", label: "No date yet", next_step: "crim_contact_info" },
                    { value: "unknown", label: "I don't know", next_step: "crim_contact_info" }
                ]
            },
            
            crim_court_date_urgent: {
                prompt: "With a court date coming up soon, time is critical. We should speak with you as soon as possible.\n\nCall us today at (555) 123-4567 or provide your contact info for an immediate callback.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // INVESTIGATION PATH
            crim_investigation_contacted: {
                prompt: "Have police or investigators contacted you?",
                input_type: "choice",
                options: [
                    { value: "contacted", label: "Yes, they've contacted me", next_step: "crim_investigation_when" },
                    { value: "suspect", label: "I believe I'm under investigation", next_step: "crim_investigation_rights" },
                    { value: "witnessed", label: "I was questioned as a witness", next_step: "crim_investigation_rights" }
                ]
            },
            crim_investigation_when: {
                prompt: "⚠️ IMPORTANT: Do not speak to police without an attorney present. You have the right to remain silent.\n\nHave you already spoken to investigators?",
                input_type: "choice",
                options: [
                    { value: "not_yet", label: "No, haven't spoken to them", next_step: "crim_investigation_rights" },
                    { value: "already_spoke", label: "Yes, I already spoke to them", next_step: "crim_investigation_rights" }
                ]
            },
            crim_investigation_rights: {
                prompt: "Remember: You have the right to remain silent. You have the right to an attorney. If you can't afford one, one will be appointed.\n\nNever speak to police without your attorney present, even if you think you're just a witness.\n\nLet's get you legal representation immediately.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // WARRANT PATH
            crim_warrant_urgent: {
                prompt: "If there's a warrant out for your arrest, it's critical to handle this properly to protect your rights and potentially avoid additional charges.\n\nDo you know what the warrant is for?",
                input_type: "choice",
                options: [
                    { value: "know", label: "Yes, I know", next_step: "crim_warrant_action" },
                    { value: "unsure", label: "Not sure / don't know", next_step: "crim_warrant_action" }
                ]
            },
            crim_warrant_action: {
                prompt: "We can help arrange a voluntary surrender with the court, which can help your case significantly compared to being arrested.\n\nCall our office immediately at (555) 123-4567 for emergency warrant assistance.\n\nDo NOT turn yourself in without legal representation.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // COURT PENDING PATH
            crim_court_when: {
                prompt: "When is your next court date?",
                input_type: "choice",
                options: [
                    { value: "within_week", label: "Within 1 week", next_step: "crim_court_represented" },
                    { value: "1_2_weeks", label: "1-2 weeks", next_step: "crim_court_represented" },
                    { value: "2_4_weeks", label: "2-4 weeks", next_step: "crim_court_represented" },
                    { value: "over_month", label: "Over a month away", next_step: "crim_court_represented" }
                ]
            },
            crim_court_represented: {
                prompt: "Do you currently have legal representation?",
                input_type: "choice",
                options: [
                    { value: "no", label: "No attorney yet", next_step: "crim_contact_info" },
                    { value: "public_defender", label: "Public defender", next_step: "crim_contact_info" },
                    { value: "unhappy", label: "Yes, but unhappy with current attorney", next_step: "crim_change_attorney" }
                ]
            },
            crim_change_attorney: {
                prompt: "You have the right to change attorneys, but timing matters. We should discuss this urgently.\n\nPlease provide your contact information so we can review your case immediately.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // PROBATION PATH
            crim_probation_violation: {
                prompt: "What is your probation/parole situation?",
                input_type: "choice",
                options: [
                    { value: "violation_alleged", label: "Accused of violation", next_step: "crim_probation_hearing" },
                    { value: "technical", label: "Missed appointment / technical violation", next_step: "crim_probation_hearing" },
                    { value: "new_charges", label: "New charges while on probation", next_step: "crim_probation_urgent" },
                    { value: "warrant_probation", label: "Warrant issued", next_step: "crim_probation_urgent" }
                ]
            },
            crim_probation_hearing: {
                prompt: "Do you have a probation violation hearing scheduled?",
                input_type: "choice",
                options: [
                    { value: "yes", label: "Yes", next_step: "crim_contact_info" },
                    { value: "no", label: "No", next_step: "crim_contact_info" },
                    { value: "dont_know", label: "I don't know", next_step: "crim_contact_info" }
                ]
            },
            crim_probation_urgent: {
                prompt: "This is serious. Probation violations with new charges can result in additional jail time.\n\nCall us immediately at (555) 123-4567 for emergency assistance.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // EXPUNGEMENT PATH
            crim_expungement_intro: {
                prompt: "We can help clear your criminal record in many cases.\n\nHow long ago was your conviction?",
                input_type: "choice",
                options: [
                    { value: "under_1yr", label: "Less than 1 year", next_step: "crim_expungement_waiting" },
                    { value: "1_3_yrs", label: "1-3 years ago", next_step: "crim_expungement_eligible" },
                    { value: "over_3yrs", label: "Over 3 years ago", next_step: "crim_expungement_eligible" },
                    { value: "no_conviction", label: "Case dismissed / no conviction", next_step: "crim_expungement_eligible" }
                ]
            },
            crim_expungement_waiting: {
                prompt: "Most jurisdictions require a waiting period before expungement is available. However, let's review your case to see if you qualify for any immediate relief.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            crim_expungement_eligible: {
                prompt: "You may be eligible for expungement or record sealing. This can help with:\n\n• Employment opportunities\n• Housing applications\n• Professional licenses\n• Background checks\n\nLet's review your case and eligibility.",
                input_type: "none",
                next_step: "crim_contact_info"
            },
            
            // OTHER CRIMINAL
            crim_other_intro: {
                prompt: "Please briefly describe your criminal law matter.",
                input_type: "text",
                next_step: "crim_contact_info"
            },
            
            // CRIMINAL CONTACT INFO
            crim_contact_info: {
                prompt: `
                    <div class="contact-info-box" style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px; border-radius: 8px; margin: 8px 0;">
                        <p style="font-size: 14px; color: #92400e; font-weight: 600; margin-bottom: 6px;">⚖️ Confidential Legal Consultation</p>
                        <p style="font-size: 13px; color: #1f2937; margin-bottom: 8px; line-height: 1.4;">All communications are protected by attorney-client privilege. Your information is completely confidential.</p>
                        <p style="font-size: 13px; color: #374151; margin-bottom: 6px;"><strong>Enter:</strong> Full name, phone number, email (separated by commas)</p>
                        <p style="font-size: 13px; color: #374151; background: white; padding: 6px 8px; border-radius: 4px; font-family: monospace; margin-bottom: 0;">
                            Robert Johnson, 555-123-4567, robert@email.com
                        </p>
                    </div>
                `,
                input_type: "text",
                next_step: "crim_consult"
            },
            crim_consult: {
                prompt: "Would you like to schedule an urgent consultation with our criminal defense attorney?",
                input_type: "choice",
                options: [
                    { value: "yes_urgent", label: "Yes - URGENT (call me today)", next_step: "crim_urgent_schedule" },
                    { value: "yes", label: "Yes - schedule consultation", next_step: "crim_schedule" },
                    { value: "no", label: "No, just have someone call me", next_step: "crim_end" }
                ]
            },
            crim_urgent_schedule: {
                prompt: "Understood. Our criminal defense team will call you within 2 hours during business hours, or first thing the next business day.\n\nFor immediate assistance, call (555) 123-4567.",
                input_type: "none",
                next_step: "crim_end"
            },
            crim_schedule: {
                prompt: "What date and time works best? (e.g., 'Tomorrow afternoon' or 'This Friday at 10am')",
                input_type: "text",
                next_step: "crim_end"
            },
            crim_end: {
                prompt: "Your information has been sent to our criminal defense team. All communications are confidential and protected.\n\n⚠️ REMEMBER: Do not speak to police without an attorney present.\n\nYou'll hear from us within 4 business hours. For urgent matters, call (555) 123-4567 immediately.",
                input_type: "none"
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

        // ==============================================================================
        // INITIALIZE STRIPE ON PAGE LOAD
        // ==============================================================================
        
        async function initializeStripe() {
            try {
                const response = await fetch(`${API_URL}/api/payments/stripe-config`);
                const data = await response.json();
                
                if (data.success && data.publishable_key) {
                    stripe = Stripe(data.publishable_key);
                    console.log('✅ Stripe initialized');
                    return true;
                }
            } catch (error) {
                console.error('❌ Error initializing Stripe:', error);
                return false;
            }
        }
        function openChatFromCTA() {
            const chatWindow = getEl('lawfirm-chat-window');
            if (!chatWindow) return;
            if (!chatWindow.classList.contains('open')) {
                chatWindow.classList.add('open');
            }
            const input = getEl('chat-input');
            if (input) input.focus();
        }


        // Initialize on page load - UPDATED VERSION
        

        // (DOMContentLoaded init moved to multi-style loader)

        function initializeWidget(wid) {
          const state = widgetStates[wid];
          if (!state) return;

          // Reset to start
          state.currentStep = 'start';
          state.collectedData = {};

          const messagesDiv = document.getElementById(`${wid}-chat-messages`);
          console.log('messagesDiv found:', messagesDiv);

          if (!messagesDiv) {
            console.error(`Could not find messages div for ${wid}`);
            return;
          }

          // Clear any existing messages FIRST
          messagesDiv.innerHTML = '';
              
          // THEN add welcome message
          const welcomeMsg = `Hello and welcome to our law firm. I can help you with a free case evaluation, scheduling, or questions about our services.\n\nWhat type of legal issue do you need help with?`;
              
          const messageDiv = document.createElement('div');
          messageDiv.className = 'chat-message bot';
          messageDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="message-content">${welcomeMsg.replace(/\n/g, '<br>')}</div>
          `;
          messagesDiv.appendChild(messageDiv);
          messagesDiv.scrollTop = messagesDiv.scrollHeight;

          // Add welcome message directly
          const welcomeStep = flowSteps['start'];
          if (welcomeStep && welcomeStep.prompt) {
            // Create bot message
            const messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message bot';

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '🤖';

            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = welcomeStep.prompt;

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            messagesDiv.appendChild(messageDiv);

            // Add quick action buttons if available
            if (welcomeStep.options) {
              const optionsContainer = document.createElement('div');
              optionsContainer.className = 'chat-options';
              optionsContainer.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;';
            
              welcomeStep.options.forEach(option => {
                const button = document.createElement('button');
                button.className = 'quick-action-btn';
                button.textContent = option.label;
                button.type = 'button';
                button.onclick = () => {
                  // Remove options
                  optionsContainer.remove();
                  // Add user message
                  const userMsg = document.createElement('div');
                  userMsg.className = 'chat-message user';
                  const userAvatar = document.createElement('div');
                  userAvatar.className = 'message-avatar';
                  userAvatar.textContent = '👤';
                  const userContent = document.createElement('div');
                  userContent.className = 'message-content';
                  userContent.textContent = option.label;
                  userMsg.appendChild(userAvatar);
                  userMsg.appendChild(userContent);
                  messagesDiv.appendChild(userMsg);
                
                  // Save data and continue
                  state.collectedData[state.currentStep] = option.value;
                  if (option.next_step) {
                    state.currentStep = option.next_step;
                    // Continue flow here if needed
                  }
                };
                optionsContainer.appendChild(button);
              });

              messagesDiv.appendChild(optionsContainer);
            }

            messagesDiv.scrollTop = messagesDiv.scrollHeight;
          }
        }

        function generateSessionId() {
            return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        }

        function toggleChat() {
            const chatWindow = getEl('lawfirm-chat-window');
            chatWindow.classList.toggle('open');
            if (chatWindow.classList.contains('open')) {
                getEl('chat-input').focus();
                getEl('chat-notification').style.display = 'none';
            }
        }

        function addMessage(sender, text) {
            const messagesContainer = getEl('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${sender}`;

            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = sender === 'user' ? '👤' : '🤖';

            const content = document.createElement('div');
            content.className = 'message-content';
            if (sender === 'bot' && (text.includes('<') || text.includes('>'))) {
                content.innerHTML = text;
            } else {
                content.textContent = text;
            }

            const time = document.createElement('div');
            time.className = 'message-time';
            time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            const body = document.createElement('div');
            body.appendChild(content);
            body.appendChild(time);

            messageDiv.appendChild(avatar);
            messageDiv.appendChild(body);

            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function showOptions(options) {
        const messagesContainer = getEl('chat-messages');
        
        const optionsContainer = document.createElement('div');
        optionsContainer.className = 'chat-options';
        optionsContainer.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;';
        
        options.forEach(option => {
            const button = document.createElement('button');
            button.className = 'quick-action-btn';
            button.textContent = option.label;
            button.type = 'button';
            button.onclick = () => handleOptionClick(option.value, option.label, option.next_step);
            optionsContainer.appendChild(button);
        });
        
        messagesContainer.appendChild(optionsContainer);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function handleOptionClick(value, label, nextStep) {
            const state = widgetStates[ACTIVE_WID];
            
            // Remove all option buttons
            const messagesContainer = getEl('chat-messages');
            const optionDivs = messagesContainer.querySelectorAll('.chat-options');
            optionDivs.forEach(div => div.remove());
            
            // Add user's choice as message
            addMessage('user', label);
            
            // Save the data
            state.collectedData[state.currentStep] = value;
            
            // Move to next step
            if (nextStep) {
                state.currentStep = nextStep;
                addScriptStep(nextStep);
            }
        }
        
        function addScriptStep(stepId) {
            const step = flowSteps[stepId];
            if (!step) return;
            currentStep = stepId;

            addMessage('bot', step.prompt);

            if (stepId === 'payment_flow') {
                setTimeout(() => {
                    startPaymentFlow();
                }, 500);
                return;
            }

            if (step.options && step.options.length) {
                const messagesContainer = getEl('chat-messages');
                const btnGroup = document.createElement('div');
                btnGroup.className = 'script-button-group';

                step.options.forEach(opt => {
                    const btn = document.createElement('button');
                    btn.className = 'script-button';
                    btn.textContent = opt.label;
                    btn.onclick = () => handleOptionClick(stepId, opt);
                    btnGroup.appendChild(btn);
                });

                messagesContainer.appendChild(btnGroup);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            // ✅ AUTO-ADVANCE FOR "NONE" STEPS - INSIDE THE FUNCTION!
            if (step.input_type === 'none' && step.next_step) {
                console.log('🔄 Auto-advancing from', stepId, 'to', step.next_step);
                setTimeout(() => {
                    addScriptStep(step.next_step);
                }, 1500);
                return;
            }
            
            // ✅ FINAL STEP - ASK "ANYTHING ELSE?"
            if (step.input_type === 'none' && !step.next_step) {
                console.log('✅ Final step reached:', stepId);
                setTimeout(() => {
                    addMessage('bot', 'Is there anything else I can assist you with today?');
                    intakeActive = false;
                    awaitingFarewellResponse = true;
                }, 2000);
            }
        }  // ← FUNCTION ENDS HERE - AFTER THE AUTO-ADVANCE CODE!
    
        function handleOptionClick(stepId, option) {
            document.querySelectorAll('.script-button-group').forEach(el => el.remove());
            addMessage('user', option.label);
            collectedData[stepId] = option.value;

            const nextId = option.next_step || flowSteps[stepId].next_step;
            
            if (nextId === 'payment_flow') {
                addScriptStep(nextId);
                return;
            }
            
            if (nextId) {
                addScriptStep(nextId);
            } else {
                intakeActive = false;
                addMessage('bot', "Thank you. Your information has been received.");
            }
        }

        function quickAction(action) {
            if (action === 'make_payment') {
                startPaymentFlow();
                return;
            }
            // Map quick actions to flow steps
            const actionMap = {
                'personal_injury': 'pi_intro',
                'family_law': 'family_intro',
                'immigration': 'imm_intro',
                'criminal_defense': 'crim_intro',
                'schedule': 'pi_consult'  // Or create a general scheduling step
            };
            
            const stepId = actionMap[action];
            
            if (stepId) {
                // Add user message showing what they clicked
                const labels = {
                    'personal_injury': 'Personal Injury',
                    'family_law': 'Family Law',
                    'immigration': 'Immigration',
                    'criminal_defense': 'Criminal Defense',
                    'schedule': 'Schedule Consultation'
                };
                
                addMessage('user', labels[action]);
                
                // Navigate to the appropriate flow
                intakeActive = true;
                addScriptStep(stepId);
            }
        }

        // PAYMENT FLOW FUNCTIONS
        function startPaymentFlow() {
            paymentFlowActive = true;
            paymentClientData = {};
            
            addMessage('user', "I'd like to make a payment.");
            
            const referenceId = 'PAY-' + Date.now().toString(36).toUpperCase() + '-' + Math.random().toString(36).substr(2, 4).toUpperCase();
            collectedData['payment_reference'] = referenceId;
            
            const infoMessage = `
                <div class="payment-info-box" style="background: #dbeafe; border-left: 4px solid #3b82f6; padding: 16px; border-radius: 8px; margin: 8px 0;">
                    <p style="font-size: 14px; color: #1e40af; font-weight: 600; margin-bottom: 8px;">💳 Payment Request</p>
                    <p style="font-size: 13px; color: #1f2937; margin-bottom: 12px;">Please share your contact details so our team can process your payment.</p>
                    <p style="font-size: 13px; color: #374151; margin-bottom: 8px;"><strong>Enter:</strong> Full name, email (separated by commas)</p>
                    <p style="font-size: 13px; color: #374151; background: white; padding: 8px; border-radius: 4px; font-family: monospace;">
                        John Smith, john@email.com
                    </p>
                    <p style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 11px;">
                        <strong>Reference:</strong> ${referenceId}
                    </p>
                </div>
            `;
            
            addMessage('bot', infoMessage);
            getEl('chat-input').focus();
        }

        function parsePaymentInfo(message) {
            const parts = message.split(',').map(p => p.trim());
            
            if (parts.length !== 2) {
                return null;
            }
            
            const nameParts = parts[0].split(' ').filter(n => n.length > 0);
            if (nameParts.length < 2) {
                return null;
            }
            
            const firstName = nameParts[0];
            const lastName = nameParts.slice(1).join(' ');
            
            return {
                first_name: firstName,
                last_name: lastName,
                full_name: parts[0],
                email: parts[1]
            };
        }

        async function verifyAndShowPayment(clientInfo) {
            showTyping(true);
            
            try {
                const response = await fetch(`${API_URL}/api/payments/client-payment-request`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                    body: new URLSearchParams({
                        first_name: clientInfo.first_name,
                        last_name: clientInfo.last_name,
                        email: clientInfo.email,
                        payment_type: 'retainer',
                        reference_id: collectedData['payment_reference'] || 'N/A'
                    })
                });
                
                const data = await response.json();
                showTyping(false);
                
                if (data.client_found) {
                    paymentClientData = data.client;
                    
                    const paymentMessage = `
                        <div class="payment-info-box" style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; border-radius: 8px; margin: 8px 0;">
                            <p style="font-size: 14px; color: #065f46; font-weight: 600; margin-bottom: 8px;">✓ Client Verified</p>
                            <p style="font-size: 13px; color: #1f2937; margin-bottom: 12px;">${data.message}</p>
                            <div style="background: white; padding: 12px; border-radius: 6px; margin: 10px 0;">
                                <p style="font-size: 13px; color: #374151;"><strong>Amount:</strong> ${data.amount.toFixed(2)}</p>
                                <p style="font-size: 13px; color: #374151;"><strong>Payment Type:</strong> ${data.payment_type.replace('_', ' ')}</p>
                                <p style="font-size: 12px; color: #6b7280; margin-top: 8px;"><strong>Reference:</strong> ${collectedData['payment_reference']}</p>
                            </div>
                        </div>
                        <div class="payment-button-container" style="display: flex; gap: 10px; margin-top: 12px;">
                            <button class="payment-method-btn stripe-btn" onclick="processPayment('stripe', '${data.client.id}', ${data.amount}, '${data.description}')" style="flex: 1; padding: 12px 20px; background: #635bff; color: white; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px;">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M13.5 2C9.9 2 7 4.9 7 8.5S9.9 15 13.5 15 20 12.1 20 8.5 17.1 2 13.5 2zm0 11c-2.5 0-4.5-2-4.5-4.5S11 4 13.5 4 18 6 18 8.5 16 13 13.5 13z"/>
                                </svg>
                                Pay with Stripe
                            </button>
                            <button class="payment-method-btn paypal-btn" onclick="processPayment('paypal', '${data.client.id}', ${data.amount}, '${data.description}')">
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M19.5 6.5c.5 1.1.5 2.4 0 3.8-.8 2.5-2.8 4.2-5.4 4.2H12l-.8 4.5h-2l.8-4.5H8.5L7 20H5l2-11h7.1c1.8 0 3.3.6 4.2 1.7.5.5.8 1.1 1.2 1.8z"/>
                                </svg>
                                Pay with PayPal
                            </button>
                        </div>
                        
                    `;
                    
                    addMessage('bot', paymentMessage);
                    paymentFlowActive = false;
                    
                } else {
                    const notFoundMessage = `
                        <div class="payment-info-box" style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 16px; border-radius: 8px; margin: 8px 0;">
                            <p style="font-size: 14px; color: #92400e; font-weight: 600; margin-bottom: 8px;">⚠️ Client Record Not Found</p>
                            <p style="font-size: 13px; color: #1f2937;">I couldn't find a matching client record in our system.</p>
                            <p style="font-size: 13px; color: #374151; margin-top: 8px;">Are you a returning client?</p>
                        </div>
                    `;
                    
                    addMessage('bot', notFoundMessage);
                    
                    const returningClientButtons = `
                        <div class="script-button-group">
                            <button class="script-button" onclick="handleReturningClientResponse('yes')">Yes, I'm a returning client</button>
                            <button class="script-button" onclick="handleReturningClientResponse('no')">No, I'm a new client</button>
                        </div>
                    `;
                    
                    addMessage('bot', returningClientButtons);
                }
                
            } catch (error) {
                showTyping(false);
                console.error('Payment verification error:', error);
                addMessage('bot', "I'm having trouble verifying your information. Please try again or contact our office.");
                paymentFlowActive = false;
            }
        }

        function handleReturningClientResponse(answer) {
            document.querySelectorAll('.script-button-group').forEach(el => el.remove());
            
            addMessage('user', answer === 'yes' ? "Yes, I'm a returning client" : "No, I'm a new client");
            
            handOffToLiveAgent();
            paymentFlowActive = false;
        }

        function handOffToLiveAgent() {
            const handOffMessage = `
                <div class="payment-info-box" style="background: #fef3c7; border-color: #f59e0b;">
                    <p><strong>🙋 Connecting you with a live agent...</strong></p>
                    <p style="margin-top: 10px;">A member of our team will assist you shortly. Please stay on the line.</p>
                    <p style="margin-top: 10px; font-size: 13px;">Or call us directly at: <strong>(555) 123-4567</strong></p>
                </div>
            `;
            
            addMessage('bot', handOffMessage);
            notifyTeamOfHandoff();
        }

        async function notifyTeamOfHandoff() {
            try {
                await fetch(`${API_URL}/api/notifications/handoff`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        reason: 'payment_client_not_found',
                        client_info: paymentClientData,
                        timestamp: new Date().toISOString()
                    })
                });
            } catch (error) {
                console.error('Failed to notify team:', error);
            }
        }

        // ✅ UPDATED PROCESS PAYMENT - SAVES INFO BEFORE REDIRECT
        async function processPayment(provider, clientId, amount, description) {
            if (provider === 'stripe') {
                try {
                    if (!stripe) {
                        addMessage('bot', '❌ Payment system not initialized. Please refresh the page.');
                        return;
                    }
                    
                    showTyping(true);
                    
                    let currentUrl = window.location.href.split('?')[0].split('#')[0];
                    currentUrl = currentUrl.replace(/\/$/, '');
                    
                    console.log('📍 Will return to:', currentUrl);
                    
                    const response = await fetch(`${API_URL}/api/payments/create-stripe-link`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            client_id: clientId,
                            amount: amount,
                            description: description,
                            payment_type: 'retainer',
                            reference_id: collectedData['payment_reference'] || 'N/A',
                            return_url: currentUrl
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}`);
                    }
                    
                    const data = await response.json();
                    showTyping(false);
                    
                    if (data.success && data.payment_url) {
                        console.log('✅ Redirecting to Stripe checkout...');
                        
                        // ✅ SAVE SESSION INFO BEFORE REDIRECT
                        localStorage.setItem('payment_in_progress', 'true');
                        localStorage.setItem('payment_client_name', paymentClientData.full_name || paymentClientData.name || 'Valued Client');
                        localStorage.setItem('payment_amount', amount.toString());
                        
                        addMessage('bot', '🔄 Redirecting to secure payment page...');
                        
                        setTimeout(() => {
                            window.location.href = data.payment_url;
                        }, 500);
                    } else {
                        addMessage('bot', '❌ Sorry, there was an issue creating the payment.');
                    }
                    
                } catch (error) {
                    console.error('❌ Payment error:', error);
                    showTyping(false);
                    addMessage('bot', '❌ Payment error. Please try again or call our office.');
                }
            }
        }

        // Notify team of handoff
        async function notifyTeamOfHandoff() {
            try {
                await fetch(`${API_URL}/api/notifications/handoff`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        session_id: sessionId,
                        reason: 'payment_client_not_found',
                        client_info: paymentClientData,
                        timestamp: new Date().toISOString()
                    })
                });
            } catch (error) {
                console.error('Failed to notify team:', error);
            }
        }

        // Monitor payment status
        async function monitorPaymentStatus(paymentId) {
            const checkInterval = setInterval(async () => {
                try {
                    const response = await fetch(`${API_URL}/api/payments/${paymentId}/status`);
                    const data = await response.json();
                    
                    if (data.status === 'completed' || data.status === 'succeeded') {
                        clearInterval(checkInterval);
                        addMessage('bot', '🎉 Payment received! Thank you. Your transaction is complete.');
                    } else if (data.status === 'failed' || data.status === 'canceled') {
                        clearInterval(checkInterval);
                        addMessage('bot', '❌ Payment was not completed. If you need assistance, please contact our office.');
                    }
                } catch (error) {
                    console.error('Error checking payment status:', error);
                }
            }, 5000); // Check every 5 seconds
            
            // Stop checking after 5 minutes
            setTimeout(() => clearInterval(checkInterval), 300000);
        }

        // Modify the existing sendMessage function to handle payment flow
        async function sendMessage() {
            const input = getEl('chat-input');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            addMessage('user', message);

            // ✅ SMART FAREWELL DETECTION
            const farewellWords = ['goodbye', 'bye', 'thanks bye', 'thank you bye', 'see you', 'have a good day'];
            const messageLower = message.toLowerCase().trim();

            // Check if this is an actual farewell
            const isActualFarewell = farewellWords.some(word => messageLower.includes(word));

            console.log('🔍 awaitingFarewellResponse:', awaitingFarewellResponse);
            console.log('🔍 isActualFarewell:', isActualFarewell);
            console.log('🔍 messageLower:', messageLower);

            // Trigger farewell if:
            // 1. User explicitly says goodbye/bye, OR
            // 2. User says "no" and we're awaiting a response to "anything else?"
            if (isActualFarewell || (messageLower === 'no' && awaitingFarewellResponse)) {
                console.log('✅ FAREWELL TRIGGERED!');
                
                // Reset flag
                awaitingFarewellResponse = false;
                
                const goodbyeMessage = `
                    <div style="padding: 16px; text-align: center;">
                        <p style="font-size: 16px; margin-bottom: 8px;">👋 Thank you for contacting <strong>${typeof LAW_FIRM_NAME !== 'undefined' ? LAW_FIRM_NAME : 'us'}</strong>!</p>
                        <p style="color: #6b7280; font-size: 14px; margin-bottom: 6px; line-height: 1.4;">
                            We appreciate your time today. If you need anything else, we're here 24/7.
                        </p>
                        <p style="color: #6b7280; font-size: 14px; margin-bottom: 8px; line-height: 1.4;">
                            📞 ${typeof LAW_FIRM_PHONE !== 'undefined' ? LAW_FIRM_PHONE : '(555) 123-4567'} | 📧 ${typeof LAW_FIRM_EMAIL !== 'undefined' ? LAW_FIRM_EMAIL : 'contact@lawfirm.com'}
                        </p>
                        <p style="margin-top: 8px; margin-bottom: 0; font-size: 14px; color: #059669;">
                            Have a great day! ✨
                        </p>
                    </div>
                `;
                
                setTimeout(() => {
                    addMessage('bot', goodbyeMessage);
                }, 500);
                
                return; // Don't send to backend
            }


            // Check if in payment flow
            if (paymentFlowActive) {
                const clientInfo = parsePaymentInfo(message);
                
                if (clientInfo) {
                    await verifyAndShowPayment(clientInfo);
                } else {
                    const errorMessage = `
                        <div class="payment-info-box" style="background: #fee2e2; border-left: 4px solid #ef4444; padding: 16px; border-radius: 8px; margin: 8px 0;">
                            <p style="font-size: 13px; color: #991b1b; font-weight: 600;">❌ Incorrect Format</p>
                            <p style="font-size: 13px; color: #1f2937; margin-top: 8px;">Please provide your information in the correct format:</p>
                            <p style="font-size: 13px; color: #374151; background: white; padding: 8px; border-radius: 4px; font-family: monospace; margin-top: 8px;">
                                Full Name, Email
                            </p>
                            <p style="margin-top: 8px; color: #6b7280; font-size: 12px;">
                                <strong>Example:</strong> John Smith, john@email.com
                            </p>
                        </div>
                    `;
                    addMessage('bot', errorMessage);
                }
                return;
            }
            // ✅ COMPREHENSIVE DEBUG LOGGING
            console.log('=== SEND MESSAGE DEBUG ===');
            console.log('Message:', message);
            console.log('intakeActive:', intakeActive);
            console.log('currentStep:', currentStep);
            console.log('flowSteps[currentStep]:', flowSteps[currentStep]);
            
            // If we're in a scripted TEXT step, use it as the answer
            if (intakeActive && flowSteps[currentStep] && flowSteps[currentStep].input_type === 'text') {
                console.log('✅ IN TEXT INPUT MODE');
                console.log('Saving to collectedData[' + currentStep + ']');
                
                collectedData[currentStep] = message;
                const nextId = flowSteps[currentStep].next_step;
                
                console.log('Next step ID:', nextId);
                
                if (nextId) {
                    console.log('✅ Moving to next step:', nextId);
                    addScriptStep(nextId);
                } else {
                    console.log('⚠️ No next step - ending intake');
                    intakeActive = false;
                    addMessage('bot', "Thank you. Your information has been received.");
                }
                return;
            }
            
            console.log('❌ NOT in text input mode - proceeding to backend');

            // Otherwise: GENERAL CHAT
            showTyping(true);
            try {
                const data = await sendToBackend(message);
                showTyping(false);
                if (data && data.response) {
                    addMessage('bot', data.response);
                } else {
                    addMessage('bot', "I'm having trouble connecting right now. Please call our office at (555) 123-4567.");
                }
            } catch (err) {
                console.error(err);
                showTyping(false);
                addMessage('bot', "I'm having trouble connecting right now. Please call our office at (555) 123-4567.");
            }
        }

        async function sendToBackend(message) {
            const response = await fetch(`${API_URL}/api/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId,
                    client_id: clientId
                })
            });
            return response.json();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') sendMessage();
        }

        function showTyping(show) {
            const indicator = getEl('typing-indicator');
            if (show) indicator.classList.add('active');
            else indicator.classList.remove('active');
        }

        // --- FILE UPLOAD ---
        async function handleFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;

            // Validate file size (10MB max)
            const maxSize = 10 * 1024 * 1024;
            if (file.size > maxSize) {
                addUploadStatus('error', `File too large. Maximum size is 10MB.`);
                event.target.value = '';
                return;
            }

            // Show uploading status
            const statusId = addUploadStatus('uploading', `Uploading: ${file.name}`);

            try {
                const formData = new FormData();
                formData.append('file', file);
                formData.append('session_id', sessionId);
                formData.append('client_id', clientId || 'guest');
                formData.append('case_id', 'intake_' + sessionId);

                const response = await fetch(`${API_URL}/api/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // Remove uploading status
                removeUploadStatus(statusId);
                
                if (data.success) {
                    addUploadStatus('success', `✓ ${file.name} uploaded successfully`);
                    addMessage('bot', 'Thank you for uploading the document. I\'ve saved it to your case file.');
                } else {
                    addUploadStatus('error', `✗ Upload failed: ${data.error || 'Unknown error'}`);
                }
            } catch (error) {
                console.error('Upload error:', error);
                removeUploadStatus(statusId);
                addUploadStatus('error', `✗ Upload failed. Please check your connection and try again.`);
            }

            event.target.value = '';
        }

        function addUploadStatus(type, message) {
            const messagesContainer = getEl('chat-messages');
            const statusDiv = document.createElement('div');
            const statusId = 'status_' + Date.now();
            statusDiv.id = statusId;
            statusDiv.className = `upload-status ${type}`;
            
            const icon = type === 'success' ? '✓' : type === 'error' ? '✗' : '⏳';
            statusDiv.innerHTML = `<span>${icon}</span><span>${message}</span>`;
            
            messagesContainer.appendChild(statusDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            return statusId;
        }

        function removeUploadStatus(statusId) {
            const element = document.getElementById(statusId);
            if (element) {
                element.remove();
            }
        }

        // --- APPOINTMENT SCHEDULING ---
        async function scheduleAppointment() {
            showTyping(true);
            
            try {
                // Extract collected data
                const contactInfo = collectedData['pi_contact_info'] || '';
                const preferredTime = collectedData['pi_schedule'] || '';
                
                // More flexible parsing - handle single line or multi-line input
                let name = 'Guest User';
                let phone = null;
                let email = 'no-email@example.com';
                
                // Regex patterns
                const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;
                const phoneRegex = /(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/;
                
                // Extract email
                const emailMatch = contactInfo.match(emailRegex);
                if (emailMatch) {
                    email = emailMatch[0];
                }
                
                // Extract phone
                const phoneMatch = contactInfo.match(phoneRegex);
                if (phoneMatch) {
                    phone = phoneMatch[0].trim();
                }
                
                // Extract name - remove email and phone from the string, then get what's left
                let nameText = contactInfo
                    .replace(emailRegex, '')
                    .replace(phoneRegex, '')
                    .replace(/[,;]/g, ' ')  // Replace commas/semicolons with spaces
                    .trim()
                    .split(/\s+/)
                    .filter(word => word.length > 0)
                    .join(' ');
                
                if (nameText && nameText.length > 2) {
                    name = nameText;
                }
                
                // Format notes as human-readable text
                const formattedNotes = `Intake Completed: ${new Date().toLocaleString()}

Case Type: ${(collectedData['start'] || 'personal_injury').replace('_', ' ').toUpperCase()}

Incident Details:
- Date/Time: ${collectedData['pi_intro'] || 'Not specified'}
- Injury Type: ${collectedData['pi_injury_type'] || 'Not specified'}
- Injury Details: ${collectedData['pi_injury_details'] || 'N/A'}
- Medical Treatment: ${collectedData['pi_medical_treatment'] || 'Not specified'}

Legal Status:
- Currently Has Attorney: ${collectedData['pi_has_attorney'] || 'Not specified'}

Documents:
- Documents Submitted: ${collectedData['pi_docs'] || 'Not specified'}

Consultation Preference:
- Type: ${collectedData['pi_consult'] || 'Not specified'}
- Preferred Time: ${preferredTime}`;
                
                const response = await fetch(`${API_URL}/api/appointments/schedule`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        client_name: name,
                        client_email: email,
                        client_phone: phone,
                        preferred_date: preferredTime,
                        case_type: (collectedData['start'] || 'Personal Injury').replace('_', ' '),
                        notes: formattedNotes
                    })
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    console.error('Schedule error:', response.status, errorData);
                    throw new Error(`HTTP ${response.status}: ${JSON.stringify(errorData)}`);
                }
                
                const data = await response.json();
                showTyping(false);
                
                if (data.success) {
                    if (data.calendar_link) {
                        // Calendly link available
                        addMessage('bot', `✓ Perfect! I've found availability for you.\n\nClick the link below to confirm your consultation time:\n\n🔗 ${data.calendar_link}\n\nYou can choose from available time slots and you'll receive a confirmation email immediately.`);
                    } else {
                        // Manual scheduling fallback
                        addMessage('bot', `✓ Thank you! I've received your request for ${preferredTime}.\n\nOur scheduling team will review your preferred time and confirm availability within 2 hours. You'll receive:\n\n📧 Confirmation email\n📱 Text message with your appointment details\n\nIf ${preferredTime} isn't available, we'll suggest the closest alternative times.`);
                    }
                    
                    // Add final message
                    setTimeout(() => {
                        addMessage('bot', `Your case intake is complete! Reference ID: ${data.appointment_id.substring(0, 8).toUpperCase()}\n\nWhat happens next:\n1. You'll receive confirmation within 2 hours\n2. We'll review your case details\n3. A lawyer will be prepared for your consultation\n\nUrgent? Call us now at (555) 123-4567`);
                    }, 500);
                } else {
                    addMessage('bot', `I had a small technical issue, but don't worry - I've saved all your information!\n\nOur team will contact you at ${phone || email} within 2 hours to confirm your consultation for ${preferredTime}.\n\nNeed immediate assistance? Call (555) 123-4567`);
                }
            } catch (error) {
                console.error('Scheduling error:', error);
                showTyping(false);
                
                const preferredTime = collectedData['pi_schedule'] || 'your preferred time';
                const phone = collectedData['pi_contact_info']?.split('\n')[1] || 'the number you provided';
                
                addMessage('bot', `I've saved all your information! Our scheduling team will contact you at ${phone} within 2 hours to confirm your consultation.\n\nPreferred time: ${preferredTime}\n\nFor immediate assistance, call (555) 123-4567`);
            }
        }

function placeChatWindow(wid) {
  // Center-modal widget uses fixed positioning; skip
  if (wid === "w5") return;

  const btn = document.getElementById(`${wid}-lawfirm-chat-button`);
  const win = document.getElementById(`${wid}-lawfirm-chat-window`);
  if (!btn || !win) return;

  const rect = btn.getBoundingClientRect();
  const vh = window.innerHeight;

  const spaceAbove = rect.top - 16;
  const spaceBelow = vh - rect.bottom - 16;

  // Top widgets (w1 picture, w4 video) always open UPWARD —
  // cap height to whatever space is available above so it never clips.
  const isTopWidget = (wid === "w1" || wid === "w4");

  // If top widget doesn't have enough room above, drop to bottom of viewport
  if (isTopWidget && spaceAbove < 420) {
      win.style.position = "fixed";
      win.style.bottom = "90px";
      win.style.top = "auto";
      win.style.left = (wid === "w1") ? "24px" : "auto";
      win.style.right = (wid === "w4") ? "24px" : "auto";
      win.style.height = Math.min(600, vh - 110) + "px";
      win.style.maxHeight = win.style.height;
      return;
  }
  
  const openDown = spaceBelow > spaceAbove;
  
  const avail = openDown ? spaceBelow : spaceAbove;
  const maxH = Math.min(600, avail - 78 - 16);
  win.style.minHeight = "0px"; // prevent any CSS min-height from overriding
  win.style.maxHeight = `${maxH}px`;

  if (openDown) {
    win.style.top = (rect.bottom + 8) + "px";
    win.style.bottom = "auto";
    win.style.maxHeight = spaceBelow - 8 + "px";
    win.style.height = spaceBelow - 8 + "px";   // ← add this
  } else {
    win.style.bottom = (vh - rect.top + 8) + "px";
    win.style.top = "auto";
    win.style.maxHeight = spaceAbove - 8 + "px";
    win.style.height = spaceAbove - 8 + "px";   // ← add this
  }
}
    
function openWidget(wid) {
  setActiveWidget(wid);
  if (wid === "w1" || wid === "w4") bringToFront(wid);

  // Add class for w1 and w4 to hide their toggles on small screens
  if (wid === "w1" || wid === "w4") {
    const container = document.getElementById(`${wid}-lawfirm-chatbot-container`);
    if (container) container.classList.add('widget-open');
  }

  if (wid === "w5") {
    const overlay = getElFor(wid, "chat-modal-overlay");
    const win = getElFor(wid, "lawfirm-chat-window");
    if (overlay) overlay.classList.add("open");
    if (win) win.classList.add("open");

    // Initialize if empty
    const messagesDiv = document.getElementById(`${wid}-chat-messages`);
    if (messagesDiv && messagesDiv.children.length === 0) {
      initializeWidget(wid);
    }
    const input = getElFor(wid, "chat-input");
    if (input) input.focus();
    return;
  }

  const win = getElFor(wid, "lawfirm-chat-window");
  if (win) win.classList.add("open");

  // Initialize if empty
  const messagesDiv = document.getElementById(`${wid}-chat-messages`);
  if (messagesDiv && messagesDiv.children.length === 0) {
    initializeWidget(wid);
  }

  const input = getElFor(wid, "chat-input");
  if (input) input.focus();
}

function closeWidget(wid) {
  // Remove class for w1 and w4
  if (wid === "w1" || wid === "w4") {
    const container = document.getElementById(`${wid}-lawfirm-chatbot-container`);
    if (container) container.classList.remove('widget-open');
  }

  if (wid === "w5") {
    const overlay = getElFor(wid, "chat-modal-overlay");
    const win = getElFor(wid, "lawfirm-chat-window");
    if (overlay) overlay.classList.remove("open");
    if (win) win.classList.remove("open");
    return;
  }

  const win = getElFor(wid, "lawfirm-chat-window");
  if (win) win.classList.remove("open");

  const c = document.getElementById(`${wid}-lawfirm-chatbot-container`);
  if (c) c.classList.remove("is-front");

}

function toggleWidget(wid) {
  setActiveWidget(wid);

  if (wid === "w5") {
    const win = getElFor(wid, "lawfirm-chat-window");
    const isOpen = win && win.classList.contains("open");
    if (isOpen) closeWidget(wid); else openWidget(wid);
    return;
  }

  const win = getElFor(wid, "lawfirm-chat-window");
  if (!win) return;
  win.classList.toggle("open");
  if (win.classList.contains("open")) {
    const input = getElFor(wid, "chat-input");
    if (input) input.focus();
    const badge = getElFor(wid, "chat-notification");
    if (badge) badge.style.display = "none";
  }
}

function bindWidgetEvents() {
  WIDGET_IDS.forEach((wid) => {
    const root = document.getElementById(`${wid}-lawfirm-chatbot-container`);
    if (!root) return;

    // If the user interacts with a widget, keep it on top (important when multiple toggles exist)
    root.addEventListener("mousedown", () => {
      const win = getElFor(wid, "lawfirm-chat-window");
      if (win && win.classList.contains("open")) setFrontWidget(wid);
    });
    root.addEventListener("touchstart", () => {
      const win = getElFor(wid, "lawfirm-chat-window");
      if (win && win.classList.contains("open")) setFrontWidget(wid);
    }, { passive: true });
    

    const btn = document.getElementById(`${wid}-lawfirm-chat-button`);
    if (btn) {
      btn.addEventListener("click", (e) => {
        e.preventDefault();
        toggleWidget(wid);
      });
    }

    const closeBtn = root.querySelector('[data-action="close"]');
    if (closeBtn) {
      closeBtn.addEventListener("click", (e) => {
        e.preventDefault();
        closeWidget(wid);
      });
    }

    if (wid === "w5") {
      const overlay = document.getElementById(`${wid}-chat-modal-overlay`);
      if (overlay) overlay.addEventListener("click", () => closeWidget(wid));
    }

    root.querySelectorAll("[data-quick-action]").forEach((qaBtn) => {
      qaBtn.addEventListener("click", (e) => {
        e.preventDefault();
        setActiveWidget(wid);
        const action = qaBtn.getAttribute("data-quick-action");
        if (action) quickAction(action);
      });
    });

    // Bind quick action buttons
    const quickActionBtns = root.querySelectorAll('[data-quick-action]');
    quickActionBtns.forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        const action = btn.getAttribute('data-quick-action');
        handleQuickAction(wid, action);
      });
    });

    

    const sendBtn = root.querySelector('[data-action="send"]');
    if (sendBtn) {
      sendBtn.addEventListener("click", (e) => {
        e.preventDefault();
        setActiveWidget(wid);
        sendMessage();
      });
    }

    const input = document.getElementById(`${wid}-chat-input`);
    if (input) {
      input.addEventListener("keypress", (e) => {
        setActiveWidget(wid);
        handleKeyPress(e);
      });
    }

    const uploadBtn = root.querySelector('[data-action="upload"]');
    const fileInput = document.getElementById(`${wid}-file-input`);
    if (uploadBtn && fileInput) {
      uploadBtn.addEventListener("click", (e) => {
        e.preventDefault();
        fileInput.click();
      });
      fileInput.addEventListener("change", (e) => {
        setActiveWidget(wid);
        handleFileUpload(e);
      });
    }
  });

  const cta = document.getElementById("try-live-demo-btn");
  if (cta) {
    cta.addEventListener("click", (e) => {
      e.preventDefault();
      openWidget("w6");
    });
  }
}

function handleQuickAction(wid, action) {
  setActiveWidget(wid);
  const state = widgetStates[wid];
  
  // Find the step for this action
  const step = flowSteps[action] || flowSteps['start'];
  state.currentStep = action;
  
  // Add user message
  addMessage('user', `I need help with ${action.replace(/_/g, ' ')}`);
  
  // Add bot response
  if (step && step.prompt) {
    setTimeout(() => {
      addMessage('bot', step.prompt);
    }, 500);
  }
}


async function handlePaymentReturn() {
  const urlParams = new URLSearchParams(window.location.search);
  const paymentStatus = urlParams.get("payment");
  const stripeSessionId = urlParams.get("session_id");

  if (paymentStatus === "success" && stripeSessionId) {
    setActiveWidget("w6");
    openWidget("w6");

    setTimeout(() => addMessage("bot", "⏳ Verifying your payment..."), 450);

    try {
      const response = await fetch(`${API_URL}/api/payments/${stripeSessionId}/status`);
      const data = await response.json();

      if (data.success && data.status === "completed") {
        const clientName = localStorage.getItem("payment_client_name") || "Valued Client";
        const paymentAmount = localStorage.getItem("payment_amount") || "500";

        localStorage.removeItem("payment_in_progress");
        localStorage.removeItem("payment_client_name");
        localStorage.removeItem("payment_amount");

        const successMessage = `
          <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 16px; border-radius: 12px; text-align: center; margin: 12px 0;">
            <h3 style="margin: 0 0 8px 0; font-size: 22px;">✅ Payment Successful!</h3>
            <p style="margin: 6px 0; opacity: 0.95; font-size: 16px;">Thank you, ${clientName}!</p>
            <p style="margin: 6px 0; opacity: 0.95; font-size: 16px;">Amount: $${paymentAmount}</p>
          </div>
        `;
        addMessage("bot", successMessage);
      } else {
        addMessage("bot", "I couldn't verify the payment yet. If you were charged, our team will confirm shortly.");
      }
    } catch (err) {
      console.error("Payment status error:", err);
      addMessage("bot", "I had trouble verifying the payment. If you were charged, our team will confirm shortly.");
    }
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  bindWidgetEvents();

  // Initialize Stripe once (optional)
  try {
    const response = await fetch(`${API_URL}/api/payments/stripe-config`);
    const data = await response.json();
    if (data.success && data.publishable_key) {
      stripe = Stripe(data.publishable_key);
      console.log("✅ Stripe initialized (multi-style)");
    }
  } catch (e) {
    console.warn("Stripe init skipped:", e);
  }

  await handlePaymentReturn();
});
