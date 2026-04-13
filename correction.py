import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

def correct_claim(claim: str, evidence: str) -> str:
    """
    Module 7: Correction Module
    Uses an LLM to correct an unsupported claim based on retrieved evidence.
    """
    # Using Gemini model, defaulting to mock if API key isn't provided to prevent errors
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    
    if not api_key:
        print("No API key found. Using mock correction.")
        # Mock behavior for testing without an API key mapping to user's example
        if "Antibiotics cure diabetes" in claim or "cure" in claim.lower():
            return "Insulin manages diabetes."
        return "Corrected claim based on evidence (Mocked - API key needed)"

    try:
        # Initialize LangChain with Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        
        template = """
        You are a medical verification expert. 
        You have identified an incorrect medical claim in an LLM-generated summary. 
        Using ONLY the provided evidence, rewrite the claim to be medically accurate.
        Keep the correction concise, factual, and in the same tone as the original claim.

        Incorrect Claim: {claim}
        Retrieved Evidence: {evidence}
        
        Corrected Claim:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["claim", "evidence"])
        chain = prompt | llm
        
        response = chain.invoke({"claim": claim, "evidence": evidence})
        return response.content.strip()
    except Exception as e:
        return f"Error during correction: {str(e)}"
