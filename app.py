import streamlit as st
import pandas as pd
import time

# Import our custom modules
from decomposition import decompose_claims, extract_entities
from retrieval import retrieve_evidence
from scoring import score_claim
from correction import correct_claim

# --- Theme & Visual Settings ---
st.set_page_config(
    page_title="Factual Hallucination Detection",
    page_icon="⚕️",
    layout="wide",
)

st.title("⚕️ Factual Hallucination Detection & Correction")
st.markdown("### Full Post-Generation Verification Pipeline for Medical Text")

# --- Integrated Core Pipeline Function ---
def run_full_pipeline(summary_text):
    results = []
    
    # Module 1: Decompose
    st.toast("Module 1: Extracting claims...", icon="✂️")
    claims = decompose_claims(summary_text)
    
    progress_bar = st.progress(0)
    
    for i, claim in enumerate(claims):
        status_text = st.empty()
        status_text.text(f"Processing Claim {i+1}/{len(claims)}...")
        
        # Module 2: Concept Extraction
        entities = extract_entities(claim)
        query = " ".join(entities)
        
        # Module 3: Evidence Retrieval
        status_text.text(f"Module 3: Retrieving PubMed Evidence for '{query}'...")
        raw_evidence = retrieve_evidence(query)
        
        # Modules 4 & 5: Scoring 
        status_text.text(f"Modules 4 & 5: Semantic Scoring...")
        score_data = score_claim(claim, raw_evidence)
        
        results.append({
            "id": i + 1,
            "claim": claim,
            "entities": entities,
            "search_query": query,
            "evidence": score_data["best_evidence_sentence"],
            "raw_evidence_context": raw_evidence,
            "score": score_data["score"],
            "result": score_data["result"]
        })
        
        progress_bar.progress((i + 1) / len(claims))
        time.sleep(0.5) # Slight UX delay
        status_text.empty()
        
    return results


# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key (for LLM correction)", type="password")
    if api_key:
        import os
        os.environ["GOOGLE_API_KEY"] = api_key
    st.info("Enter your Gemini API key for dynamic LLM corrections in Module 7. Without it, the app gracefully falls back to mock logic.")

    st.markdown("---")
    st.markdown("**Active Pipeline Modules:**")
    st.markdown('''
    1. **Decomposition:** scispacy (Sentence Split)
    2. **Entity Extraction:** spacy NER
    3. **Retrieval:** PubMed e-utils API 🟢
    4. **Embedding:** TF/Sentence-Transformers 🟢
    5. **Scoring:** NLI / Cosine Similarity 🟢
    6. **Pipeline Routing** 🟢
    7. **Correction:** LangChain LLM 🟢
    8. **Output:** Streamlit UI 🟢
    ''')

# --- Main UI ---
st.subheader("Input Medical Summary")
sample_text = "The patient was diagnosed with type 2 diabetes. Metformin is a common treatment for type 2 diabetes. To permanently resolve the condition, antibiotics cure diabetes."
summary_input = st.text_area("LLM-Generated Summary", value=sample_text, height=150)

if st.button("Run Verification Pipeline", type="primary"):
    with st.spinner("Initializing Pipeline..."):
        pipeline_results = run_full_pipeline(summary_input)
        
        st.divider()
        st.subheader("🖥️ Output Module (Results)")
        
        # Module 8 output rendering
        for item in pipeline_results:
            expander_icon = "❌" if item["result"] == "Unsupported" else "✅"
                
            with st.expander(f"{expander_icon} Claim {item['id']}: {item['claim']} - {item['result']}", expanded=True):
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Medical Entities:** {', '.join(item['entities']) if item['entities'] else 'None'}")
                    st.markdown(f"**PubMed Query:** `{item['search_query']}`")
                with col2:
                    st.markdown(f"**Confidence Score:** {item['score']}")
                
                st.markdown(f"**Best Match Evidence:**  \n_{item['evidence']}_")
                
                if item["result"] == "Unsupported":
                    st.error(f"❌ Original Claim: {item['claim']}")
                    
                    # Module 7: Correction Module Trigger
                    with st.spinner("Module 7: Correcting unsupported claim via LangChain..."):
                        corrected_text = correct_claim(item['claim'], item['raw_evidence_context'])
                    
                    item['corrected'] = corrected_text
                    st.success(f"✅ Corrected Claim: {corrected_text}")
                else:
                    item['corrected'] = item['claim']
                    
        st.divider()
        # Module 8 final reconstruction
        st.subheader("📝 Final Corrected Summary")
        
        final_summary = summary_input
        for item in pipeline_results:
            if item["result"] == "Unsupported" and "corrected" in item:
                final_summary = final_summary.replace(item['claim'], item['corrected'])
                
        st.info(final_summary)
