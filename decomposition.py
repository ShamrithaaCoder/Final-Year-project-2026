import re

# Fallback mechanism for spacy/scispacy so testing doesn't break if models take long to download
try:
    import spacy
    # Try to load the medical sciSpacy model
    try:
        nlp = spacy.load("en_core_sci_sm")
    except OSError:
        # Fallback to standard spacy model if sciSpacy isn't installed yet
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
except ImportError:
    nlp = None

def decompose_claims(summary_text: str) -> list:
    """
    Module 1: Claim Decomposition.
    Takes a full medical summary and splits it into atomic claims (sentences).
    """
    if nlp is not None:
        doc = nlp(summary_text)
        claims = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    else:
        # Graceful fallback purely using regex heuristics
        sentences = re.split(r'(?<=[.!?]) +', summary_text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    return claims

def extract_entities(claim: str) -> list:
    """
    Module 2: Concept Extraction.
    Extract key medical entities to use as search queries.
    """
    if nlp is not None:
        doc = nlp(claim)
        # Extract entities using spacy's NER
        entities = [ent.text for ent in doc.ents]
        if not entities:
            # Fallback to noun chunks
            entities = [chunk.text for chunk in doc.noun_chunks]
    else:
        # Very basic fallback if NLP is missing (removes common stopwords basically)
        stopwords = {"is", "the", "a", "an", "for", "to", "with", "patient", "was", "diagnosed", "of", "and"}
        words = re.findall(r'\b\w+\b', claim)
        entities = [w for w in words if w.lower() not in stopwords and len(w) > 3]
    
    # Return top 2/3 important entities for PubMed search
    return list(set(entities))[:3]

# Test locally if script run by itself
if __name__ == "__main__":
    test_text = "The patient was diagnosed with type 2 diabetes. Metformin is a common treatment for type 2 diabetes. To permanently resolve the condition, antibiotics cure diabetes."
    extracted = decompose_claims(test_text)
    for c in extracted:
        print(f"Claim: {c}")
        print(f"Entities: {extract_entities(c)}\n")
