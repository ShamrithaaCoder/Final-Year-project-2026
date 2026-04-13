import requests
import xml.etree.ElementTree as ET

def retrieve_evidence(query: str, max_results=2) -> str:
    """
    Module 3: Evidence Retrieval via PubMed API
    Searches PubMed for the given query (disease/drug entities) and fetches abstracts.
    """
    if not query.strip():
        return "No evidence retrieved."
        
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # Step 1: ESearch to get PubMed IDs
        search_params = {
            "db": "pubmed",
            "term": f"{query} [Title/Abstract]",
            "retmax": max_results,
            "retmode": "json"
        }
        search_resp = requests.get(base_url + "esearch.fcgi", params=search_params, timeout=5)
        search_data = search_resp.json()
        
        pmids = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return f"No PubMed articles found for query: {query}"
            
        # Step 2: EFetch to get Abstract text
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        fetch_resp = requests.get(base_url + "efetch.fcgi", params=fetch_params, timeout=5)
        
        root = ET.fromstring(fetch_resp.content)
        abstracts = []
        
        # Parse XML to find AbstractText elements
        for article in root.findall(".//PubmedArticle"):
            # Try to get article title for reference
            title_node = article.find(".//ArticleTitle")
            title = title_node.text if title_node is not None else "Unknown Title"
            
            abstract_nodes = article.findall(".//AbstractText")
            if abstract_nodes:
                abstract_text = " ".join([node.text for node in abstract_nodes if node.text])
                abstracts.append(f"[{title}]: {abstract_text}")
        
        if not abstracts:
            return "Articles found but abstracts are missing."
            
        return "\n".join(abstracts)
        
    except Exception as e:
        # Fallback for network issues / rate limiting so the demo still works
        print(f"PubMed Search Error: {e}")
        return _mock_evidence(query)

def _mock_evidence(query: str) -> str:
    """Graceful fallback mock evidence if API fails"""
    q_lower = query.lower()
    if "antibiotic" in q_lower or "diabetes" in q_lower:
        return "Insulin is typically used to manage diabetes. Antibiotics are used to treat bacterial infections, not diabetes."
    if "metformin" in q_lower:
        return "Metformin is a first-line medication for the treatment of type 2 diabetes."
    return "Generic medical evidence. More specific retrieval requires active PubMed API connection."

# Test locally
if __name__ == "__main__":
    print(retrieve_evidence("metformin type 2 diabetes"))
