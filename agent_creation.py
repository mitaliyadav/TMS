import networkx as nx
import json
import time
from typing import Dict, Any, Optional, List, Tuple
# Assuming these modules are available in your environment:
from tkg_builder import CausalMemoryEvent, TKGSearchIndex, build_temporal_knowledge_graph
import requests # Used for external API calls
import pandas as pd # Needed for demonstration/data handling

# --- Configuration ---
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
# NOTE: The API_KEY will be automatically provided by the Canvas environment for the fetch call.
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="
API_KEY = "" 

# --- Placeholder for retrieve_context ---
# Since memory_module.py is not available, we define a placeholder function
# that simulates the output needed by the ContextRetrievalAgent.
def retrieve_context(tkg: nx.DiGraph, target_event_id: str, policy: str, time_window_days: int) -> str:
    """
    Placeholder for the memory module's context retrieval function.
    Simulates retrieving the target event and its immediate predecessors from the TKG.
    """
    if not tkg.has_node(target_event_id):
        return "ERROR: Target event not found."
    
    context_events = []
    # Simple simulation: retrieve the target event and its immediate predecessors (up to 3)
    target_node = tkg.nodes[target_event_id]
    context_events.append(f"TARGET_EVENT: {target_node.get('event_id', target_event_id)} ({target_node.get('time_anchor_str', 'N/A')[:10]}) - {target_node.get('actor_subj', 'N/A')} vs {target_node.get('recipient_obj', 'N/A')} - {target_node.get('event_type', 'N/A')} @ Intensity {target_node.get('intensity', 0.0):.1f}")

    predecessors = list(tkg.predecessors(target_event_id))
    for i, predecessor_id in enumerate(predecessors[:3]):
        pred_node = tkg.nodes[predecessor_id]
        context_events.append(f"PRECURSOR {i+1}: {pred_node.get('event_id', predecessor_id)} ({pred_node.get('time_anchor_str', 'N/A')[:10]}) - {pred_node.get('actor_subj', 'N/A')} vs {pred_node.get('recipient_obj', 'N/A')} - {pred_node.get('event_type', 'N/A')} @ Intensity {pred_node.get('intensity', 0.0):.1f}")

    return "\n".join(context_events)

# --- LLM Structured Output Schemas ---

QUERY_PARSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "actor": {"type": "STRING", "description": "Primary actor mentioned (e.g., 'Protestors', 'Government of India'). Use an empty string if not found."},
        "recipient": {"type": "STRING", "description": "Primary recipient mentioned (e.g., 'Police', 'NGO'). Use an empty string if not found."},
        "location": {"type": "STRING", "description": "Specific location, state, or country mentioned (e.g., 'Mumbai', 'MH'). Use an empty string if not found."},
        "date": {"type": "STRING", "description": "Specific date mentioned in YYYY-MM-DD format. Use an empty string if not found."}
    },
    "required": ["actor", "recipient", "location", "date"]
}

MULTI_PREDICTION_SCHEMA = {
    "type": "ARRAY",
    "description": "A list of 3-5 potential next events, ranked by the agent's calculated probability.",
    "items": {
        "type": "OBJECT",
        "properties": {
            "rank": {"type": "INTEGER", "description": "The probability rank, starting at 1 (most likely)."},
            "event_summary": {"type": "STRING", "description": "A concise summary of the predicted event (e.g., 'Government issues an executive order affecting rural farmers')."},
            "probability": {"type": "NUMBER", "description": "The estimated probability of this event occurring, between 0.0 and 1.0 (e.g., 0.45)."},
            "causal_link": {"type": "STRING", "description": "Brief explanation referencing the context events that justify this prediction."}
        },
        "required": ["rank", "event_summary", "probability", "causal_link"]
    }
}

# --- Shared LLM API Caller ---
def call_llm_api(prompt: str, system_instruction: str, output_schema: Dict[str, Any]) -> Optional[Dict[str, Any] | List[Dict[str, Any]]]:
    """Generic function to call the Gemini API with structured output and backoff."""
    
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": output_schema
        },
        "systemInstruction": {
            "parts": [{ "text": system_instruction }]
        }
    }

    # Exponential backoff mechanism
    for attempt in range(4):
        try:
            # We use the requests library for API calls
            response = requests.post(
                GEMINI_API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            
            response.raise_for_status() 
            result = response.json()
            
            candidate = result.get('candidates', [{}])[0]
            json_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
            
            if json_text:
                return json.loads(json_text)
            
            raise ValueError("LLM response content was empty or missing.")

        except requests.exceptions.RequestException as e:
            # print(f"API Request Error on attempt {attempt + 1}: {e}") # Do not log retries
            pass
        except Exception as e:
            # print(f"Error processing LLM response on attempt {attempt + 1}: {e}") # Do not log retries
            pass
            
        if attempt < 3:
            time.sleep(2 ** attempt) 
        else:
            break 

    print("Error: Failed to get valid response from LLM after multiple retries.")
    return None


# --- 1. Query Parsing Agent ---
class QueryParsingAgent:
    """Agent responsible for extracting entities (actor, recipient, location, date) from a natural language query."""
    
    def parse_query(self, query: str) -> Dict[str, str]:
        """Uses an LLM to extract structured entities from the query."""
        system_instruction = (
            "You are a sophisticated NLP parser. Your goal is to extract the primary "
            "actor, recipient, location (state/country), and date from the user query. "
            "Dates must be standardized to YYYY-MM-DD. Respond STRICTLY in the required JSON format."
        )
        
        user_prompt = f"Analyze the following user prediction request: '{query}'"
        
        parsed_result = call_llm_api(user_prompt, system_instruction, QUERY_PARSE_SCHEMA)
        
        if parsed_result and isinstance(parsed_result, dict):
            return {
                "actor": parsed_result.get("actor", ""),
                "recipient": parsed_result.get("recipient", ""),
                "location": parsed_result.get("location", ""),
                "date": parsed_result.get("date", "")
            }
        
        print("Warning: Query Parsing failed. Using empty entities.")
        return {"actor": "", "recipient": "", "location": "", "date": ""}

# --- 2. Context Retrieval Agent ---
class ContextRetrievalAgent:
    """
    Agent responsible for finding the most relevant historical context (a single target event ID 
    and its chain of predecessors) based on parsed entities.
    """
    def __init__(self, search_index: TKGSearchIndex, event_list: List[CausalMemoryEvent]):
        self.search_index = search_index
        self.event_list = event_list
        
    def find_and_format_context(
        self, 
        entities: Dict[str, str], 
        tkg: nx.DiGraph, 
        retrieval_policy: str = 'causal_chain'
    ) -> Tuple[str, Optional[str]]:
        """
        Finds the most recent event matching the entities and retrieves its context.
        
        Returns: (formatted_context_string, target_event_id)
        """
        actor = entities.get('actor')
        location = entities.get('location')
        date = entities.get('date')
        
        # 1. Resolve to Target Event ID (Find the anchor event for the prediction)
        matching_ids: List[str] = [e.event_id for e in self.event_list]

        if date:
            date_matches = set(self.search_index.search_events(query_type='date', query_value=date))
            matching_ids = list(set(matching_ids).intersection(date_matches))
        
        if actor or entities.get('recipient'):
            actor_or_recip = actor or entities['recipient']
            actor_matches = set(self.search_index.search_events(query_type='actor', query_value=actor_or_recip))
            matching_ids = list(set(matching_ids).intersection(actor_matches))
        
        if location:
            loc_matches = set(self.search_index.search_events(query_type='location', query_value=location))
            matching_ids = list(set(matching_ids).intersection(loc_matches))

        if not matching_ids:
            return "ERROR: No recent historical event found matching the query criteria.", None

        # Find the most recent event among the matches to serve as the prediction anchor
        matched_events = [e for e in self.event_list if e.event_id in matching_ids]
        matched_events.sort(key=lambda x: x.time_anchor, reverse=True)

        target_event_id = matched_events[0].event_id
        
        # 2. Retrieve Context (using the retrieve_context function)
        formatted_context = retrieve_context(tkg, target_event_id, policy=retrieval_policy, time_window_days=300)
        
        if "ERROR" in formatted_context:
            return formatted_context, None
            
        return formatted_context, target_event_id

# --- 3. Multi-Prediction Agent ---
class MultiPredictionAgent:
    """
    Agent responsible for generating multiple, ranked predictions based on the context.
    """
    def generate_predictions(self, formatted_context: str) -> Optional[List[Dict[str, Any]]]:
        """
        Calls the LLM to analyze the context and return a ranked list of likely future events.
        """
        
        system_instruction = (
            "You are a geopolitical forecasting model. Analyze the provided historical context "
            "to predict 3 to 5 most likely immediate follow-up events. Rank them by probability (1 is highest). "
            "You MUST output a structured JSON array conforming to the schema. The 'causal_link' must "
            "explicitly connect the prediction to the provided context events."
        )

        user_prompt = (
            f"Analyze the following temporal event history (sorted oldest to newest) and predict a ranked list of 3-5 potential next events:\n\n"
            f"{formatted_context}\n\n"
        )
        
        json_response = call_llm_api(user_prompt, system_instruction, MULTI_PREDICTION_SCHEMA)
        
        # The result should be a list (the output of json.loads on the JSON array text)
        if json_response and isinstance(json_response, list):
            # Sort the final list by rank just to be safe, then return
            return sorted(json_response, key=lambda x: x.get('rank', 99))
        
        return None

# --- 4. Orchestrator Agent ---
class OrchestratorAgent:
    """
    The central agent that manages the entire prediction workflow by coordinating all specialized agents.
    """
    def __init__(self, tkg: nx.DiGraph, event_list: List[CausalMemoryEvent]):
        self.tkg = tkg
        self.event_list = event_list
        # The search index is crucial for the retrieval agent
        self.search_index = TKGSearchIndex(event_list)
        
        # Initialize sub-agents
        self.parser_agent = QueryParsingAgent()
        self.retrieval_agent = ContextRetrievalAgent(self.search_index, self.event_list)
        self.prediction_agent = MultiPredictionAgent()
        
        print("Orchestrator Agent and Sub-Agents initialized.")

    def run_prediction_pipeline(self, user_query: str, retrieval_policy: str = 'causal_chain') -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Executes the full Plan-and-Solve process using specialized agents.
        
        Returns: (Execution Trace, Ranked Prediction List)
        """
        trace = ["--- Orchestrator: Starting Prediction Pipeline ---"]
        
        # --- STEP 1: Parse Query ---
        trace.append("\n[1/3] Query Parsing Agent: Extracting entities...")
        entities = self.parser_agent.parse_query(user_query)
        trace.append(f"   -> Entities Found: Actor='{entities['actor']}', Location='{entities['location']}', Date='{entities['date']}'")
        
        # --- STEP 2: Retrieve Context ---
        trace.append("\n[2/3] Context Retrieval Agent: Finding anchor event and context...")
        formatted_context, target_event_id = self.retrieval_agent.find_and_format_context(
            entities, self.tkg, retrieval_policy
        )
        
        if target_event_id is None or "ERROR" in formatted_context:
            trace.append(f"   -> Retrieval failed: {formatted_context}")
            trace.append("--- Orchestrator: Pipeline Aborted ---")
            return "\n".join(trace), None

        trace.append(f"   -> Anchor Event ID Found: {target_event_id}")
        trace.append(f"   -> Context Retrieved ({len(formatted_context.splitlines())} events).")

        # --- STEP 3: Generate Predictions (SOLVE) ---
        trace.append("\n[3/3] Multi-Prediction Agent: Generating and ranking events...")
        ranked_predictions = self.prediction_agent.generate_predictions(formatted_context)
        
        if ranked_predictions is None:
            trace.append("   -> Prediction failed. Check LLM API response.")
            trace.append("--- Orchestrator: Pipeline Finished with Error ---")
            return "\n".join(trace), None

        trace.append(f"   -> Successfully generated {len(ranked_predictions)} ranked predictions.")
        
        return "\n".join(trace), ranked_predictions

# --- Example Usage (Main block) ---

if __name__ == "__main__":
    
    # --- Setup: Simulate TKG and Events ---
    print("Initializing dummy TKG and Event List for demonstration...")
    
    # Create a tiny dummy DataFrame/Events and TKG for testing the agent flow
    dummy_data = {
        'event_id': ['E001', 'E002', 'E003', 'E004'],
        'time_anchor': ['2024-10-01', '2024-10-05', '2024-10-10', '2024-10-15'],
        'event_type': ['PROTEST', 'MEETING', 'ARRESTS', 'NEGOTIATION'],
        'intensity': [5.0, 2.0, 7.0, 3.0],
        'actor_subj': ['Union', 'Govt', 'Police', 'Govt'],
        'actor_subj_country': ['India', 'India', 'India', 'India'],
        'recipient_obj': ['Govt', 'Union', 'Union', 'Union'],
        'recipient_obj_country': ['India', 'India', 'India', 'India'],
        'obj_state': ['MH, DL', 'MH', 'DL', 'MH'],
        'context_summary': ['Protest over wages.', 'Talks held.', 'Riot police deployed.', 'New talks scheduled.']
    }
    event_list = [
        CausalMemoryEvent(
            event_id=dummy_data['event_id'][i],
            time_anchor=pd.to_datetime(dummy_data['time_anchor'][i]),
            event_type=dummy_data['event_type'][i],
            intensity=dummy_data['intensity'][i],
            actor_subj=dummy_data['actor_subj'][i],
            actor_subj_country=dummy_data['actor_subj_country'][i],
            recipient_obj=dummy_data['recipient_obj'][i],
            recipient_obj_country=dummy_data['recipient_obj_country'][i],
            # Note: We need to manually convert 'MH, DL' string to list ['MH', 'DL'] for the dataclass
            obj_state=[s.strip() for s in dummy_data['obj_state'][i].split(',')], 
            context_summary=dummy_data['context_summary'][i]
        ) for i in range(len(dummy_data['event_id']))
    ]
    
    TKG = nx.DiGraph()
    TKG.add_node('E001', event_id='E001', time_anchor_str='2024-10-01', actor_subj='Union', recipient_obj='Govt', event_type='PROTEST', intensity=5.0)
    TKG.add_node('E002', event_id='E002', time_anchor_str='2024-10-05', actor_subj='Govt', recipient_obj='Union', event_type='MEETING', intensity=2.0)
    TKG.add_node('E003', event_id='E003', time_anchor_str='2024-10-10', actor_subj='Police', recipient_obj='Union', event_type='ARRESTS', intensity=7.0)
    TKG.add_node('E004', event_id='E004', time_anchor_str='2024-10-15', actor_subj='Govt', recipient_obj='Union', event_type='NEGOTIATION', intensity=3.0)
    TKG.add_edge('E001', 'E002')
    TKG.add_edge('E002', 'E003')
    TKG.add_edge('E003', 'E004')
    
    # --- Initialize and Run ---
    orchestrator = OrchestratorAgent(tkg=TKG, event_list=event_list)
    
    sample_query = "What is the most likely event to happen next involving the Union in MH after the meeting on 2024-10-05?"
    
    print(f"\n--- Running Prediction Pipeline for Query: '{sample_query}' ---")
    
    trace, predictions = orchestrator.run_prediction_pipeline(sample_query)
    
    print("\n" + "="*50)
    print("          EXECUTION TRACE")
    print("="*50)
    print(trace)
    
    print("\n" + "="*50)
    print("       FINAL RANKED PREDICTIONS")
    print("="*50)
    if predictions:
        for p in predictions:
            # Check for keys safely
            rank = p.get('rank', 'N/A')
            probability = p.get('probability', 0.0)
            summary = p.get('event_summary', 'N/A')
            causal_link = p.get('causal_link', 'N/A')

            print(f"Rank {rank} (Prob: {probability:.2f}): {summary}")
            print(f"  Causal Link: {causal_link}\n")
    else:
        print("No predictions were generated.")
