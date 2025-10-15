import networkx as nx
import json
import time
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from memory_module import retrieve_context # Assumes retrieval function is available
from tkg_builder import CausalMemoryEvent, TKGSearchIndex, convert_csv_to_events # Import necessary TKG components
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

# --- Configuration for LLM API ---
# NOTE: In a production environment, use a proper HTTP client with exponential backoff.
# We simulate the fetch logic here.
OPENAI_MODEL = "gpt-4o-mini" # Placeholder for an OpenAI-style model
OPENAI_URL = "https://api.openai.com/v1/chat/completions" # OpenAI Endpoint
load_dotenv()

# --- Pydantic Model for Structured Prediction Output (P3.1) ---
# The LLM is instructed to return the prediction in this precise format.
class OutputSchema(BaseModel):
    event_type_prediction: str
    actor_subj_prediction: str 
    recipient_obj_prediction: str 
    intensity_prediction: float
    reasoning_trace: str

class PredictionAgent:
    """
    Implements the agent's core intelligence (P3.1) and continual learning (P3.3).
    It orchestrates memory retrieval and LLM interaction.
    """
    def __init__(self, tkg: nx.DiGraph, event_list: List[CausalMemoryEvent]):
        """
        Initializes the agent with the Temporal Knowledge Graph (TKG) and 
        builds the TKG Search Index for flexible lookups.
        """
        self.tkg = tkg
        self.event_list = event_list
        # The TKGSearchIndex is now used for resolving actor/location/date queries
        self.search_index = TKGSearchIndex(event_list)
        self.client = OpenAI()
        print("PredictionAgent initialized with TKG and Search Index.")

    def _call_openai_api(self, prompt: str, system_instruction: str) -> Optional[Dict[str, Any]]:
        """
        Handles the LLM API call, including structured output configuration 
        and exponential backoff for robustness.
        """
        json_schema_desc = json.dumps(OutputSchema.model_json_schema(), indent=2)
        full_system_instruction = (
            f"{system_instruction} Your response MUST be a JSON object that strictly conforms to the following schema: "
            f"\n\n{json_schema_desc}"
        )
        
        completion = self.client.responses.parse(
            model=OPENAI_MODEL,
            input = [
                    {"role": "system", "content": full_system_instruction},
                    {"role": "user", "content": prompt}
                ],
                text_format= OutputSchema,
                temperature=  0.0 # Low temp for factual prediction
        )
        print(completion.output_parsed)
        return completion.output_parsed

    def _resolve_query_to_event_id(
        self, 
        actor: Optional[str], 
        location: Optional[str], 
        date: Optional[str]
    ) -> Tuple[Optional[str], str]:
        """
        Uses the TKGSearchIndex to find the most recent matching event ID 
        based on the provided criteria.

        Returns: (event_id, plan_trace_message)
        """
        plan_trace = "PLAN 1/3: Resolving query to a single target event ID.\n"
        
        # Start with all event IDs or filter based on criteria in order of specificity
        matching_ids: List[str] = [e.event_id for e in self.event_list]

        if date:
            date_matches = set(self.search_index.search_events(query_type='date', query_value=date))
            matching_ids = list(set(matching_ids).intersection(date_matches))
            plan_trace += f"  - Date search ('{date}') returned {len(matching_ids)} events.\n"
        
        if actor:
             actor_matches = set(self.search_index.search_events(query_type='actor', query_value=actor))
             matching_ids = list(set(matching_ids).intersection(actor_matches))
             plan_trace += f"  - Refined with Actor ('{actor}'): {len(matching_ids)} events remaining.\n"
        
        if location:
             loc_matches = set(self.search_index.search_events(query_type='location', query_value=location))
             matching_ids = list(set(matching_ids).intersection(loc_matches))
             plan_trace += f"  - Refined with Location ('{location}'): {len(matching_ids)} events remaining.\n"
        
        if not matching_ids:
            return None, plan_trace + "ERROR 1/3: No events found matching the criteria. Prediction aborted."

        # Find the most recent event among the matches
        # We need to look up the CausalMemoryEvent objects to sort them by timestamp
        matched_events = [e for e in self.event_list if e.event_id in matching_ids]
        matched_events.sort(key=lambda x: x.time_anchor, reverse=True) # Sort newest first

        # The target is the most recent event found. The prediction will be about its successor.
        target_event_id = matched_events[0].event_id
        plan_trace += f"STATUS 1/3: Resolved query to the most recent matching event ID: '{target_event_id}'.\n"
        return target_event_id, plan_trace


    def run_plan_and_solve_prediction(
        self, 
        actor: Optional[str] = None,
        location: Optional[str] = None,
        date: Optional[str] = None,
        target_event_id: Optional[str] = None, # Still allow ID for direct use
        retrieval_policy: str = 'causal_chain'
    ) -> Tuple[str, Optional[Dict[str, Any]], str]:
        """
        Implements the Plan-and-Solve agentic framework for event prediction.
        Accepts either a direct ID or flexible query arguments (actor, location, date).

        Args:
            actor, location, date: Optional criteria to find the target event.
            target_event_id: Optional direct ID of the event to predict based on.
            retrieval_policy: The memory policy for context creation.

        Returns:
            A tuple: (Plan trace string, Parsed prediction JSON, Formatted context string)
        """
        plan_trace_parts = []
        
        # --- PLAN STEP 1: Determine Target Event ID (Resolution) ---
        if target_event_id:
            target_id = target_event_id
            plan_trace_parts.append(f"PLAN 1/3: Using direct target event ID: '{target_id}'.")
        else:
            # Use the search index to resolve the query
            target_id, resolve_trace = self._resolve_query_to_event_id(actor, location, date)
            plan_trace_parts.append(resolve_trace)
            
            if not target_id:
                # If resolution failed, return the error trace
                return "\n".join(plan_trace_parts), None, ""
        
        # --- PLAN STEP 2: Find Event in the Graph (Search) ---
        if not self.tkg.has_node(target_id):
            error_msg = f"ERROR: Target event ID '{target_id}' not found in TKG structure. Cannot proceed with prediction."
            plan_trace_parts.append(error_msg)
            print(error_msg)
            return "\n".join(plan_trace_parts), None, ""
        
        plan_trace_parts.append(f"STATUS 2/3: Target event found: '{target_id}'. Proceeding to context creation.")
        
        # --- PLAN STEP 3: Create Context and Predict (Solve) ---
        plan_trace_parts.append(f"PLAN 3/3: Creating context using '{retrieval_policy}' policy and executing prediction.")

        prediction_result, formatted_context = self.predict_next_event(target_id, retrieval_policy)
        
        if prediction_result:
            plan_trace_parts.append("STATUS 3/3: Prediction successful. JSON response received.")
            
        else:
            plan_trace_parts.append("ERROR 3/3: Prediction failed. Check context or API connection.")
        
        # --- SOLVE: Return the results of the plan ---
        return "\n".join(plan_trace_parts), prediction_result, formatted_context

    def predict_next_event(
        self, 
        target_event_id: str, 
        retrieval_policy: str = 'causal_chain'
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        P3.1: Adaptive Reasoning. Predicts the next event using TKG context 
        and the LLM's CoT reasoning.
        """
        # 1. Retrieve Context (P3.2: Adaptive Memory)
        formatted_context = retrieve_context(
            self.tkg, target_event_id, policy=retrieval_policy, time_window_days=300
        )

        if "ERROR: Target event not found" in formatted_context:
            return None, formatted_context
        
        # 2. Build Prompt
        system_instruction = (
            "You are a geopolitical forecasting engine. Your task is to predict the immediate "
            "next event following the provided context events. You MUST output a structured JSON response. "
            "Use the Chain-of-Thought (CoT) pattern in the 'reasoning_trace' field by explicitly referencing "
            "event IDs and attributes from the context history to justify your prediction."
        )

        user_prompt = (
            f"Analyze the following temporal history of events (sorted oldest to newest):\n\n"
            f"{formatted_context}\n\n"
            f"Based on the trends and causal dependencies in this history, provide a structured prediction for the *immediate next event*."
        )

        # 3. Get Prediction
        prediction_result = self._call_openai_api(user_prompt, system_instruction) 
        
        if prediction_result:
            print("Prediction successful. Reasoning trace received.")
        else:
            print("Prediction failed.")

        return prediction_result, formatted_context

    def learn_from_prediction(
        self, 
        prediction_result: Dict[str, Any], 
        ground_truth_event: CausalMemoryEvent
    ):
        """
        P3.3: Continual Learning. Updates the TKG with the agent's reasoning trace 
        and the ground truth outcome.
        """
        if not prediction_result:
            print("Cannot learn: Prediction result is missing.")
            return

        # 1. Determine Correctness (Simplified match logic)
        predicted_type = prediction_result.get('event_type_prediction', '').lower()
        actual_type = ground_truth_event.event_type.lower()
        
        # A simple check for demonstration: is the predicted event type correct?
        is_correct = predicted_type == actual_type 
        
        # Note: A real-world check would involve fuzzy matching on actor/recipient/intensity.

        # 2. Extract Learning Data
        reasoning_trace = prediction_result.get('reasoning_trace', 'No trace provided.')
        
        # 3. Update the TKG node
        event_id_to_update = ground_truth_event.event_id
        
        if self.tkg.has_node(event_id_to_update):
            # Update the event node with the learning feedback
            self.tkg.nodes[event_id_to_update]['reasoning_trace'] = reasoning_trace
            self.tkg.nodes[event_id_to_update]['is_correct_prediction'] = is_correct
            
            # Log the learning outcome
            outcome = "CORRECT" if is_correct else "INCORRECT"
            print(f"Learning complete. Event {event_id_to_update} updated with outcome: {outcome}.")
        else:
            print(f"Learning failed: Event {event_id_to_update} not found in TKG to update.")

if __name__ == "__main__":
    # --- Setup Data and TKG for Agent Initialization ---
    try:
        # Load raw data and convert to event list
        event_data_df = pd.read_csv(r"C:\Users\Mitsy\itcs6190\TMS\data\IND_train.csv")
        event_list_for_agent = convert_csv_to_events(event_data_df)
        
        # Load the pre-built TKG
        TKG = nx.read_gml(r"C:\Users\Mitsy\itcs6190\TMS\tkg_memory_loc2.gml")
        print(f"TKG loaded successfully with {TKG.number_of_nodes()} nodes.")
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found. Ensure 'IND_train.csv' and 'tkg_memory_loc2.gml' exist.")
        print(e)
        TKG = nx.DiGraph() # Initialize empty graph to prevent crash
        event_list_for_agent = []

    # --- Initialize Agent ---
    pred_agent = PredictionAgent(tkg=TKG, event_list=event_list_for_agent) 
    
    # --- Example 1: Query by Actor and Date (Flexible) ---
    
    print("\n--- TEST 1: Prediction Query by ACTOR and DATE ---")
    plan_trace_actor, result_actor, context_actor = pred_agent.run_plan_and_solve_prediction(
        actor="Supreme Court", # Example Actor Name from the dataset
        date="2022-01-03",    # Example Date from the dataset
        retrieval_policy='recency'
    )
    print(plan_trace_actor)
    
    """
    # --- Example 2: Query by direct ID (Original way) ---
    if TKG.number_of_nodes() > 0:
        sample_id = list(TKG.nodes.keys())[0]
        print(f"\n--- TEST 2: Prediction Query by direct ID ({sample_id}) ---")
        plan_trace_id, result_id, context_id = pred_agent.run_plan_and_solve_prediction(
            target_event_id=sample_id,
            retrieval_policy='causal_chain'
        )
        print(plan_trace_id)
    """