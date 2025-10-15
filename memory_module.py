import networkx as nx
import pandas as pd
from dataclasses import asdict
from typing import List, Optional, Dict, Any, Tuple
from datetime import timedelta
from tkg_builder import CausalMemoryEvent, convert_csv_to_events


# --- P2.3: Adaptive Retrieval Functionality ---

def format_context_node(node_id: str, attributes: Dict[str, Any]) -> str:
    """Formats a single TKG node (event) into a concise string for the LLM prompt."""
    
    return (
        f"[EVENT ID: {node_id}]\n"
        f"  TIME: {pd.to_datetime(attributes['time_anchor_seconds'], unit='s').strftime('%Y-%m-%d')}\n"
        f"  TYPE: {attributes['event_type']} (Intensity: {attributes['intensity']})\n"
        f"  Location: {attributes['obj_state']}\n"
        f"  ACTION: {attributes['actor_subj']} ({attributes['actor_subj_country']}) -> {attributes['recipient_obj']} ({attributes['recipient_obj_country']})\n"
        f"  SUMMARY: {attributes['context_summary']}\n"
    )

def retrieve_context(
    tkg: nx.DiGraph, 
    target_event_id: str, 
    policy: str, 
    time_window_days: int
) -> str:
    """
    Retrieves a context slice (STM/LTM) from the TKG based on an adaptive policy.

    Args:
        tkg: The Temporal Knowledge Graph (networkx.DiGraph).
        target_event_id: The ID of the event whose successor we are predicting.
        policy: The retrieval strategy ('baseline', 'intensity_aware', 'actor_history').
        time_window_days: The default temporal window for the baseline policy.

    Returns:
        A concatenated string of relevant historical events (the STM/LTM slice).
    """
    if target_event_id not in tkg:
        return ""

    target_node = tkg.nodes[target_event_id]
    target_time_seconds = target_node['time_anchor_seconds']
    
    # Start with the target event itself, as context for the prediction target
    context_nodes_list = [(target_event_id, target_node)]
    
    # Find all predecessor nodes (potential causes)
    predecessor_ids = list(tkg.predecessors(target_event_id))
    
    # Convert predecessor IDs to nodes with attributes
    predecessor_nodes: List[Tuple[str, Dict[str, Any]]] = [
        (nid, tkg.nodes[nid]) for nid in predecessor_ids
    ]
    
    # Sort by time, oldest first
    predecessor_nodes.sort(key=lambda x: x[1]['time_anchor_seconds'])

    # --- Apply Adaptive Policies ---
    
    if policy == 'baseline':
        # Baseline (STM/Recency): Use only nodes within the fixed time window
        context_nodes_list.extend(predecessor_nodes)
        
    elif policy == 'intensity_aware':
        # Intensity-Aware (STM Focus with Priority): Prioritize high-intensity events
        
        # Define high intensity events (e.g., intensity > 5 or < -5)
        HIGH_INTENSITY_THRESHOLD = 5.0
        
        # Retrieve all events within the time window
        recent_nodes = [
            (nid, attr) for nid, attr in predecessor_nodes
            if target_time_seconds - attr['time_anchor_seconds'] <= time_window_days * 86400  # seconds in a day
        ]
        
        # Separate high-intensity and moderate-intensity events
        high_intensity = [
            (nid, attr) for nid, attr in recent_nodes
            if attr['intensity'] >= HIGH_INTENSITY_THRESHOLD or attr['intensity'] <= -HIGH_INTENSITY_THRESHOLD
        ]
        moderate_intensity = [
            (nid, attr) for nid, attr in recent_nodes
            if abs(attr['intensity']) < HIGH_INTENSITY_THRESHOLD
        ]
        
        # Prioritize high intensity and then add moderate ones (e.g., 50/50 split to keep context size manageable)
        context_nodes_list.extend(high_intensity)
        context_nodes_list.extend(moderate_intensity)
        
    elif policy == 'actor_history':
        # Actor-Based (LTM Focus): Retrieve relevant long-term history for key actors
        
        key_actors = {target_node['actor_subj'], target_node['recipient_obj']}
        relevant_history = []
        
        # Iterate over ALL nodes in the TKG to find relevant LTM history
        for nid, attr in tkg.nodes.items():
            if nid != target_event_id:
                # Check for shared key actors
                if attr['actor_subj'] in key_actors or attr['recipient_obj'] in key_actors:
                    relevant_history.append((nid, attr))

        # Sort the entire relevant history by time
        relevant_history.sort(key=lambda x: x[1]['time_anchor_seconds'])

        # LTM Optimization: Select the 10 most recent and 5 most distant events
        # This keeps the LTM access token-efficient (the "summarized" or "clustered" idea)
        RECENT_LIMIT = 10
        OLDEST_LIMIT = 5
        
        # The LTM context should be the most recent history plus the most distant history
        ltm_slice = relevant_history[-RECENT_LIMIT:] + relevant_history[:OLDEST_LIMIT]

        unique_events_map = {}
        for nid, attr in ltm_slice:
            unique_events_map[nid] = attr
            
        # Rebuild the list of (nid, attr) tuples
        unique_ltm_slice = [(nid, attr) for nid, attr in unique_events_map.items()]
        unique_ltm_slice.sort(key=lambda x: x[1]['time_anchor_seconds'])
                
        context_nodes_list.extend(unique_ltm_slice)
        
    else:
        # Fallback to baseline if policy is unknown
        context_nodes_list.extend(predecessor_nodes)

    # 3. Format the final context string, ensuring temporal order (oldest first)
    
    # Remove duplicates introduced by policy mixing (if any) and re-sort by time (oldest first)
    # The target event should remain the LAST element in the formatted output for the LLM to process
    
    # Convert list of (id, attr) tuples to a list of attributes only
    final_context_attributes = [attr for nid, attr in context_nodes_list]
    
    # Sort by time, oldest first
    final_context_attributes.sort(key=lambda attr: attr['time_anchor_seconds'])

    # Format into a single string
    formatted_context = "\n---\n".join(
        format_context_node(attr['event_id'], attr) for attr in final_context_attributes
    )
    
    return formatted_context

# --- Main Execution for Testing ---


# 2. Build TKG

TKG = nx.read_gml(r"C:\Users\Mitsy\itcs6190\TMS\tkg_memory_loc.gml")

# 3. Test Retrieval Policies
TARGET_EVENT_ID = '20220731-9012-5Fc459300405_Request' 
LTM_TARGET_EVENT_ID = '20220731-9012-5Fc459300405_Accuse' 

print("\n" + "="*50)
print(f"TESTING RETRIEVAL FOR TARGET: {TARGET_EVENT_ID}")

# Test 1: Baseline (STM Recency)
print("\n--- Policy: baseline (STM Recency) ---")
context_baseline = retrieve_context(TKG, TARGET_EVENT_ID, 'baseline', time_window_days=3)
print(context_baseline)

# Test 2: Intensity-Aware (STM Priority)
print("\n--- Policy: intensity_aware (STM Priority) ---")
context_intensity = retrieve_context(TKG, TARGET_EVENT_ID, 'intensity_aware', time_window_days=3)
print(context_intensity)

# Test 3: Actor-History (LTM Access) - Using E005 which has a very old predecessor E001
print("\n" + "="*50)
print(f"TESTING RETRIEVAL FOR TARGET: {LTM_TARGET_EVENT_ID}")
print("\n--- Policy: actor_history (LTM Access) ---")
context_actor = retrieve_context(TKG, LTM_TARGET_EVENT_ID, 'actor_history', time_window_days=3)
print(context_actor)
