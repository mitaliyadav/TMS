import networkx as nx
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Set, Any
import datetime
import math
import json # Import the JSON module for structured string serialization


@dataclass
class CausalMemoryEvent:
    """
    Defines the core structure for a single event to be stored as a node in the TKG.
    """
    event_id: str
    time_anchor: pd.Timestamp
    event_type: str
    intensity: float
    actor_subj: str
    actor_subj_country: str
    recipient_obj: str
    recipient_obj_country: str
    obj_state: List[str]
    context_summary: str
    reasoning_trace: Optional[str] = None
    is_correct_prediction: Optional[bool] = None


# --- 2. Temporal Knowledge Graph Builder Function (P2.2) ---

def build_temporal_knowledge_graph(
    event_list: List[CausalMemoryEvent],
    time_window_days: int = 7,
    max_lookback_events: int = 5
) -> nx.DiGraph:
    """
    Constructs a Directed Graph (Temporal Knowledge Graph) from a list of CausalMemoryEvent objects.

    Nodes: Each CausalMemoryEvent.
    Edges: Temporal links from an earlier event to a later event IF they occur
           within a defined time window AND involve common entities (Actors/Recipients).

    Args:
        event_list: A list of CausalMemoryEvent objects.
        time_window_days: Maximum number of days an event can temporally link to a previous event.
        max_lookback_events: Maximum number of preceding events to check for temporal links.

    Returns:
        A networkx.DiGraph representing the TKG.
    """
    G = nx.DiGraph()

    # CRITICAL: Sort events chronologically to ensure temporal links are created correctly (A -> B)
    event_list.sort(key=lambda x: x.time_anchor)

    # Dictionary to quickly look up events by entity for linking purposes
    # Key: Actor/Recipient Country, Value: List of recent event IDs
    recent_events_by_entity = {}

    for i, current_event in enumerate(event_list):
        # 1. Add the current event as a Node
        # We use the event_id as the node name for simplicity and use the dataclass fields as attributes.
        
        node_attributes = asdict(current_event)
        
        # FIX 1: Ensure all attributes are serializable.
        # NetworkX cannot serialize the raw pandas Timestamp object.
        node_attributes['time_anchor_seconds'] = current_event.time_anchor.timestamp()
        node_attributes['time_anchor_str'] = current_event.time_anchor.isoformat()
        
        # Remove the non-serializable object before passing to NetworkX
        del node_attributes['time_anchor'] 
        
        # FIX 2: REVERT to using the raw Python List[str] for 'obj_state'.
        # NetworkX's GML writer should handle the List[str] attribute 
        # by producing the repeated-key format (e.g., obj_state "MH", obj_state "GJ") 
        # that the user's environment is expecting.
        
        G.add_node(current_event.event_id, **node_attributes)

        # 2. Establish Temporal Edges (Forward Linking)
        # We look back at a fixed number of preceding events to find potential causal ancestors.
        start_index = max(0, i - max_lookback_events)
        
        for j in range(start_index, i):
            preceding_event = event_list[j]

            time_diff = current_event.time_anchor - preceding_event.time_anchor
            time_diff_days = time_diff.total_seconds() / (60*60*24)
            
            # Check 1: Temporal Proximity
            if time_diff_days <= time_window_days:
                # Check 2: Shared Entity (Actor-Recipient/Location overlap is essential for causal linking)

                # Collect all principal entities and specific locations/states from the preceding event
                preceding_entities = (
                    {preceding_event.actor_subj_country, preceding_event.recipient_obj_country} | 
                    set(preceding_event.obj_state)
                )
                
                # Collect all principal entities and specific locations/states from the current event
                current_entities = (
                    {current_event.actor_subj_country, current_event.recipient_obj_country} | 
                    set(current_event.obj_state)
                )
                
                # Check for overlap between the two expanded sets
                if preceding_entities.intersection(current_entities):
                    
                    # Add a directed edge from the preceding event to the current event
                    # This is the "Temporal" link (A happened before B and they are related)
                    G.add_edge(
                        preceding_event.event_id,
                        current_event.event_id,
                        type='TEMPORAL',
                        time_gap_days=round(time_diff_days, 2),
                        # Future Phase 3/4: We will add the 'CAUSAL' label and 'causal_certainty' here
                        causal_label='UNKNOWN'
                    )

    return G

def convert_csv_to_events(df: pd.DataFrame) -> List[CausalMemoryEvent]:
    """
    Converts a pandas DataFrame (loaded from the CSV) into a list of 
    CausalMemoryEvent objects, handling necessary data cleaning and mapping.

    Args:
        df: The pandas DataFrame loaded from the source CSV file.

    Returns:
        A list of CausalMemoryEvent instances.
    """
    events: List[CausalMemoryEvent] = []
    
    # Standardize column names for easier access and handle potential NaNs 
    df = df.fillna('')
    
    for _, row in df.iterrows():
        try:
            # 1. Handle Time/Date Conversion
            # UPDATED: Using 'Event Date' based on the new screenshot
            time_anchor_str = str(row['Event Date']).split()[0] # Remove potential time parts
            time_anchor = pd.to_datetime(time_anchor_str, errors='coerce')

            if pd.isna(time_anchor):
                print(f"Skipping event {row.get('Event ID', 'Unknown')}: Invalid date.")
                continue

            # 2. Extract and Clean Location/Contexts
            # 'Contexts' column contains context keywords, separated by '|' in some datasets
            context_list = [c.strip() for c in str(row['Contexts']).split('|') if c.strip()]
            
            # Use 'Raw Placename' for the specific location/state codes (obj_state)
            raw_placename = str(row.get('Raw Placename', '')).strip()

            # --- CRITICAL FIX: Clean Python list notation from the string ---
            # Remove [ ] ' " characters that make the location codes look like a single string list
            cleaned_placename = raw_placename.replace('[', '').replace(']', '').replace("'", '').replace('"', '').strip()

            # Split by the delimiter (assuming ';') to get a list of individual state codes
            obj_state_list = [p.strip() for p in cleaned_placename.split(';') if p.strip()] if cleaned_placename else []
            # -----------------------------------------------------------------


            # 3. Create the CausalMemoryEvent object
            event = CausalMemoryEvent(
                event_id=str(row['Event ID']),
                time_anchor=time_anchor,
                event_type=str(row['Event Type']),
                intensity=float(row['Event Intensity']) if str(row['Event Intensity']) else 0.0,
                
                # Actor/Recipient Mapping
                actor_subj=str(row['Actor Name']),
                actor_subj_country=str(row['Actor Country']),
                recipient_obj=str(row['Recipient Name']),
                recipient_obj_country=str(row['Recipient Country']),
                
                # Summary and Location
                context_summary=f"Contexts: {', '.join(context_list)}. Summary: {str(row['Actor Title'])} acts against {str(row['Recipient Title'])}.",
                obj_state=obj_state_list,
                
                # Reasoning and Prediction fields remain None for raw data
                reasoning_trace="",
                is_correct_prediction=""
            )
            events.append(event)
        
        except (ValueError, KeyError, TypeError) as e:
            # Added TypeError and improved error message clarity
            print(f"Error processing row with ID {row.get('Event ID', 'Unknown')}: {e}")
            continue
            
    return events
# --- 3. TKG Search Index (Merged from tkg_search_index.py) ---

class TKGSearchIndex:
    """
    Creates an inverted index structure in memory to quickly search for
    CausalMemoryEvents based on Actor names, Country codes, or Date.

    This acts as the 'database' lookup layer for the master agent when the
    user provides non-ID-based queries.
    """
    def __init__(self, events: List[CausalMemoryEvent]):
        """
        Initializes the index by processing a list of all CausalMemoryEvent objects.
        """
        self.events = events
        self.actor_index: Dict[str, Set[str]] = {}       # Maps Actor Name -> {Event IDs}
        self.location_index: Dict[str, Set[str]] = {}    # Maps Country/Location Code -> {Event IDs}
        self.date_index: Dict[str, Set[str]] = {}        # Maps Date (YYYY-MM-DD) -> {Event IDs}
        self._build_index()

    def _build_index(self):
        """
        Processes all events and populates the inverted index dictionaries.
        """
        for event in self.events:
            event_id = event.event_id
            
            # --- 1. Actor Index (Includes Subject and Recipient) ---
            
            # Actor Subject
            actor_subj_key = event.actor_subj.upper().strip()
            self.actor_index.setdefault(actor_subj_key, set()).add(event_id)
            
            # Actor Recipient
            actor_recipient_key = event.recipient_obj.upper().strip()
            self.actor_index.setdefault(actor_recipient_key, set()).add(event_id)

            # --- 2. Location/Country Index (Uses available country codes) ---
            
            # Index by Subject Country
            subj_country_key = event.actor_subj_country.upper().strip()
            self.location_index.setdefault(subj_country_key, set()).add(event_id)

            # Index by Recipient Country
            recip_country_key = event.recipient_obj_country.upper().strip()
            self.location_index.setdefault(recip_country_key, set()).add(event_id)

            # --- 3. Date Index ---
            date_key = event.time_anchor.strftime('%Y-%m-%d')
            self.date_index.setdefault(date_key, set()).add(event_id)

    def search_events(
        self, 
        query_type: str, 
        query_value: str
    ) -> List[str]:
        """
        Searches the index for matching Event IDs based on the query type and value.
        
        Args:
            query_type: The field to search ('actor', 'location', 'date', 'country').
            query_value: The value to match (e.g., 'India', '2023-01-01').
            
        Returns:
            A list of matching Event IDs.
        """
        query_value = query_value.upper().strip()
        matching_ids: Optional[Set[str]] = None
        
        if query_type == 'actor':
            # Direct lookup in the actor index
            matching_ids = self.actor_index.get(query_value, set())
            
        elif query_type == 'location' or query_type == 'state' or query_type == 'country':
            # Lookup in the location index (which now maps to countries/locations)
            matching_ids = self.location_index.get(query_value, set())

        elif query_type == 'date':
            # Direct lookup in the date index (requires YYYY-MM-DD format)
            matching_ids = self.date_index.get(query_value, set())
            
        if matching_ids is None:
            return []
            
        return list(matching_ids)


# --- 4. Example Usage and Verification ---

if __name__ == "__main__":
    # Simulate data loading (similar to the previous step's output)
    event_data_df = pd.read_csv(r"C:\Users\Mitsy\itcs6190\TMS\data\IND_train.csv")

    # FIX: Convert the DataFrame into the required list of CausalMemoryEvent objects
    event_list = convert_csv_to_events(event_data_df)
    print(f"Converted {len(event_list)} rows to CausalMemoryEvents.")

    # Build the TKG
    TKG = build_temporal_knowledge_graph(
        event_list, # Pass the converted list, not the DataFrame
        time_window_days=8,
        max_lookback_events=3
    )

    print(f"--- TKG Creation Summary ---")
    print(f"Total Nodes (Events): {TKG.number_of_nodes()}")
    print(f"Total Edges (Temporal Links): {TKG.number_of_edges()}")

    if TKG.number_of_nodes() > 0:
        # Get the ID of the first node for inspection
        sample_node_id = list(TKG.nodes.keys())[0]
        node_data = TKG.nodes[sample_node_id]
        
        print(f"\n--- DEBUG: Inspecting raw node attributes for ID {sample_node_id} ---")
        
        # Check the obj_state attribute
        obj_state_val = node_data.get('obj_state')
        
        print(f"Value of obj_state: {obj_state_val}")
        print(f"Data type of obj_state: {type(obj_state_val)}")
        if isinstance(obj_state_val, list):
            print(f"obj_state is a list of size: {len(obj_state_val)}")
            print("Status: The attribute is correctly stored as a Python list in NetworkX's memory.")
        else:
            print("Status: ERROR - The attribute is NOT a Python list in memory.")
            
        # Check the context_summary attribute (for comparison)
        print(f"Value of context_summary: {node_data.get('context_summary')}")
    # Build the Search Index
    # FIX: Pass the 'event_list' (list of objects) instead of 'event_data_df' (DataFrame)
    index = TKGSearchIndex(event_list)
    
    print("\n--- TKG Search Index Verification ---")
    
    # Test 1: Actor Search
    union_events = index.search_events(query_type='actor', query_value='Shiromani Akali Dal')
    #print(f"Events involving 'Union': {union_events}") 
    
    # Test 2: Country Search
    country_a_events = index.search_events(query_type='country', query_value='India')
    #print(f"Events involving Country 'A': {country_a_events}") 

    # Test 3: Date Search
    date_events = index.search_events(query_type='date', query_value='2022-01-12')
    #print(f"Events on '2025-01-02': {date_events}") 
    # IMPORTANT STEP for P2.4: Demonstrate persistence
    nx.write_gml(TKG, "tkg_memory_loc2.gml")
    print("\nGraph saved to tkg_memory_loc2.gml (Persistence Checkpoint).")
