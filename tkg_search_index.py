import pandas as pd
from typing import Dict, List, Set, Any, Optional
from event_data_model import CausalMemoryEvent # Assumes CausalMemoryEvent is imported

class TKGSearchIndex:
    """
    Creates an inverted index structure in memory to quickly search for
    CausalMemoryEvents based on Actor names, Location names, or Date.

    This acts as the 'database' lookup layer for the master agent when the
    user provides non-ID-based queries.
    """
    def __init__(self, events: List[CausalMemoryEvent]):
        """
        Initializes the index by processing a list of all CausalMemoryEvent objects.
        """
        self.events = events
        self.actor_index: Dict[str, Set[str]] = {}       # Maps Actor Name -> {Event IDs}
        self.location_index: Dict[str, Set[str]] = {}    # Maps Location (State) -> {Event IDs}
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

            # --- 2. Location Index (Uses state/country codes from obj_state) ---
            # NOTE: event.obj_state is assumed to be a list of state/country codes (e.g., ['RJ', 'UP'])
            for location_code in event.obj_state:
                location_key = location_code.upper().strip()
                self.location_index.setdefault(location_key, set()).add(event_id)
                
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
            query_value: The value to match (e.g., 'New Delhi', 'India', '2023-01-01').
            
        Returns:
            A list of matching Event IDs.
        """
        query_value = query_value.upper().strip()
        matching_ids: Optional[Set[str]] = None
        
        if query_type == 'actor':
            # Direct lookup in the actor index
            matching_ids = self.actor_index.get(query_value, set())
            
        elif query_type == 'location' or query_type == 'state':
            # Direct lookup in the location index (using state/UT codes)
            matching_ids = self.location_index.get(query_value, set())
        
        elif query_type == 'country':
            # For country, we look for matches across all location codes, as country codes 
            # might be mixed with state codes in the location_index if the data is global.
            # In a clean Indian dataset, this might be a single code (e.g., 'IN')
            # Here we assume a direct lookup on the code is sufficient based on the index design.
            matching_ids = self.location_index.get(query_value, set())

        elif query_type == 'date':
            # Direct lookup in the date index (requires YYYY-MM-DD format)
            matching_ids = self.date_index.get(query_value, set())
            
        if matching_ids is None:
            return []
            
        return list(matching_ids)