"""
Multi-Agent Event Prediction System
Implements a six-agent architecture for temporal event prediction with ranking and consensus.
"""

import networkx as nx
import pandas as pd
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import uuid

# Import existing components
from tkg_builder import CausalMemoryEvent, TKGSearchIndex
from memory_module import retrieve_context

load_dotenv()

# ============================================================================
# 1. SCHEMA DEFINITIONS
# ============================================================================

class ParsedQuery(BaseModel):
    """Query Parser Output Schema"""
    time_anchor: str  # pd.Datetime as ISO string (reference event date)
    event_type: str = ""  # Default to empty string
    intensity: float = 0.0  # Default to 0.0
    actor_subj: str = ""  # Default to empty string
    recipient_obj: str = ""  # Default to empty string
    obj_state: List[str] = []  # Default to empty list
    context_summary: str
    reasoning_trace: Optional[str] = None
    prediction_horizon_days: int
    is_historical_query: bool = False  # True if asking about past events
    reference_event_description: Optional[str] = None  # Description of the reference event


class PredictedEvent(BaseModel):
    """Structure for a single predicted event"""
    event_type: str
    predicted_timestamp: str  # pd.Datetime as ISO string
    actors: List[str]
    location: str
    intensity: float
    description: str


class PolicyPrediction(BaseModel):
    """Prediction Output Structure from a single policy"""
    policy_name: str  # "baseline" | "intensity_aware" | "actor_history"
    predicted_event: PredictedEvent
    confidence_raw: float  # Policy-specific confidence (0-1)
    supporting_events: List[str]  # Event IDs
    reasoning: str


class RankedPrediction(BaseModel):
    """Ranked Output Structure"""
    prediction_id: str
    combined_prediction: PredictedEvent
    confidence_score: float  # Historical accuracy-based score
    contributing_policies: List[str]
    policy_confidences: Dict[str, float]
    consensus_bonus: float
    explanation: str


# ============================================================================
# 2. HISTORICAL ACCURACY DATABASE
# ============================================================================

HISTORICAL_ACCURACY = {
    "baseline": {
        "overall_accuracy": 0.65,
        "by_event_type": {
            "protest": 0.72,
            "conflict": 0.58,
            "negotiation": 0.70,
            "aid": 0.68,
            "accuse": 0.63,
            "threaten": 0.60
        },
        "by_horizon": {
            7: 0.75,
            30: 0.60,
            90: 0.50
        }
    },
    "intensity_aware": {
        "overall_accuracy": 0.58,
        "by_event_type": {
            "protest": 0.55,
            "conflict": 0.68,
            "negotiation": 0.45,
            "aid": 0.50,
            "accuse": 0.60,
            "threaten": 0.65
        },
        "by_horizon": {
            7: 0.62,
            30: 0.58,
            90: 0.53
        }
    },
    "actor_history": {
        "overall_accuracy": 0.70,
        "by_event_type": {
            "protest": 0.68,
            "conflict": 0.71,
            "negotiation": 0.75,
            "aid": 0.72,
            "accuse": 0.69,
            "threaten": 0.67
        },
        "by_horizon": {
            7: 0.68,
            30: 0.70,
            90: 0.72
        }
    }
}


# ============================================================================
# 3. QUERY PARSER AGENT
# ============================================================================

class QueryParserAgent:
    """
    Parses natural language queries into structured format.
    Extracts actors, locations, event types, and temporal anchors.
    """

    def __init__(self):
        self.client = OpenAI()
        print("QueryParserAgent initialized.")

    def parse(self, user_query: str, prediction_horizon_days: int = 30) -> ParsedQuery:
        """
        Parse natural language query into structured format.

        Args:
            user_query: Raw user query string
            prediction_horizon_days: How many days ahead to predict

        Returns:
            ParsedQuery object
        """
        system_instruction = """You are an expert NLP parser for geopolitical event queries.
        Extract actors, locations, event types, and time references from queries.

        There are TWO types of queries:

        1. FUTURE PREDICTION (is_historical_query = false):
           "What will Narendra Modi do in the next month in UP?"
           - time_anchor: current date
           - is_historical_query: false

        2. HISTORICAL EVENT-BASED (is_historical_query = true):
           "The Supreme Court accused the Government of UP on May 6, 2022. What happened after 2 months?"
           - time_anchor: "2022-05-06" (the reference event date)
           - reference_event_description: "The Supreme Court accused the Government of UP regarding healthcare"
           - is_historical_query: true
           - prediction_horizon_days: 60 (2 months)

        For historical queries, extract:
        - The specific date mentioned
        - The event that occurred on that date
        - The time period to look ahead (convert to days)"""

        user_prompt = f"""Parse this query and return JSON:

"{user_query}"

Determine if this is:
A) A historical event-based query (references a specific past event and date)
B) A future prediction query

Extract and return as JSON:
- actor_subj: Primary actor (subject)
- recipient_obj: Recipient (if any)
- obj_state: Location(s) as a list
- event_type: Type of event mentioned (if any)
- time_anchor: The reference date (past event date OR current date)
- is_historical_query: true if asking about what happened AFTER a past event, false otherwise
- reference_event_description: Description of the reference event (only for historical queries)
- context_summary: Brief summary of the query

Current date: {datetime.now().strftime('%Y-%m-%d')}
Default horizon: {prediction_horizon_days} days

Return your answer in JSON format.
"""

        try:
            # Use standard chat completion with JSON mode as fallback
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            # Parse JSON response
            response_json = json.loads(completion.choices[0].message.content)

            # Determine if historical query
            is_historical = response_json.get("is_historical_query", False)

            # Parse time anchor
            time_anchor_str = response_json.get("time_anchor", "")
            if time_anchor_str:
                try:
                    # Try to parse the provided date
                    time_anchor = pd.to_datetime(time_anchor_str).isoformat()
                except:
                    time_anchor = datetime.now().isoformat()
            else:
                time_anchor = datetime.now().isoformat()

            # Create ParsedQuery object
            result = ParsedQuery(
                time_anchor=time_anchor,
                event_type=response_json.get("event_type", ""),
                intensity=float(response_json.get("intensity", 0.0)),
                actor_subj=response_json.get("actor_subj", ""),
                recipient_obj=response_json.get("recipient_obj", ""),
                obj_state=response_json.get("obj_state", []),
                context_summary=response_json.get("context_summary", user_query),
                reasoning_trace=response_json.get("reasoning_trace", ""),
                prediction_horizon_days=prediction_horizon_days,
                is_historical_query=is_historical,
                reference_event_description=response_json.get("reference_event_description", None)
            )

            if is_historical:
                print(f"Historical query parsed: Event on {time_anchor[:10]}, looking {prediction_horizon_days} days ahead")
            else:
                print(f"Future query parsed: {result.event_type or 'general'} involving {result.actor_subj}")

            return result

        except Exception as e:
            print(f"Query parsing failed: {e}")
            # Return default query with user-provided info
            return ParsedQuery(
                time_anchor=datetime.now().isoformat(),
                event_type="",
                intensity=0.0,
                actor_subj="",
                recipient_obj="",
                obj_state=[],
                context_summary=user_query,
                reasoning_trace=f"Parsing failed: {str(e)}",
                prediction_horizon_days=prediction_horizon_days,
                is_historical_query=False,
                reference_event_description=None
            )


# ============================================================================
# 4. CONTEXT RETRIEVAL AGENT
# ============================================================================

class ContextRetrievalAgent:
    """
    Retrieves relevant historical events from TKG based on parsed query.
    Supports iterative refinement with Query Parser.
    """

    # Location name to code mapping
    LOCATION_MAPPINGS = {
        "uttar pradesh": "UP",
        "delhi": "DL",
        "maharashtra": "MH",
        "west bengal": "WB",
        "punjab": "PB",
        "haryana": "HR",
        "gujarat": "GJ",
        "madhya pradesh": "MP",
        "karnataka": "KA",
        "bihar": "BR",
        "rajasthan": "RJ",
        "tamil nadu": "TN"
    }

    def __init__(self, tkg: nx.DiGraph, event_list: List[CausalMemoryEvent]):
        self.tkg = tkg
        self.event_list = event_list
        self.search_index = TKGSearchIndex(event_list)
        print("ContextRetrievalAgent initialized with TKG and Search Index.")

    def _normalize_location(self, location: str) -> str:
        """Normalize location names to state codes."""
        if not location:
            return ""

        # Try exact match first
        loc_lower = location.lower().strip()
        if loc_lower in self.LOCATION_MAPPINGS:
            return self.LOCATION_MAPPINGS[loc_lower]

        # If already a code (2 letters), return as-is
        if len(location) == 2:
            return location.upper()

        # Return original
        return location

    def _match_event_type(self, query_type: str, actual_type: str) -> bool:
        """
        Flexible event type matching that handles word variations.

        Examples:
            'accusation' matches 'Accuse'
            'arrested' matches 'Arrest'
            'sanction' matches 'Sanction'
        """
        query_lower = query_type.lower().strip()
        actual_lower = actual_type.lower().strip()

        # Exact match
        if query_lower == actual_lower:
            return True

        # Substring match (bidirectional)
        if query_lower in actual_lower or actual_lower in query_lower:
            return True

        # Word stem matching for common variations
        # Map query words to their canonical event types
        event_type_mappings = {
            'accuse': 'accuse',
            'accusation': 'accuse',
            'accused': 'accuse',
            'arrest': 'arrest',
            'arrested': 'arrest',
            'arresting': 'arrest',
            'sanction': 'sanction',
            'sanctioned': 'sanction',
            'sanctioning': 'sanction',
            'protest': 'protest',
            'protested': 'protest',
            'protesting': 'protest',
            'demand': 'demand',
            'demanded': 'demand',
            'demanding': 'demand',
            'request': 'request',
            'requested': 'request',
            'requesting': 'request',
            'response': 'response',
            'respond': 'response',
            'responded': 'response',
        }

        # Extract first word from query type (e.g., "accusation regarding flood" -> "accusation")
        query_first_word = query_lower.split()[0] if query_lower else ''
        query_stem = event_type_mappings.get(query_first_word, query_first_word)
        actual_stem = event_type_mappings.get(actual_lower, actual_lower)

        return query_stem == actual_stem

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        policy: str = 'baseline'
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """
        Retrieve historical context based on parsed query.

        Args:
            parsed_query: Structured query from QueryParserAgent
            policy: Retrieval policy ('baseline', 'intensity_aware', 'actor_history')

        Returns:
            (formatted_context, target_event_id, retrieval_stats)
        """
        # Extract search parameters
        actors = [parsed_query.actor_subj, parsed_query.recipient_obj]
        actors = [a for a in actors if a]  # Remove empty strings

        # Normalize location
        raw_location = parsed_query.obj_state[0] if parsed_query.obj_state else ""
        location = self._normalize_location(raw_location)

        # Parse time anchor
        time_anchor = pd.to_datetime(parsed_query.time_anchor)
        if time_anchor.tz is not None:
            time_anchor = time_anchor.tz_localize(None)

        # Different logic for historical vs future queries
        if parsed_query.is_historical_query:
            # For historical queries: Find events matching the reference event
            # on or near the specified date
            date_str = time_anchor.strftime('%Y-%m-%d')
            date_matches = set(self.search_index.search_events(query_type='date', query_value=date_str))

            matching_ids = list(date_matches)

            # Filter by event type if specified (flexible matching with word stems)
            if parsed_query.event_type:
                matching_ids = [
                    e.event_id for e in self.event_list
                    if e.event_id in matching_ids and
                    self._match_event_type(parsed_query.event_type, e.event_type)
                ]

            # Filter by recipient (for historical queries, often the key identifier)
            if parsed_query.recipient_obj:
                recipient_matches = set()
                recipient_matches.update(
                    self.search_index.search_events(query_type='actor', query_value=parsed_query.recipient_obj)
                )
                matching_ids = list(set(matching_ids).intersection(recipient_matches))

            # Filter by actors if specified
            if actors:
                # For historical queries, be more flexible with actor matching
                # Search for actor names and also check actor titles in events
                actor_matches = set()
                for actor in actors:
                    # Clean actor name (remove "The", etc.)
                    clean_actor = actor.replace("The ", "").replace("the ", "").strip()

                    # Try exact match first
                    matches = set(self.search_index.search_events(query_type='actor', query_value=actor))

                    # If no exact match, try cleaned version
                    if not matches:
                        matches = set(self.search_index.search_events(query_type='actor', query_value=clean_actor))

                    # Still no match? Try partial match
                    if not matches:
                        # Try partial match (e.g., "Supreme Court" matches "Court Supreme")
                        actor_lower = actor.lower()
                        clean_lower = clean_actor.lower()
                        for event in self.event_list:
                            if (actor_lower in event.actor_subj.lower() or actor_lower in event.recipient_obj.lower() or
                                clean_lower in event.actor_subj.lower() or clean_lower in event.recipient_obj.lower()):
                                matches.add(event.event_id)

                    actor_matches.update(matches)

                if actor_matches:
                    matching_ids = list(set(matching_ids).intersection(actor_matches))

            # Filter by location if specified
            if location:
                loc_matches = set(
                    self.search_index.search_events(query_type='location', query_value=location)
                )
                matching_ids = list(set(matching_ids).intersection(loc_matches))

        else:
            # For future queries: Find events before time anchor
            matching_ids = [e.event_id for e in self.event_list]

            matching_ids = [
                e.event_id for e in self.event_list
                if e.time_anchor <= time_anchor and e.event_id in matching_ids
            ]

            # Filter by actors if specified
            if actors:
                actor_matches = set()
                for actor in actors:
                    actor_matches.update(
                        self.search_index.search_events(query_type='actor', query_value=actor)
                    )
                matching_ids = list(set(matching_ids).intersection(actor_matches))

            # Filter by location if specified
            if location:
                loc_matches = set(
                    self.search_index.search_events(query_type='location', query_value=location)
                )
                matching_ids = list(set(matching_ids).intersection(loc_matches))

        retrieval_stats = {
            "total_events_found": len(matching_ids),
            "query_id": str(uuid.uuid4()),
            "retrieval_timestamp": datetime.now().isoformat(),
            "is_historical_query": parsed_query.is_historical_query
        }

        if not matching_ids:
            print(f"WARNING: No events found matching criteria for policy {policy}")
            return None, None, retrieval_stats

        # Find the target event
        matched_events = [e for e in self.event_list if e.event_id in matching_ids]

        if parsed_query.is_historical_query:
            # For historical queries, use the event closest to the reference date
            matched_events.sort(key=lambda x: abs((x.time_anchor - time_anchor).total_seconds()))
        else:
            # For future queries, use the most recent event
            matched_events.sort(key=lambda x: x.time_anchor, reverse=True)

        target_event_id = matched_events[0].event_id

        # Use existing retrieve_context function from memory_module
        formatted_context = retrieve_context(
            self.tkg,
            target_event_id,
            policy=policy,
            time_window_days=300
        )

        retrieval_stats["target_event_id"] = target_event_id
        retrieval_stats["context_events_count"] = len(formatted_context.split("---"))

        if parsed_query.is_historical_query:
            target_event = matched_events[0]
            print(f"Historical reference event found: {target_event.event_type} on {target_event.time_anchor.strftime('%Y-%m-%d')}")
            print(f"Context retrieved: {retrieval_stats['context_events_count']} events for policy '{policy}'")
        else:
            print(f"Context retrieved: {retrieval_stats['context_events_count']} events for policy '{policy}'")

        return formatted_context, target_event_id, retrieval_stats


# ============================================================================
# 5. PREDICTION AGENT (Multi-Policy)
# ============================================================================

class PredictionAgent:
    """
    Generates event predictions using three different policies:
    1. Baseline (recency-based)
    2. Intensity-Aware (high-intensity cascades)
    3. Actor History (long-term behavioral patterns)
    """

    def __init__(self):
        self.client = OpenAI()
        print("PredictionAgent initialized for multi-policy predictions.")

    def predict_all_policies(
        self,
        parsed_query: ParsedQuery,
        contexts: Dict[str, str]  # {policy_name: formatted_context}
    ) -> List[PolicyPrediction]:
        """
        Generate predictions for all three policies.

        Args:
            parsed_query: Structured query
            contexts: Dict mapping policy names to their retrieved contexts

        Returns:
            List of PolicyPrediction objects (one per policy)
        """
        predictions = []

        for policy_name, context in contexts.items():
            if context is None or context == "":
                print(f"Skipping {policy_name}: no context available")
                continue

            prediction = self._predict_single_policy(
                policy_name,
                parsed_query,
                context
            )

            if prediction:
                predictions.append(prediction)

        return predictions

    def _predict_single_policy(
        self,
        policy_name: str,
        parsed_query: ParsedQuery,
        formatted_context: str
    ) -> Optional[PolicyPrediction]:
        """Generate prediction for a single policy."""

        # Policy-specific instructions
        policy_instructions = {
            "baseline": """Focus on RECENT events (last few entries in the context).
            Predict by extrapolating recent trends and patterns.
            High confidence if events show clear recent momentum.""",

            "intensity_aware": """Focus on HIGH-INTENSITY events (high absolute intensity values).
            Look for cascade patterns where intense events trigger follow-ups.
            Predict escalation or de-escalation based on intensity trends.""",

            "actor_history": """Focus on ACTOR BEHAVIORAL PATTERNS across the entire history.
            Identify what this actor typically does in similar situations.
            High confidence if actor has consistent historical behavior."""
        }

        system_instruction = f"""You are a geopolitical forecasting engine using the {policy_name} policy.

{policy_instructions.get(policy_name, '')}

Analyze the historical context and predict the immediate next event.
Your prediction must be realistic and directly connected to the context provided."""

        # Different prompts for historical vs future queries
        if parsed_query.is_historical_query:
            reference_date = pd.to_datetime(parsed_query.time_anchor).strftime('%Y-%m-%d')
            target_date = (pd.to_datetime(parsed_query.time_anchor) + timedelta(days=parsed_query.prediction_horizon_days)).strftime('%Y-%m-%d')

            user_prompt = f"""HISTORICAL ANALYSIS: Analyze what happened AFTER a reference event.

Reference Event: {parsed_query.reference_event_description}
Reference Date: {reference_date}
Time Period: {parsed_query.prediction_horizon_days} days after the reference event
Target Date Range: {reference_date} to {target_date}

Historical Context (events leading up to and including the reference event):
{formatted_context}

Based on the reference event and the historical context, predict what ACTUALLY HAPPENED in the {parsed_query.prediction_horizon_days} days following {reference_date}.

Return prediction as JSON:
1. event_type - Type of event that occurred
2. actors - List of key actors involved
3. location - Location
4. intensity - Event intensity (float)
5. description - Description of what happened next
6. confidence - Your confidence (0-1)
7. reasoning - Reasoning based on the pattern from the reference event
8. supporting_events - List of event IDs used

Return your answer in JSON format.
"""
        else:
            user_prompt = f"""FUTURE PREDICTION: Analyze historical context and predict the next event.

{formatted_context}

Query Context: {parsed_query.context_summary}
Prediction Target Date: {(pd.to_datetime(parsed_query.time_anchor) + timedelta(days=parsed_query.prediction_horizon_days)).strftime('%Y-%m-%d')}

Predict the next event and provide as JSON:
1. event_type - Type of event
2. actors - List of key actors involved
3. location - Location
4. intensity - Expected intensity (float)
5. description - Description
6. confidence - Your confidence (0-1)
7. reasoning - Reasoning based on the historical context
8. supporting_events - List of event IDs used

Return your answer in JSON format.
"""

        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )

            # Parse JSON response
            response_json = json.loads(completion.choices[0].message.content)

            # Extract prediction from response
            location_raw = response_json.get("location", "")
            # Ensure location is a string, not a list
            if isinstance(location_raw, list):
                location = location_raw[0] if location_raw else (parsed_query.obj_state[0] if parsed_query.obj_state else "")
            else:
                location = location_raw or (parsed_query.obj_state[0] if parsed_query.obj_state else "")

            actors_raw = response_json.get("actors", [])
            # Ensure actors is a list
            if isinstance(actors_raw, str):
                actors = [actors_raw]
            else:
                actors = actors_raw or [parsed_query.actor_subj]

            predicted_event = PredictedEvent(
                event_type=response_json.get("event_type", "Unknown"),
                predicted_timestamp=(
                    pd.to_datetime(parsed_query.time_anchor) +
                    timedelta(days=parsed_query.prediction_horizon_days)
                ).isoformat(),
                actors=actors,
                location=location,
                intensity=float(response_json.get("intensity", 0.0)),
                description=response_json.get("description", "")
            )

            result = PolicyPrediction(
                policy_name=policy_name,
                predicted_event=predicted_event,
                confidence_raw=float(response_json.get("confidence", 0.5)),
                supporting_events=response_json.get("supporting_events", []),
                reasoning=response_json.get("reasoning", "")
            )

            print(f"  {policy_name}: Predicted '{result.predicted_event.event_type}' (confidence: {result.confidence_raw:.2f})")

            return result

        except Exception as e:
            print(f"Prediction failed for {policy_name}: {e}")
            return None


# ============================================================================
# 6. RANKING AGENT
# ============================================================================

class RankingAgent:
    """
    Ranks and combines predictions based on:
    - Historical accuracy scores
    - Consensus detection (similar predictions)
    - Confidence boosting for multi-policy agreement
    """

    def __init__(self, historical_accuracy: Dict[str, Any] = None):
        self.historical_accuracy = historical_accuracy or HISTORICAL_ACCURACY
        print("RankingAgent initialized with historical accuracy database.")

    def rank_and_combine(
        self,
        predictions: List[PolicyPrediction],
        parsed_query: ParsedQuery
    ) -> List[RankedPrediction]:
        """
        Rank predictions and combine similar ones.

        Args:
            predictions: List of predictions from different policies
            parsed_query: Original parsed query for context

        Returns:
            Ranked list of combined predictions
        """
        if not predictions:
            return []

        # Step 1: Calculate confidence scores for each prediction
        scored_predictions = []
        for pred in predictions:
            confidence = self._calculate_final_confidence(
                pred,
                parsed_query
            )
            scored_predictions.append({
                "prediction": pred,
                "confidence": confidence
            })

        # Step 2: Detect similar predictions and combine them
        combined_predictions = self._combine_similar_predictions(
            scored_predictions,
            parsed_query
        )

        # Step 3: Sort by confidence (highest first)
        combined_predictions.sort(key=lambda x: x.confidence_score, reverse=True)

        print(f"Ranking complete: {len(combined_predictions)} final predictions")

        return combined_predictions

    def _calculate_final_confidence(
        self,
        prediction: PolicyPrediction,
        parsed_query: ParsedQuery
    ) -> float:
        """Calculate confidence score based on historical accuracy."""

        policy = prediction.policy_name
        event_type = prediction.predicted_event.event_type.lower()
        horizon = parsed_query.prediction_horizon_days

        # Get base accuracy
        base_accuracy = self.historical_accuracy[policy]["overall_accuracy"]

        # Get event-type-specific accuracy
        type_accuracy = self.historical_accuracy[policy]["by_event_type"].get(
            event_type,
            base_accuracy
        )

        # Get horizon-specific accuracy
        horizon_bucket = min([h for h in [7, 30, 90] if h >= horizon], default=90)
        horizon_accuracy = self.historical_accuracy[policy]["by_horizon"].get(
            horizon_bucket,
            base_accuracy
        )

        # Weighted average
        confidence = (
            0.4 * base_accuracy +
            0.3 * type_accuracy +
            0.3 * horizon_accuracy
        )

        # Apply policy's raw confidence as multiplier
        confidence *= prediction.confidence_raw

        return min(confidence, 1.0)

    def _are_predictions_similar(
        self,
        pred1: PolicyPrediction,
        pred2: PolicyPrediction
    ) -> bool:
        """Check if two predictions are similar enough to combine."""

        # Check actor overlap
        actors1 = set(pred1.predicted_event.actors)
        actors2 = set(pred2.predicted_event.actors)
        actor_overlap = len(actors1.intersection(actors2)) > 0

        # Check location overlap
        loc1 = pred1.predicted_event.location
        loc2 = pred2.predicted_event.location
        location_match = (loc1 == loc2) or (loc1 in loc2) or (loc2 in loc1)

        return actor_overlap or location_match

    def _combine_similar_predictions(
        self,
        scored_predictions: List[Dict[str, Any]],
        parsed_query: ParsedQuery
    ) -> List[RankedPrediction]:
        """Combine similar predictions and add consensus bonus."""

        combined_predictions = []
        processed = set()

        for i, pred_i in enumerate(scored_predictions):
            if i in processed:
                continue

            # Find all similar predictions
            similar_group = [pred_i]
            for j, pred_j in enumerate(scored_predictions[i+1:], start=i+1):
                if j in processed:
                    continue

                if self._are_predictions_similar(
                    pred_i["prediction"],
                    pred_j["prediction"]
                ):
                    similar_group.append(pred_j)
                    processed.add(j)

            # Combine or create single prediction
            if len(similar_group) > 1:
                combined = self._merge_predictions(similar_group)
            else:
                combined = self._create_single_prediction(pred_i)

            combined_predictions.append(combined)
            processed.add(i)

        return combined_predictions

    def _merge_predictions(
        self,
        similar_group: List[Dict[str, Any]]
    ) -> RankedPrediction:
        """Merge multiple similar predictions with consensus bonus."""

        predictions = [item["prediction"] for item in similar_group]
        confidences = [item["confidence"] for item in similar_group]

        # Take highest confidence prediction as base
        base_idx = confidences.index(max(confidences))
        base_event = predictions[base_idx].predicted_event

        # Merge actors from all predictions
        all_actors = set()
        for p in predictions:
            all_actors.update(p.predicted_event.actors)

        # Average intensity
        avg_intensity = sum(
            p.predicted_event.intensity for p in predictions
        ) / len(predictions)

        # Combined description
        combined_desc = f"{base_event.description} (Consensus from {len(predictions)} policies)"

        # Consensus bonus: +10% per additional agreeing policy
        base_confidence = max(confidences)
        consensus_bonus = 0.1 * (len(similar_group) - 1)
        combined_confidence = min(base_confidence + consensus_bonus, 1.0)

        # Generate explanation
        policy_names = [p.policy_name for p in predictions]
        explanation = f"Consensus prediction from {len(predictions)} policies: "
        explanation += ", ".join([
            f"{name} ({conf:.2%})"
            for name, conf in zip(policy_names, confidences)
        ])
        explanation += ". Agreement across multiple strategies increases confidence."

        return RankedPrediction(
            prediction_id=str(uuid.uuid4()),
            combined_prediction=PredictedEvent(
                event_type=base_event.event_type,
                predicted_timestamp=base_event.predicted_timestamp,
                actors=list(all_actors),
                location=base_event.location,
                intensity=avg_intensity,
                description=combined_desc
            ),
            confidence_score=combined_confidence,
            contributing_policies=policy_names,
            policy_confidences={
                predictions[i].policy_name: confidences[i]
                for i in range(len(predictions))
            },
            consensus_bonus=consensus_bonus,
            explanation=explanation
        )

    def _create_single_prediction(
        self,
        scored_pred: Dict[str, Any]
    ) -> RankedPrediction:
        """Create ranked prediction from single policy prediction."""

        pred = scored_pred["prediction"]
        confidence = scored_pred["confidence"]

        explanation = f"Prediction based on {pred.policy_name} policy (confidence: {confidence:.2%})"

        return RankedPrediction(
            prediction_id=str(uuid.uuid4()),
            combined_prediction=pred.predicted_event,
            confidence_score=confidence,
            contributing_policies=[pred.policy_name],
            policy_confidences={pred.policy_name: confidence},
            consensus_bonus=0.0,
            explanation=explanation
        )


# ============================================================================
# 7. OUTPUT FORMATTER
# ============================================================================

class OutputFormatter:
    """
    Formats ranked predictions into user-friendly output.
    Provides explanations and supporting evidence.
    """

    @staticmethod
    def format(
        ranked_predictions: List[RankedPrediction],
        verbose: bool = False,
        policy_predictions: Optional[List[PolicyPrediction]] = None
    ) -> str:
        """
        Format ranked predictions for display.

        Args:
            ranked_predictions: Final ranked and combined predictions
            verbose: If True, show individual policy predictions before ranked results
            policy_predictions: List of individual policy predictions (for verbose mode)

        Returns:
            Formatted string output
        """

        if not ranked_predictions:
            return "No predictions generated."

        output = []

        # Verbose mode: Show individual policy predictions first
        if verbose and policy_predictions:
            output.append("=" * 80)
            output.append("INDIVIDUAL POLICY PREDICTIONS")
            output.append("=" * 80)
            output.append("")

            for i, pred in enumerate(policy_predictions, 1):
                output.append(f"POLICY #{i}: {pred.policy_name.upper()}")
                output.append("-" * 80)
                output.append(f"Event Type: {pred.predicted_event.event_type}")
                output.append(f"Predicted Date: {pred.predicted_event.predicted_timestamp}")
                output.append(f"Actors: {', '.join(pred.predicted_event.actors)}")
                output.append(f"Location: {pred.predicted_event.location}")
                output.append(f"Intensity: {pred.predicted_event.intensity:.1f}")
                output.append(f"Description: {pred.predicted_event.description}")
                output.append(f"Raw Confidence: {pred.confidence_raw:.1%}")
                output.append("")
                output.append(f"Reasoning: {pred.reasoning}")
                output.append("")
                if pred.supporting_events:
                    output.append(f"Supporting Events: {', '.join(pred.supporting_events[:5])}")
                    if len(pred.supporting_events) > 5:
                        output.append(f"  ... and {len(pred.supporting_events) - 5} more")
                output.append("")

            output.append("")

        # Final ranked results
        output.append("=" * 80)
        output.append("FINAL RANKED PREDICTIONS")
        output.append("=" * 80)
        output.append("")

        for i, pred in enumerate(ranked_predictions, 1):
            output.append(f"PREDICTION #{i} (Confidence: {pred.confidence_score:.1%})")
            output.append("-" * 80)
            output.append(f"Event Type: {pred.combined_prediction.event_type}")
            output.append(f"Predicted Date: {pred.combined_prediction.predicted_timestamp}")
            output.append(f"Actors: {', '.join(pred.combined_prediction.actors)}")
            output.append(f"Location: {pred.combined_prediction.location}")
            output.append(f"Intensity: {pred.combined_prediction.intensity:.1f}")
            output.append(f"Description: {pred.combined_prediction.description}")
            output.append("")
            output.append(f"Contributing Policies: {', '.join(pred.contributing_policies)}")
            if pred.consensus_bonus > 0:
                output.append(f"Consensus Bonus: +{pred.consensus_bonus:.1%}")
            output.append(f"Explanation: {pred.explanation}")
            output.append("")

        return "\n".join(output)

    @staticmethod
    def format_json(ranked_predictions: List[RankedPrediction]) -> str:
        """Format as JSON for API responses."""
        return json.dumps(
            [pred.model_dump() for pred in ranked_predictions],
            indent=2
        )


# ============================================================================
# 8. ORCHESTRATOR AGENT
# ============================================================================

class OrchestratorAgent:
    """
    Central coordinator that manages the entire prediction workflow.
    Orchestrates all specialized agents.
    """

    def __init__(
        self,
        tkg: nx.DiGraph,
        event_list: List[CausalMemoryEvent]
    ):
        self.tkg = tkg
        self.event_list = event_list

        # Initialize all agents
        self.query_parser = QueryParserAgent()
        self.context_retriever = ContextRetrievalAgent(tkg, event_list)
        self.prediction_agent = PredictionAgent()
        self.ranking_agent = RankingAgent()
        self.output_formatter = OutputFormatter()

        print("OrchestratorAgent initialized with all sub-agents.")

    def orchestrate(
        self,
        user_query: str,
        prediction_horizon_days: int = 30,
        output_format: str = "text",  # "text" or "json"
        verbose: bool = False
    ) -> str:
        """
        Execute the complete prediction workflow.

        Args:
            user_query: Natural language query
            prediction_horizon_days: Days ahead to predict
            output_format: "text" or "json"
            verbose: If True, show individual policy predictions

        Returns:
            Formatted prediction results
        """
        print("\n" + "=" * 80)
        print("ORCHESTRATOR: Starting Prediction Workflow")
        print("=" * 80)

        # Step 1: Parse Query
        print("\n[Step 1/5] Query Parsing...")
        parsed_query = self.query_parser.parse(user_query, prediction_horizon_days)
        print(f"  Parsed: actor='{parsed_query.actor_subj}', location={parsed_query.obj_state}, event_type='{parsed_query.event_type}'")

        # Step 2: Retrieve Context for All Three Policies
        print("\n[Step 2/5] Context Retrieval (3 policies)...")
        contexts = {}
        for policy in ["baseline", "intensity_aware", "actor_history"]:
            context, target_id, stats = self.context_retriever.retrieve(
                parsed_query,
                policy=policy
            )
            contexts[policy] = context

        # Step 3: Generate Predictions Using All Policies
        print("\n[Step 3/5] Generating Predictions (3 policies)...")
        predictions = self.prediction_agent.predict_all_policies(
            parsed_query,
            contexts
        )

        if not predictions:
            return "ERROR: No predictions could be generated."

        # Step 4: Rank and Combine Predictions
        print("\n[Step 4/5] Ranking and Combining Predictions...")
        ranked_predictions = self.ranking_agent.rank_and_combine(
            predictions,
            parsed_query
        )

        # Step 5: Format Output
        print("\n[Step 5/5] Formatting Output...")
        if output_format == "json":
            result = self.output_formatter.format_json(ranked_predictions)
        else:
            result = self.output_formatter.format(
                ranked_predictions,
                verbose=verbose,
                policy_predictions=predictions
            )

        print("\nORCHESTRATOR: Workflow Complete")
        print("=" * 80)

        return result


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def load_data(csv_path: str, tkg_path: str) -> Tuple[nx.DiGraph, List[CausalMemoryEvent]]:
    """
    Load TKG and event list from files.

    Args:
        csv_path: Path to CSV file with event data
        tkg_path: Path to GML file with TKG

    Returns:
        Tuple of (TKG graph, event list)
    """
    print("Loading TKG and event data...")

    # Load events from CSV
    event_data_df = pd.read_csv(csv_path)
    from tkg_builder import convert_csv_to_events
    event_list = convert_csv_to_events(event_data_df)

    # Load TKG
    TKG = nx.read_gml(tkg_path)
    print(f"TKG loaded: {TKG.number_of_nodes()} nodes, {TKG.number_of_edges()} edges")
    print(f"Event list loaded: {len(event_list)} events")

    return TKG, event_list


def create_orchestrator(tkg: nx.DiGraph, event_list: List[CausalMemoryEvent]) -> OrchestratorAgent:
    """
    Create an orchestrator agent with pre-loaded data.

    Args:
        tkg: Pre-loaded Temporal Knowledge Graph
        event_list: Pre-loaded list of CausalMemoryEvent objects

    Returns:
        Initialized OrchestratorAgent
    """
    return OrchestratorAgent(tkg=tkg, event_list=event_list)


if __name__ == "__main__":
    # Load data once
    TKG, event_list = load_data(
        csv_path=r"C:\Users\Mitsy\itcs6190\TMS\data\IND_train.csv",
        tkg_path=r"C:\Users\Mitsy\itcs6190\TMS\tkg_memory_loc2.gml"
    )

    # Create orchestrator once
    orchestrator = create_orchestrator(tkg=TKG, event_list=event_list)

    # Now you can run multiple queries without reloading data
    print("\n" + "=" * 80)
    print("RUNNING MULTIPLE QUERIES")
    print("=" * 80)

    # Query 1: Historical event-based query (VERBOSE)
    print("\n--- QUERY 1: HISTORICAL EVENT-BASED (VERBOSE) ---")
    result1 = orchestrator.orchestrate(
        user_query="The Supreme Court accused the Government of Uttar Pradesh regarding healthcare-related issues on May 6, 2022. What were the next set of steps that happened after 2 months?",
        prediction_horizon_days=60,  # 2 months
        output_format="text",
        verbose=True
    )
    print("\n" + result1)

    # Query 2: Future prediction (CONCISE)
    print("\n\n--- QUERY 2: FUTURE PREDICTION (CONCISE) ---")
    result2 = orchestrator.orchestrate(
        user_query="What will Narendra Modi do in the next month in UP?",
        prediction_horizon_days=30,
        output_format="text",
        verbose=False
    )
    print("\n" + result2)
