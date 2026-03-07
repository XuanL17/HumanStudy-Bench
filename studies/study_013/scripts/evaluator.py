import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from stats_lib import parse_p_value_from_reported

# Module-level cache for ground truth and metadata
_ground_truth_cache = None
_metadata_cache = None


def parse_agent_responses(response_text: str) -> Dict[int, str]:
    """
    Parse standardized responses: Qk=Value or Qk: Value
    Returns dict mapping question number (int) to value string.
    """
    parsed = {}
    matches = re.findall(r"Q(\d+)\s*[:=]\s*([^,\n]+)", response_text)
    for q_num, val in matches:
        parsed[int(q_num)] = val.strip()
    return parsed


def _expected_direction_to_int(expected_dir_str: str) -> int:
    """Convert expected_direction string to int: 1 = positive, -1 = negative, 0 = unknown."""
    if not expected_dir_str:
        return 0
    s = str(expected_dir_str).lower()
    if s in ("positive", "greater", ">"):
        return 1
    if s in ("negative", "less", "<"):
        return -1
    return 0


def extract_numeric(text: str, default: float = None):
    """Extract a numeric value from response text."""
    if text is None:
        return default
    match = re.search(r"(-?\d+\.?\d*)", str(text))
    return float(match.group(1)) if match else default


def compute_participant_scores(response_text: str, trial_info: Dict) -> Dict[str, Any]:
    """
    Parse a single participant's responses and compute all construct scores.
    Returns a dict of construct scores, or None if insufficient data.
    """
    responses = parse_agent_responses(response_text)
    items_a = trial_info.get("items_a", [])
    items_b = trial_info.get("items_b", [])
    items_c = trial_info.get("items_c", [])
    items_d = trial_info.get("items_d", [])

    # --- Risk Propensity: count of risky choices (0-5) ---
    risk_propensity = 0
    for item in items_a:
        q_idx = item.get("q_idx")
        if q_idx and q_idx in responses:
            choice = responses[q_idx].strip().lower()
            risky = item.get("metadata", {}).get("risky_option", "a")
            if choice == risky:
                risk_propensity += 1

    # --- Planning Fallacy: sum of planning fallacy items ---
    planning_fallacy = 0
    pf_count = 0
    for item in items_b:
        if item.get("metadata", {}).get("construct") == "planning_fallacy":
            q_idx = item.get("q_idx")
            if q_idx and q_idx in responses:
                val = extract_numeric(responses[q_idx])
                if val is not None and 1 <= val <= 7:
                    planning_fallacy += val
                    pf_count += 1

    # --- Illusion of Control: sum of IoC items ---
    illusion_of_control = 0
    ioc_count = 0
    for item in items_b:
        if item.get("metadata", {}).get("construct") == "illusion_of_control":
            q_idx = item.get("q_idx")
            if q_idx and q_idx in responses:
                val = extract_numeric(responses[q_idx])
                if val is not None and 1 <= val <= 7:
                    illusion_of_control += val
                    ioc_count += 1

    # --- Overconfidence: items where correct answer outside [lower, upper] ---
    overconfidence = 0
    oc_count = 0
    for item in items_c:
        q_lower = item.get("q_idx_lower")
        q_upper = item.get("q_idx_upper")
        correct = item.get("correct_answer")
        if q_lower and q_upper and correct is not None:
            if q_lower in responses and q_upper in responses:
                try:
                    lower = float(responses[q_lower])
                    upper = float(responses[q_upper])
                    oc_count += 1
                    if correct < lower or correct > upper:
                        overconfidence += 1
                except (ValueError, TypeError):
                    pass

    # --- Risk Perception: sum of risk perception items ---
    risk_perception = 0
    rp_count = 0
    for item in items_d:
        if item.get("metadata", {}).get("construct") == "risk_perception":
            q_idx = item.get("q_idx")
            if q_idx and q_idx in responses:
                val = extract_numeric(responses[q_idx])
                if val is not None and 1 <= val <= 7:
                    risk_perception += val
                    rp_count += 1

    # --- Opportunity Evaluation: sum of OE items ---
    opportunity_evaluation = 0
    oe_count = 0
    for item in items_d:
        if item.get("metadata", {}).get("construct") == "opportunity_evaluation":
            q_idx = item.get("q_idx")
            if q_idx and q_idx in responses:
                val = extract_numeric(responses[q_idx])
                if val is not None and 1 <= val <= 7:
                    opportunity_evaluation += val
                    oe_count += 1

    # Require minimum data completeness
    if oc_count < 5 or rp_count < 3 or oe_count < 2:
        return None

    return {
        "risk_propensity": risk_propensity,
        "planning_fallacy": planning_fallacy,
        "illusion_of_control": illusion_of_control,
        "overconfidence": overconfidence,
        "risk_perception": risk_perception,
        "opportunity_evaluation": opportunity_evaluation,
    }


def evaluate_study(results):
    """
    Evaluates the agent's performance on Study 013 (Keh, Foo & Lim 2002).
    Computes construct scores from agent responses and tests hypothesized relationships.
    Returns test_results with raw stats; no BF/PAS aggregation.
    """
    global _ground_truth_cache, _metadata_cache

    # 1. Load Ground Truth and Metadata (with caching)
    study_dir = Path(__file__).resolve().parent.parent / "source"

    if _ground_truth_cache is None:
        with open(study_dir / "ground_truth.json", "r") as f:
            _ground_truth_cache = json.load(f)

    if _metadata_cache is None:
        metadata_path = study_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                _metadata_cache = json.load(f)
        else:
            _metadata_cache = {}

    ground_truth = _ground_truth_cache
    sig_level = 0.05

    # 2. Parse all agent responses into participant score vectors
    participant_scores = []

    for participant in results.get("individual_data", []):
        for response in participant.get("responses", []):
            response_text = response.get("response_text", "")
            trial_info = response.get("trial_info", {})

            scores = compute_participant_scores(response_text, trial_info)
            if scores is not None:
                participant_scores.append(scores)

    # 3. Build test results for each finding
    test_results = []

    if not participant_scores:
        # Return empty results if no valid participants
        for study_gt in ground_truth.get("studies", []):
            for finding in study_gt.get("findings", []):
                test_results.append({
                    "study_id": "keh_foo_lim_opportunity_evaluation",
                    "finding_id": finding["finding_id"],
                    "n_agent": 0,
                    "error": "No valid participant data",
                })
        return {"test_results": test_results}

    # Extract score arrays
    overconfidence_scores = np.array([p["overconfidence"] for p in participant_scores])
    risk_perception_scores = np.array([p["risk_perception"] for p in participant_scores])
    opp_eval_scores = np.array([p["opportunity_evaluation"] for p in participant_scores])
    ioc_scores = np.array([p["illusion_of_control"] for p in participant_scores])
    planning_scores = np.array([p["planning_fallacy"] for p in participant_scores])
    risk_prop_scores = np.array([p["risk_propensity"] for p in participant_scores])

    n_agent = len(participant_scores)

    for study_gt in ground_truth.get("studies", []):
        for finding in study_gt.get("findings", []):
            finding_id = finding["finding_id"]
            stat_tests = finding.get("statistical_tests", [])
            test_gt = stat_tests[0] if stat_tests else {}
            expected_dir_str = test_gt.get("expected_direction", "")
            h_expected = _expected_direction_to_int(expected_dir_str)
            reported_stats = test_gt.get("reported_statistics", "")

            # Parse human p-value
            human_p_value = None
            human_significant = None
            parsed_p, parsed_sig, _ = parse_p_value_from_reported(reported_stats, sig_level)
            if parsed_p is not None:
                human_p_value = parsed_p
                human_significant = parsed_sig

            t_stat = None
            r_stat = None
            p_value = None
            agent_significant = None
            direction_match = None
            mean_agent = None
            sd_agent = None

            if finding_id == "F1":
                # Overconfidence: one-sample t-test against baseline of 1
                baseline = 1.0
                mean_agent = float(np.mean(overconfidence_scores))
                sd_agent = float(np.std(overconfidence_scores, ddof=1)) if n_agent > 1 else 0.0

                # For F1, human significance must be computed from reported stats
                # since the reported_statistics string has no p-value
                human_mean = 5.17
                human_sd = 2.64
                human_n = 77
                human_t = (human_mean - baseline) / (human_sd / np.sqrt(human_n))
                human_p_value = float(stats.t.sf(human_t, human_n - 1))
                human_significant = human_p_value < sig_level

                if sd_agent > 0 and n_agent >= 2:
                    t_stat_val, p_value_val = stats.ttest_1samp(overconfidence_scores, baseline)
                    t_stat = float(t_stat_val)
                    # One-sided test: mean > baseline
                    p_value = float(p_value_val / 2) if t_stat > 0 else float(1 - p_value_val / 2)
                    agent_significant = p_value < sig_level
                    direction_match = (mean_agent > baseline)

            elif finding_id in ("F2", "F3", "F4", "F5"):
                # Correlation-based findings
                corr_pairs = {
                    "F2": (risk_perception_scores, opp_eval_scores),
                    "F3": (ioc_scores, risk_perception_scores),
                    "F4": (ioc_scores, opp_eval_scores),
                    "F5": (overconfidence_scores, opp_eval_scores),
                }
                x_arr, y_arr = corr_pairs[finding_id]

                # Require sufficient data and non-constant arrays
                if (len(x_arr) >= 3 and
                    np.std(x_arr) > 0 and np.std(y_arr) > 0):
                    r_val, p_val = stats.pearsonr(x_arr, y_arr)
                    if not np.isnan(r_val):
                        r_stat = float(r_val)
                        p_value = float(p_val)
                        mean_agent = r_stat
                        agent_significant = p_value < sig_level
                        agent_dir = 1 if r_stat > 0 else -1
                        direction_match = (h_expected == 0) or (agent_dir == h_expected)

            # Compute replication metric
            replication = None
            if human_significant is not None and agent_significant is not None and direction_match is not None:
                replication = human_significant and agent_significant and direction_match

            test_result = {
                "study_id": "keh_foo_lim_opportunity_evaluation",
                "sub_study_id": "keh_foo_lim_opportunity_evaluation",
                "finding_id": finding_id,
                "n_agent": n_agent,
                "mean_agent": mean_agent,
                "sd_agent": sd_agent,
                "t_stat": t_stat,
                "r_stat": r_stat,
                "p_value": float(p_value) if p_value is not None else None,
                "significant": agent_significant,
                "direction_match": direction_match,
                "human_p_value": human_p_value,
                "human_significant": human_significant,
                "replication": replication,
            }
            test_results.append(test_result)

    return {"test_results": test_results}
