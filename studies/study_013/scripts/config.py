import json
import re
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Dict, Any, List, Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from study_utils import BaseStudyConfig, PromptBuilder

import random


# Demographics distributions from Table 2 for persona generation
INDUSTRIES = [
    "Retail", "Manufacturing", "Wholesale", "Construction",
    "Transportation and Communication", "Financial", "Professional"
]

AGE_DISTRIBUTION = [
    (range(30, 40), 0.222),   # Less than 40
    (range(40, 61), 0.715),   # 40 to 60
    (range(61, 70), 0.063),   # More than 60
]

BUSINESS_SIZE_OPTIONS = [
    "Less than S$1m",
    "Between S$1m and S$25m",
    "Between S$25m and S$50m",
    "More than S$50m",
]

BUSINESS_SIZE_WEIGHTS = [0.028, 0.486, 0.444, 0.042]


def weighted_age_sample():
    """Sample an age from the Table 2 age distribution."""
    r = random.random()
    cumulative = 0
    for age_range, prob in AGE_DISTRIBUTION:
        cumulative += prob
        if r < cumulative:
            return random.choice(list(age_range))
    return random.randint(40, 60)


class CustomPromptBuilder(PromptBuilder):
    """Builds the full Keh, Foo & Lim (2002) questionnaire prompt."""

    def build_trial_prompt(self, trial_metadata):
        profile = trial_metadata.get("profile", {})
        items_a = trial_metadata.get("items_a", [])
        items_b = trial_metadata.get("items_b", [])
        items_c = trial_metadata.get("items_c", [])
        items_d = trial_metadata.get("items_d", [])
        vignette_text = trial_metadata.get("vignette_text", "")

        lines = []

        # --- Persona Introduction ---
        age = profile.get("age", 48)
        industry = profile.get("industry", "Manufacturing")
        business_size = profile.get("business_size", "Between S$1m and S$25m")
        years_exp = profile.get("years_experience", 15)
        founder = profile.get("is_founder", True)

        lines.append("You are participating in a research study on entrepreneurial decision-making.")
        lines.append(f"Imagine you are a {age}-year-old entrepreneur in Singapore who {'founded' if founder else 'acquired'} a {industry.lower()} business (annual revenue: {business_size}). You have {years_exp} years of business experience.")
        lines.append("Please answer all questions honestly based on your perspective as a business owner.\n")

        q_counter = 1

        # --- Section A: Risk Propensity (5 forced-choice items) ---
        lines.append("=" * 60)
        lines.append("SECTION A: RISK PREFERENCES")
        lines.append("=" * 60)
        lines.append("Please answer the following five items by choosing the alternative (\"a\" or \"b\") you would feel most comfortable with.\n")

        for item in items_a:
            options = item.get("options", [])
            lines.append(f"Q{q_counter}: Which would you prefer?")
            lines.append(f"  a) {options[0]}")
            lines.append(f"  b) {options[1]}")
            lines.append(f"  (Answer Q{q_counter}=a or Q{q_counter}=b)\n")
            item["q_idx"] = q_counter
            q_counter += 1

        # --- Section B: Cognitive Biases (7 Likert items) ---
        lines.append("=" * 60)
        lines.append("SECTION B: BUSINESS ATTITUDES")
        lines.append("=" * 60)
        lines.append("Please indicate how much you agree with each statement.")
        lines.append("Scale: 1 = Strongly Disagree, 2 = Disagree, 3 = Slightly Disagree, 4 = Neutral, 5 = Slightly Agree, 6 = Agree, 7 = Strongly Agree\n")

        for item in items_b:
            lines.append(f"Q{q_counter}: {item['question']}")
            lines.append(f"  (Answer Q{q_counter}=1 to Q{q_counter}=7)\n")
            item["q_idx"] = q_counter
            q_counter += 1

        # --- Section C: Overconfidence (10 confidence-interval items) ---
        lines.append("=" * 60)
        lines.append("SECTION C: GENERAL KNOWLEDGE")
        lines.append("=" * 60)
        lines.append("For each question below, provide a LOWER LIMIT and UPPER LIMIT such that you are 90% confident the correct answer falls within your range.")
        lines.append("If you have absolutely no idea, provide the widest reasonable range.\n")

        for item in items_c:
            unit = item.get("unit", "")
            lines.append(f"Q{q_counter} (Lower Limit) and Q{q_counter + 1} (Upper Limit): {item['question']}")
            lines.append(f"  Unit: {unit}")
            lines.append(f"  (Answer Q{q_counter}=<lower> Q{q_counter + 1}=<upper>)\n")
            item["q_idx_lower"] = q_counter
            item["q_idx_upper"] = q_counter + 1
            q_counter += 2

        # --- Section D: Case Vignette + Risk Perception + Opportunity Evaluation ---
        lines.append("=" * 60)
        lines.append("SECTION D: BUSINESS CASE EVALUATION")
        lines.append("=" * 60)
        lines.append("Please read the following case study carefully, then answer the questions.\n")
        lines.append(vignette_text)
        lines.append("")
        lines.append("Based on the case above, please indicate how much you agree with each statement.")
        lines.append("Scale: 1 = Strongly Disagree, 2 = Disagree, 3 = Slightly Disagree, 4 = Neutral, 5 = Slightly Agree, 6 = Agree, 7 = Strongly Agree\n")

        for item in items_d:
            if item["type"] == "likert_7":
                lines.append(f"Q{q_counter}: {item['question']}")
                lines.append(f"  (Answer Q{q_counter}=1 to Q{q_counter}=7)\n")
                item["q_idx"] = q_counter
                q_counter += 1
            elif item["type"] == "open_ended":
                lines.append(f"Q{q_counter}: {item['question']}")
                lines.append(f"  (Answer Q{q_counter}=<your brief response>)\n")
                item["q_idx"] = q_counter
                q_counter += 1

        # --- Response format ---
        lines.append("=" * 60)
        lines.append("RESPONSE FORMAT (MANDATORY)")
        lines.append("=" * 60)
        lines.append("Output ONLY answer lines in the format: Qk=<value>")
        lines.append("One answer per line. Do not include explanations.")
        lines.append(f"Expected number of answer lines: {q_counter - 1}")

        return "\n".join(lines)


class StudyStudy013Config(BaseStudyConfig):
    """Study config for Keh, Foo & Lim (2002) — Opportunity Evaluation under Risky Conditions."""

    prompt_builder_class = CustomPromptBuilder
    PROMPT_VARIANT = "v1"

    def create_trials(self, n_trials=None):
        spec = self.load_specification()
        n = n_trials if n_trials is not None else spec["participants"]["n"]

        # Load all materials
        mat_a = self.load_material("section_a_risk_propensity")
        mat_b = self.load_material("section_b_cognitive_biases")
        mat_c = self.load_material("section_c_overconfidence")
        mat_d = self.load_material("section_d_case_vignette")

        vignette_text = mat_d.get("vignette_text", "")

        trials = []
        for i in range(n):
            # Generate randomized entrepreneur profile from Table 2 demographics
            age = weighted_age_sample()
            industry = random.choice(INDUSTRIES)
            business_size = random.choices(BUSINESS_SIZE_OPTIONS, weights=BUSINESS_SIZE_WEIGHTS, k=1)[0]
            years_exp = max(3, age - random.randint(22, 30))
            is_founder = random.random() < 0.79

            profile = {
                "age": age,
                "industry": industry,
                "business_size": business_size,
                "years_experience": years_exp,
                "is_founder": is_founder,
            }

            # Deep copy items to avoid mutation across trials
            import copy
            trial = {
                "sub_study_id": "keh_foo_lim_opportunity_evaluation",
                "scenario_id": "mr_tan_vignette",
                "scenario": "mr_tan_vignette",
                "profile": profile,
                "items_a": copy.deepcopy(mat_a["items"]),
                "items_b": copy.deepcopy(mat_b["items"]),
                "items_c": copy.deepcopy(mat_c["items"]),
                "items_d": copy.deepcopy(mat_d["items"]),
                "vignette_text": vignette_text,
                "variant": self.PROMPT_VARIANT,
            }
            trials.append(trial)

        return trials

    def aggregate_results(self, raw_results):
        """Parse Qk=value responses and compute per-participant construct scores."""
        participants = []

        for record in raw_results.get("individual_data", []):
            trial_info = record.get("trial_info", {})
            response_text = record.get("response_text", "")

            # Parse Qk=Value
            responses = {}
            for line in response_text.split("\n"):
                match = re.match(r"Q(\d+)\s*[:=]\s*(.+)", line.strip())
                if match:
                    q_num = int(match.group(1))
                    responses[q_num] = match.group(2).strip()

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

            # --- Planning Fallacy: sum of B3 + B4 ---
            planning_fallacy = 0
            planning_count = 0
            for item in items_b:
                if item.get("metadata", {}).get("construct") == "planning_fallacy":
                    q_idx = item.get("q_idx")
                    if q_idx and q_idx in responses:
                        val = self.extract_numeric(responses[q_idx])
                        if 1 <= val <= 7:
                            planning_fallacy += val
                            planning_count += 1

            # --- Illusion of Control: sum of B5 + B6 + B7 ---
            illusion_of_control = 0
            ioc_count = 0
            for item in items_b:
                if item.get("metadata", {}).get("construct") == "illusion_of_control":
                    q_idx = item.get("q_idx")
                    if q_idx and q_idx in responses:
                        val = self.extract_numeric(responses[q_idx])
                        if 1 <= val <= 7:
                            illusion_of_control += val
                            ioc_count += 1

            # --- Overconfidence: count of items where correct answer is outside [lower, upper] ---
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

            # --- Risk Perception: sum of D1 + D2 + D3 + D4 ---
            risk_perception = 0
            rp_count = 0
            for item in items_d:
                if item.get("metadata", {}).get("construct") == "risk_perception":
                    q_idx = item.get("q_idx")
                    if q_idx and q_idx in responses:
                        val = self.extract_numeric(responses[q_idx])
                        if 1 <= val <= 7:
                            risk_perception += val
                            rp_count += 1

            # --- Opportunity Evaluation: sum of D5 + D6 + D7 ---
            opportunity_evaluation = 0
            oe_count = 0
            for item in items_d:
                if item.get("metadata", {}).get("construct") == "opportunity_evaluation":
                    q_idx = item.get("q_idx")
                    if q_idx and q_idx in responses:
                        val = self.extract_numeric(responses[q_idx])
                        if 1 <= val <= 7:
                            opportunity_evaluation += val
                            oe_count += 1

            # Only include participant if they have sufficient data
            if oc_count >= 5 and rp_count >= 3 and oe_count >= 2:
                participants.append({
                    "risk_propensity": risk_propensity,
                    "planning_fallacy": planning_fallacy,
                    "illusion_of_control": illusion_of_control,
                    "overconfidence": overconfidence,
                    "risk_perception": risk_perception,
                    "opportunity_evaluation": opportunity_evaluation,
                    "profile": trial_info.get("profile", {}),
                })

        # Compute descriptive statistics
        if not participants:
            return {"participants": [], "descriptive_statistics": {}, "n_valid": 0}

        constructs = ["risk_propensity", "planning_fallacy", "illusion_of_control",
                       "overconfidence", "risk_perception", "opportunity_evaluation"]

        desc_stats = {}
        for c in constructs:
            values = [p[c] for p in participants]
            desc_stats[c] = {
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "n": len(values),
            }

        return {
            "participants": participants,
            "descriptive_statistics": desc_stats,
            "n_valid": len(participants),
        }
