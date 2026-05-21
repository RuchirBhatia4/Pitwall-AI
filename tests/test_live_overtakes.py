import unittest

from src.live.overtake_model import (
    estimate_freshness_advantage,
    estimate_overtake_probability,
    estimate_recovery_probability,
)


class TestLiveOvertakeModel(unittest.TestCase):
    def test_fresher_tires_increase_overtake_probability(self):
        fresher_advantage = estimate_freshness_advantage(
            attacker_compound="MEDIUM",
            attacker_tire_age=2,
            defender_compound="MEDIUM",
            defender_tire_age=16,
        )
        worn_advantage = estimate_freshness_advantage(
            attacker_compound="MEDIUM",
            attacker_tire_age=16,
            defender_compound="MEDIUM",
            defender_tire_age=2,
        )

        prob_fresher = estimate_overtake_probability(
            pace_delta_seconds=0.30,
            tire_age_delta=8.0,
            freshness_advantage=fresher_advantage,
            overtaking_difficulty=0.60,
            track_position_importance=0.60,
        )
        prob_worn = estimate_overtake_probability(
            pace_delta_seconds=0.30,
            tire_age_delta=-8.0,
            freshness_advantage=worn_advantage,
            overtaking_difficulty=0.60,
            track_position_importance=0.60,
        )

        self.assertGreater(prob_fresher, prob_worn)

    def test_higher_overtaking_difficulty_lowers_probability(self):
        easier_track = estimate_overtake_probability(
            pace_delta_seconds=0.25,
            tire_age_delta=5.0,
            freshness_advantage=0.25,
            overtaking_difficulty=0.30,
            track_position_importance=0.55,
        )
        harder_track = estimate_overtake_probability(
            pace_delta_seconds=0.25,
            tire_age_delta=5.0,
            freshness_advantage=0.25,
            overtaking_difficulty=0.95,
            track_position_importance=0.55,
        )

        self.assertGreater(easier_track, harder_track)

    def test_probabilities_stay_in_realistic_bounds(self):
        high = estimate_overtake_probability(
            pace_delta_seconds=3.0,
            tire_age_delta=40.0,
            freshness_advantage=2.0,
            overtaking_difficulty=0.0,
            track_position_importance=0.0,
        )
        low = estimate_overtake_probability(
            pace_delta_seconds=-3.0,
            tire_age_delta=-40.0,
            freshness_advantage=-2.0,
            overtaking_difficulty=1.0,
            track_position_importance=1.0,
        )

        self.assertGreaterEqual(high, 0.05)
        self.assertLessEqual(high, 0.95)
        self.assertGreaterEqual(low, 0.05)
        self.assertLessEqual(low, 0.95)

        recovery = estimate_recovery_probability(
            base_overtake_probability=high,
            expected_positions_to_recover=12,
            current_position=2,
            field_size=22,
            pace_delta_seconds=1.5,
            freshness_advantage=1.0,
            overtaking_difficulty=0.9,
            track_position_importance=0.9,
        )
        self.assertGreaterEqual(recovery, 0.05)
        self.assertLessEqual(recovery, 0.95)

    def test_strong_fresh_tire_advantage_improves_recovery_probability(self):
        high_freshness = estimate_freshness_advantage(
            attacker_compound="SOFT",
            attacker_tire_age=0,
            defender_compound="HARD",
            defender_tire_age=18,
        )
        low_freshness = estimate_freshness_advantage(
            attacker_compound="HARD",
            attacker_tire_age=12,
            defender_compound="SOFT",
            defender_tire_age=3,
        )

        high_base = estimate_overtake_probability(
            pace_delta_seconds=0.45,
            tire_age_delta=10.0,
            freshness_advantage=high_freshness,
            overtaking_difficulty=0.55,
            track_position_importance=0.55,
        )
        low_base = estimate_overtake_probability(
            pace_delta_seconds=0.10,
            tire_age_delta=-4.0,
            freshness_advantage=low_freshness,
            overtaking_difficulty=0.55,
            track_position_importance=0.55,
        )

        high_recovery = estimate_recovery_probability(
            base_overtake_probability=high_base,
            expected_positions_to_recover=5,
            current_position=4,
            field_size=22,
            pace_delta_seconds=0.45,
            freshness_advantage=high_freshness,
            overtaking_difficulty=0.55,
            track_position_importance=0.55,
        )
        low_recovery = estimate_recovery_probability(
            base_overtake_probability=low_base,
            expected_positions_to_recover=5,
            current_position=4,
            field_size=22,
            pace_delta_seconds=0.10,
            freshness_advantage=low_freshness,
            overtaking_difficulty=0.55,
            track_position_importance=0.55,
        )

        self.assertGreater(high_recovery, low_recovery)


if __name__ == "__main__":
    unittest.main()
