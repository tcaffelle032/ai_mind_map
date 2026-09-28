"""Checks for the teaching model's math and random-node behavior."""
import math
import unittest

from ai_mind_model import MAX_NODES, VectorModel


class ModelTests(unittest.TestCase):
    def test_exact_random_count_and_reproducibility(self):
        first = VectorModel(seed=13, preset="empty", random_nodes=137)
        second = VectorModel(seed=13, preset="empty", random_nodes=137)
        self.assertEqual(first.nodes, second.nodes)
        first.add_random(11)
        self.assertEqual(len(first.nodes), 148)
        self.assertEqual(len({n["id"] for n in first.nodes}), 148)
        self.assertEqual(len({n["label"] for n in first.nodes}), 148)
        self.assertTrue(all(-10 <= x <= 10 for n in first.nodes for x in n["xyz"]))
        self.assertNotEqual(second.nodes, VectorModel(seed=14, preset="empty", random_nodes=137).nodes)

    def test_character_coordinates_match_reference(self):
        model = VectorModel(preset="characters")
        self.assertEqual(model.resolve("Warrior")[0], [9, 1, 3])
        self.assertEqual(model.resolve("Barbarian")[0], [10, 0, 2])
        self.assertEqual(model.resolve("Wizard")[0], [2, 10, 1])
        model.add_random(100)
        self.assertTrue(all(0 <= x <= 10 for n in model.nodes for x in n["xyz"]))

    def test_vector_arithmetic(self):
        result, rows = VectorModel().evaluate({"text": "king - man + woman"})
        self.assertEqual(result["xyz"], [3, 4, 0])
        self.assertEqual(rows[0]["label"], "queen")
        self.assertEqual(rows[0]["distance"], 0)
        self.assertTrue(all(row["label"] not in {"king", "man", "woman"} for row in rows))

    def test_averaging(self):
        vector, excluded, _ = VectorModel().resolve("cat dog")
        self.assertEqual(vector, [-4.5, -5.25, -2.5])
        self.assertEqual(len(excluded), 2)

    def test_probabilities_normalize_and_temperature_changes_entropy(self):
        model = VectorModel()
        distributions = []
        for t in (0.1, 0.8, 3):
            result, rows = model.evaluate({"text": "apple", "temperature": t})
            probabilities = [r["probability"] for r in rows]
            self.assertAlmostEqual(sum(probabilities), 1)
            self.assertTrue(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities))
            self.assertAlmostEqual(result["visible_probability"], sum(probabilities[:8]))
            distributions.append(-sum(p * math.log(p) for p in probabilities if p))
        self.assertLess(distributions[0], distributions[1])
        self.assertLess(distributions[1], distributions[2])

    def test_cosine_uses_direction_not_distance(self):
        model = VectorModel(preset="empty")
        model.append_node("same_direction", [10, 0, 0], "People")
        model.append_node("nearby", [1, 1, 0], "People")
        cosine, _ = model.evaluate({"xyz": [1, 0, 0], "metric": "cosine"})
        distance, _ = model.evaluate({"xyz": [1, 0, 0], "metric": "distance"})
        self.assertEqual(cosine["candidates"][0]["label"], "same_direction")
        self.assertEqual(distance["candidates"][0]["label"], "nearby")
        with self.assertRaisesRegex(ValueError, "zero vector"):
            model.evaluate({"xyz": [0, 0, 0], "metric": "cosine"})

    def test_random_candidate_opt_in(self):
        model = VectorModel(random_nodes=100)
        normal, _ = model.evaluate({"text": "apple"})
        extended, _ = model.evaluate({"text": "apple", "include_random": True})
        self.assertEqual(extended["candidate_count"], normal["candidate_count"] + 100)

    def test_sampling_matches_distribution(self):
        model = VectorModel(preset="empty")
        model.append_node("one", [1, 0, 0], "People")
        model.append_node("two", [-1, 0, 0], "People")
        labels = [model.action({"action": "sample", "xyz": [0, 0, 0]})["chosen"]["label"] for _ in range(2000)]
        self.assertTrue(850 < labels.count("one") < 1150)
        self.assertEqual(set(labels), {"one", "two"})

    def test_validation_and_failed_reset_preserve_scene(self):
        model = VectorModel()
        old_nodes = list(model.nodes)
        for count in (-1, 1.5, True, MAX_NODES):
            with self.assertRaises(ValueError):
                model.add_random(count)
        with self.assertRaises(ValueError):
            model.reset("words", MAX_NODES)
        self.assertEqual(model.nodes, old_nodes)
        for text in ("unknown", "king +", "king ** queen", "!!!"):
            with self.assertRaises(ValueError):
                model.evaluate({"text": text})
        for vector in ([0, 0], [float("nan"), 0, 0], [True, 0, 0]):
            with self.assertRaises(ValueError):
                model.evaluate({"xyz": vector})

    def test_maximum_size_scene(self):
        model = VectorModel(preset="empty", random_nodes=MAX_NODES)
        result, rows = model.evaluate({"xyz": [0, 0, 0], "include_random": True})
        self.assertEqual(result["candidate_count"], MAX_NODES)
        self.assertAlmostEqual(sum(r["probability"] for r in rows), 1)
        with self.assertRaises(ValueError):
            model.add_random(1)


if __name__ == "__main__":
    unittest.main()
