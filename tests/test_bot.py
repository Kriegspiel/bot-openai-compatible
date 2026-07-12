from __future__ import annotations

import json
import os
import threading
import time
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import bot


class BotTests(unittest.TestCase):
    def setUp(self) -> None:
        bot._LLM_PREFLIGHT_CACHE.update({"ready": None, "expires_at": 0.0, "reason": "unchecked"})
        bot._MODEL_AVAILABILITY_REPORT_CACHE.update({"ready": None, "reason": "", "reported_at": 0.0})
        bot.configure_model_call_semaphore(bot.DEFAULT_MAX_CONCURRENT_MODEL_CALLS)
        bot.load_ruleset_summary.cache_clear()
        bot.configure_runtime_paths()

    def tearDown(self) -> None:
        bot.configure_runtime_paths()

    def test_runtime_paths_isolate_instance_env_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_path = tmp_path / "model.env"
            state_path = tmp_path / "state" / "model.json"
            env_path.write_text("KRIEGSPIEL_BOT_USERNAME=modelbot\nLLM_MODEL=provider/model\n", encoding="utf-8")

            with mock.patch.dict("os.environ", {}, clear=True):
                bot.configure_runtime_paths(env_path=env_path, state_path=state_path)
                bot.load_env_file()
                bot.save_token("token-1")

                self.assertEqual(os.environ["KRIEGSPIEL_BOT_USERNAME"], "modelbot")
                self.assertEqual(os.environ["LLM_MODEL"], "provider/model")
                self.assertEqual(json.loads(state_path.read_text(encoding="utf-8"))["token"], "token-1")

    def test_normalize_ranked_decisions_filters_invalid_and_duplicates(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decisions = bot.normalize_ranked_decisions(
            {"m": ["E2E4", "e2e4", "a2a4", "ask_any"]},
            state,
        )
        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}, {"action": "ask_any", "uci": None}])

    def test_normalize_ranked_decisions_accepts_legacy_candidates(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decisions = bot.normalize_ranked_decisions(
            {
                "candidates": [
                    {"action": "move", "uci": "E2E4"},
                    {"action": "ask_any", "uci": None},
                ]
            },
            state,
        )
        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}, {"action": "ask_any", "uci": None}])

    def test_normalize_decision_accepts_legal_move(self) -> None:
        state = {"possible_actions": ["move", "ask_any"], "allowed_moves": ["e2e4", "d2d4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "E2E4"}, state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})

    def test_normalize_decision_rejects_illegal_move(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["e2e4"]}
        decision = bot.normalize_decision({"action": "move", "uci": "a2a4"}, state)
        self.assertIsNone(decision)

    def test_normalize_decision_accepts_ask_any_when_available(self) -> None:
        state = {"possible_actions": ["ask_any"], "allowed_moves": []}
        decision = bot.normalize_decision({"action": "ask_any", "uci": None}, state)
        self.assertEqual(decision, {"action": "ask_any", "uci": None})

    def test_extract_response_text_reads_chat_completion_message(self) -> None:
        payload = {"choices": [{"message": {"content": "{\"m\":[\"e2e4\"]}"}}]}
        self.assertEqual(
            bot.extract_response_text(payload),
            "{\"m\":[\"e2e4\"]}",
        )

    def test_extract_response_text_reads_reasoning_aliases(self) -> None:
        payload = {"choices": [{"message": {"content": "", "reasoning_content": "{\"m\":[\"e2e4\"]}"}}]}
        self.assertEqual(
            bot.extract_response_text(payload),
            "{\"m\":[\"e2e4\"]}",
        )

    def test_parse_model_decision_reads_openai_tool_call(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": bot.ACTION_SCHEMA_NAME,
                                    "arguments": "{\"m\":[\"e2e4\",\"ask_any\"]}",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        self.assertEqual(bot.parse_model_decision(payload), {"m": ["e2e4", "ask_any"]})

    def test_parse_model_decision_extracts_json_from_tool_call_arguments(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": bot.ACTION_SCHEMA_NAME,
                                    "arguments": "Here is the move list:\n{\"m\":[\"e2e4\"]}\nDone.",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        self.assertEqual(bot.parse_model_decision(payload), {"m": ["e2e4"]})

    def test_parse_model_decision_ignores_trailing_text_after_object(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "{\"m\":[\"e2e4\"]}\n\nI considered {a few} other moves."
                    }
                }
            ]
        }
        self.assertEqual(bot.parse_model_decision(payload), {"m": ["e2e4"]})

    def test_parse_model_decision_skips_invalid_braces_before_object(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": "Thinking in {loose notes first.\nFinal: {\"m\":[\"ask_any\"]}"
                    }
                }
            ]
        }
        self.assertEqual(bot.parse_model_decision(payload), {"m": ["ask_any"]})

    def test_parse_model_decision_rejects_non_object_tool_arguments(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": bot.ACTION_SCHEMA_NAME,
                                    "arguments": "[\"e2e4\"]",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "Tool arguments must decode to an object"):
            bot.parse_model_decision(payload)

    def test_extract_response_text_keeps_responses_fallback(self) -> None:
        payload = {"output": [{"content": [{"text": "{\"m\":[\"d2d4\"]}"}]}]}
        self.assertEqual(
            bot.extract_response_text(payload),
            "{\"m\":[\"d2d4\"]}",
        )

    def test_fallback_prefers_center_moves(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["a2a3", "e2e4", "h2h3"]}
        decision = bot.fallback_decision(state)
        self.assertEqual(decision, {"action": "move", "uci": "e2e4"})

    def test_fallback_ranked_actions_wraps_single_decision(self) -> None:
        state = {"possible_actions": ["move"], "allowed_moves": ["a2a3", "e2e4"]}
        decisions = bot.fallback_ranked_actions(state)
        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])

    def test_should_resign_for_move_limit_defaults_to_move_256(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(bot.should_resign_for_move_limit({"move_number": 255}))
            self.assertTrue(bot.should_resign_for_move_limit({"move_number": 256}))

    def test_should_resign_for_move_limit_uses_backend_ply_limit(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(bot.should_resign_for_move_limit({"move_number": 999, "ply_count": 127, "llm_bot_ply_limit": 128}))
            self.assertTrue(bot.should_resign_for_move_limit({"move_number": 2, "ply_count": 128, "llm_bot_ply_limit": 128}))

    def test_should_resign_for_move_limit_uses_legal_turns_for_backend_turn_limit(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(bot.should_resign_for_move_limit({"move_number": 255, "ply_count": 999, "llm_bot_turn_limit": 128}))
            self.assertTrue(bot.should_resign_for_move_limit({"move_number": 257, "ply_count": 128, "llm_bot_turn_limit": 128}))

    def test_should_resign_for_move_limit_does_not_let_null_turn_limit_shadow_ply_limit(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(
                bot.should_resign_for_move_limit(
                    {"move_number": 2, "ply_count": 128, "llm_bot_turn_limit": None, "llm_bot_ply_limit": 128}
                )
            )

    def test_should_resign_for_move_limit_disables_default_when_backend_is_unlimited(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(bot.should_resign_for_move_limit({"move_number": 999, "ply_count": 999, "llm_bot_turn_limit": None}))
            self.assertFalse(bot.should_resign_for_move_limit({"move_number": 999, "ply_count": 999, "llm_bot_ply_limit": None}))

    def test_maybe_play_game_resigns_at_move_limit_before_model_call(self) -> None:
        state = {
            "state": "active",
            "turn": "white",
            "your_color": "white",
            "move_number": 256,
        }
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(bot, "get_conversation_state", return_value={}):
                with mock.patch.object(bot, "get_json", side_effect=[{"rule_variant": "berkeley_any"}, state]):
                    with mock.patch.object(bot, "post_json", return_value={"result": {"winner": "black", "reason": "resignation"}}) as post_json:
                        with mock.patch.object(bot, "clear_conversation_state") as clear_conversation_state:
                            with mock.patch.object(bot, "choose_ranked_actions") as choose_ranked_actions:
                                self.assertTrue(bot.maybe_play_game("gid1"))

        post_json.assert_called_once_with("/game/gid1/resign")
        clear_conversation_state.assert_called_once_with("gid1")
        choose_ranked_actions.assert_not_called()

    def test_build_prompts_split_rules_and_state(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "state": "active",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "material_summary": {
                "white": {"pieces_remaining": 16, "pawns_captured": None},
                "black": {"pieces_remaining": 15, "pawns_captured": None},
            },
            "scoresheet": {
                "viewer_color": "white",
                "turns": [
                    {"turn": 1, "white": [{"move_uci": "e2e4", "message": "Move complete"}], "black": []},
                ],
            },
        }
        system_prompt = bot.build_system_prompt("berkeley_any")
        user_prompt = bot.build_initial_user_prompt(state)
        followup_prompt = bot.build_followup_user_prompt(
            state,
            feedback=["Rejected move e2e4: Illegal move"],
            exclude_actions=[{"action": "move", "uci": "e2e4"}],
            recent_updates=["Turn 3 black: Illegal move"],
        )
        self.assertIn("Berkeley + Any", system_prompt)
        self.assertNotIn("I. Introduction", system_prompt)
        self.assertIn("{\"m\":[\"e2e4\",\"d2d4\",\"ask_any\"]}", system_prompt)
        self.assertIn("Return minified JSON", system_prompt)
        self.assertIn("Turn JSON keys: c=your color", system_prompt)
        self.assertIn("\"fen\":\"fen\"", user_prompt)
        self.assertNotIn("private_board_fen", user_prompt)
        self.assertNotIn("rule_variant", user_prompt)
        self.assertNotIn("pawns_captured", user_prompt)
        self.assertIn("\"mat\":{\"b\":{\"pieces\":15},\"w\":{\"pieces\":16}}", user_prompt)
        self.assertIn("\"hist\":[{\"b\":[],\"n\":1,\"w\":[\"[e2e4] Move complete\"]}]", user_prompt)
        self.assertIn("Rejected move e2e4: Illegal move", followup_prompt)
        self.assertNotIn("Turn 3 black: Illegal move", followup_prompt)
        self.assertIn("\"rej\":[\"e2e4\"]", followup_prompt)

    def test_ruleset_summary_files_cover_supported_variants(self) -> None:
        summary_files = {path.stem for path in bot.RULESET_SUMMARY_DIR.glob("*.md")}
        self.assertEqual(summary_files, set(bot.SUPPORTED_RULE_VARIANTS))
        required_concepts = ["illegal", "capture", "check", "pawn", "promotion", "stalemate"]
        checklist_labels = [
            "Referee response to illegal tries:",
            "Capture announcements:",
            "Check announcements:",
            "Pawn-capture / Any? handling:",
            "Promotion announcements:",
            "Stalemate:",
        ]
        for variant in bot.SUPPORTED_RULE_VARIANTS:
            summary = bot.load_ruleset_summary(variant)
            normalized = summary.lower()
            self.assertGreaterEqual(len(summary.split()), 90)
            self.assertLessEqual(len(summary.split()), 190)
            self.assertFalse(any(line.startswith("- ") for line in summary.splitlines()))
            for label in checklist_labels:
                self.assertNotIn(label, summary)
            for concept in required_concepts:
                self.assertIn(concept, normalized)

        rand_summary = bot.load_ruleset_summary("rand")
        self.assertIn("promotions are announced in rand, but not the promoted piece type", rand_summary.lower())
        self.assertIn("the stalemated player loses in rand", rand_summary.lower())

    def test_build_system_prompt_uses_ruleset_summary_file(self) -> None:
        summary = bot.load_ruleset_summary("wild16")
        system_prompt = bot.build_system_prompt("wild16")
        self.assertIn("illegal tries private", summary)
        self.assertIn(summary, system_prompt)
        self.assertNotIn("Cincinnati", system_prompt)

    def test_turn_snapshot_requests_exact_batch_when_enough_actions_exist(self) -> None:
        allowed_moves = ["a2a3", "b2b3", "c2c3", "d2d3", "e2e3", "f2f3", "g2g3", "h2h3", "b1c3"]
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 1,
            "your_fen": "fen",
            "possible_actions": ["move", "ask_any"],
            "allowed_moves": allowed_moves,
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        payload = bot.build_turn_snapshot_payload(state)
        system_prompt = bot.build_system_prompt("berkeley_any")

        self.assertEqual(payload["n"], 10)
        self.assertIn("Return exactly n", system_prompt)
        self.assertNotIn("Return up to n", system_prompt)

    def test_llm_max_prompt_turns_has_ten_turn_floor(self) -> None:
        with mock.patch.dict("os.environ", {"LLM_MAX_PROMPT_TURNS": "5"}):
            self.assertEqual(bot.llm_max_prompt_turns(), 10)
        with mock.patch.dict("os.environ", {"LLM_MAX_PROMPT_TURNS": "12"}):
            self.assertEqual(bot.llm_max_prompt_turns(), 12)
        with mock.patch.dict("os.environ", {"LLM_MAX_PROMPT_TURNS": "invalid"}):
            self.assertEqual(bot.llm_max_prompt_turns(), 10)

    def test_max_concurrent_model_calls_parses_default_and_custom_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(bot.max_concurrent_model_calls(), 5)
        with mock.patch.dict("os.environ", {"LLM_BOT_MAX_CONCURRENT_MODEL_CALLS": "3"}):
            self.assertEqual(bot.max_concurrent_model_calls(), 3)
        with mock.patch.dict("os.environ", {"LLM_BOT_MAX_CONCURRENT_MODEL_CALLS": "0"}):
            self.assertEqual(bot.max_concurrent_model_calls(), 1)
        with mock.patch.dict("os.environ", {"LLM_BOT_MAX_CONCURRENT_MODEL_CALLS": "invalid"}):
            self.assertEqual(bot.max_concurrent_model_calls(), 5)

    def test_active_game_discovery_limit_parses_default_and_custom_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(bot.active_game_discovery_limit(), 100)
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT": "40"}):
            self.assertEqual(bot.active_game_discovery_limit(), 40)
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT": "0"}):
            self.assertEqual(bot.active_game_discovery_limit(), 1)
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT": "250"}):
            self.assertEqual(bot.active_game_discovery_limit(), 100)
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_ACTIVE_GAME_DISCOVERY_LIMIT": "invalid"}):
            self.assertEqual(bot.active_game_discovery_limit(), 100)

    def test_turn_snapshot_includes_at_least_ten_recent_turns_when_available(self) -> None:
        turns = [
            {
                "turn": turn_number,
                "white": [{"move_uci": f"a{turn_number}a{turn_number + 1}", "message": "Move complete"}],
                "black": [],
            }
            for turn_number in range(1, 13)
        ]
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 12,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "scoresheet": {"viewer_color": "white", "turns": turns},
        }

        with mock.patch.dict("os.environ", {"LLM_MAX_PROMPT_TURNS": "5"}):
            payload = bot.build_turn_snapshot_payload(state)

        self.assertEqual(len(payload["hist"]), 10)
        self.assertEqual(payload["hist"][0]["n"], 3)
        self.assertEqual(payload["hist"][-1]["n"], 12)

    def test_turn_snapshot_payload_includes_rule_specific_public_context(self) -> None:
        cincinnati_payload = bot.build_turn_snapshot_payload(
            {
                "rule_variant": "cincinnati",
                "your_color": "black",
                "turn": "black",
                "move_number": 8,
                "your_fen": "fen",
                "possible_actions": ["move"],
                "allowed_moves": ["g8f6"],
                "material_summary": {
                    "white": {"pieces_remaining": 16, "pawns_captured": 0},
                    "black": {"pieces_remaining": 15, "pawns_captured": 1},
                },
                "reserve_summary": {
                    "white": {"pawns": 0, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                    "black": {"pawns": 0, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                },
                "scoresheet": {"viewer_color": "black", "turns": []},
            }
        )
        self.assertEqual(
            cincinnati_payload["mat"],
            {
                "w": {"pieces": 16, "pawns_captured": 0},
                "b": {"pieces": 15, "pawns_captured": 1},
            },
        )
        self.assertNotIn("res", cincinnati_payload)

        crazy_payload = bot.build_turn_snapshot_payload(
            {
                "rule_variant": "crazykrieg",
                "your_color": "white",
                "turn": "white",
                "move_number": 9,
                "your_fen": "fen",
                "possible_actions": ["move", "ask_any"],
                "allowed_moves": ["g1f3"],
                "material_summary": {
                    "white": {"pieces_remaining": 17, "pawns_captured": None},
                    "black": {"pieces_remaining": 15, "pawns_captured": None},
                },
                "reserve_summary": {
                    "white": {"pawns": 0, "knights": 1, "bishops": 0, "rooks": 0, "queens": 0},
                    "black": {"pawns": 1, "knights": 0, "bishops": 0, "rooks": 0, "queens": 0},
                },
                "scoresheet": {"viewer_color": "white", "turns": []},
            }
        )
        self.assertEqual(
            crazy_payload["mat"],
            {"w": {"pieces": 17}, "b": {"pieces": 15}},
        )
        self.assertEqual(crazy_payload["res"]["w"]["knights"], 1)
        self.assertEqual(crazy_payload["res"]["b"]["pawns"], 1)

    def test_supported_rule_variants_default_to_all_playable_rulesets(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(bot.supported_rule_variants(), list(bot.SUPPORTED_RULE_VARIANTS))

    def test_supported_rule_variants_expands_legacy_two_ruleset_default(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "berkeley,berkeley_any"}):
            self.assertEqual(bot.supported_rule_variants(), list(bot.SUPPORTED_RULE_VARIANTS))

    def test_supported_rule_variants_respects_non_legacy_custom_subset(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "wild16,crazykrieg"}):
            self.assertEqual(bot.supported_rule_variants(), ["wild16", "crazykrieg"])

    def test_sync_bot_profile_posts_supported_rule_variants(self) -> None:
        with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post_json:
            self.assertTrue(bot.sync_bot_profile())

        post_json.assert_called_once_with(
            "/bots/profile",
            {"supported_rule_variants": list(bot.SUPPORTED_RULE_VARIANTS)},
        )

    def test_action_schema_uses_compact_move_list(self) -> None:
        schema = bot.action_schema()["schema"]
        self.assertEqual(schema["required"], ["m"])
        self.assertNotIn("candidates", schema["properties"])
        self.assertEqual(schema["properties"]["m"]["items"], {"type": "string"})

    def test_llm_usage_cost_usd_accounts_for_cached_input(self) -> None:
        usage = {
            "prompt_tokens": 1000,
            "prompt_tokens_details": {"cached_tokens": 400},
            "completion_tokens": 20,
        }

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_INPUT_USD_PER_MILLION_TOKENS": "0.20",
                "LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS": "0.02",
                "LLM_OUTPUT_USD_PER_MILLION_TOKENS": "1.25",
            },
        ):
            self.assertAlmostEqual(bot.llm_usage_cost_usd(usage), 0.000153)

    def test_log_llm_usage_reports_backend_usage_when_token_configured(self) -> None:
        payload = {
            "id": "chatcmpl_1",
            "usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 400}, "completion_tokens": 20},
        }

        with mock.patch.dict(
            "os.environ",
            {
                "KRIEGSPIEL_BOT_TOKEN": "token",
                "LLM_INPUT_USD_PER_MILLION_TOKENS": "0.20",
                "LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS": "0.02",
                "LLM_OUTPUT_USD_PER_MILLION_TOKENS": "1.25",
            },
            clear=False,
        ):
            with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post_json:
                bot.log_llm_usage(game_id="gid1", model="provider/model", payload=payload)
                expected_cost = bot.llm_usage_cost_usd(payload["usage"])

        post_json.assert_called_once()
        path, usage_payload = post_json.call_args.args
        self.assertEqual(path, "/bots/usage")
        self.assertEqual(
            usage_payload,
            {
                "game_id": "gid1",
                "provider": "openrouter",
                "model": "provider/model",
                "response_id": "chatcmpl_1",
                "input_tokens": 1000,
                "cached_input_tokens": 400,
                "output_tokens": 20,
                "total_tokens": 1020,
                "cost_usd": expected_cost,
            },
        )

    def test_choose_ranked_actions_is_stateless(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "material_summary": {
                "white": {"pieces_remaining": 16, "pawns_captured": None},
                "black": {"pieces_remaining": 16, "pawns_captured": None},
            },
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        raw_response = {
            "id": "chatcmpl_1",
            "choices": [{"message": {"content": json.dumps({"m": ["e2e4"]})}}],
            "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 50}, "completion_tokens": 20},
        }
        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_MODEL": "provider/model"}, clear=False):
            with mock.patch.object(bot, "call_llm", return_value=raw_response) as call_llm:
                decisions, source, response_id = bot.choose_ranked_actions(
                    state,
                    game_id="gid1",
                    conversation={"response_id": "old-response"},
                )

        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])
        self.assertEqual(source, "model")
        self.assertIsNone(response_id)
        self.assertNotIn("previous_response_id", call_llm.call_args.kwargs)
        self.assertNotIn("prompt_cache_key", call_llm.call_args.kwargs)
        self.assertIn("\"fen\":\"fen\"", call_llm.call_args.kwargs["user_prompt"])
        self.assertNotIn("private_board_fen", call_llm.call_args.kwargs["user_prompt"])

    def test_choose_ranked_actions_logs_usage_before_parse_failure(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "scoresheet": {"viewer_color": "white", "turns": []},
        }
        raw_response = {
            "id": "chatcmpl_1",
            "choices": [{"message": {"content": "not json"}}],
            "usage": {
                "prompt_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 400},
                "completion_tokens": 20,
            },
        }

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "provider/model",
                "LLM_INPUT_USD_PER_MILLION_TOKENS": "0.20",
                "LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS": "0.02",
                "LLM_OUTPUT_USD_PER_MILLION_TOKENS": "1.25",
            },
            clear=False,
        ):
            with mock.patch.object(bot, "call_llm", return_value=raw_response):
                with self.assertLogs(bot.logger, level="INFO") as logs:
                    decisions, source, response_id = bot.choose_ranked_actions(state, game_id="gid1")

        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])
        self.assertEqual(source, "fallback")
        self.assertIsNone(response_id)
        log_output = "\n".join(logs.output)
        self.assertIn("provider=openrouter", log_output)
        self.assertIn("response_id=chatcmpl_1", log_output)
        self.assertIn("input_tokens=1000", log_output)
        self.assertIn("cached_input_tokens=400", log_output)
        self.assertIn("output_tokens=20", log_output)
        self.assertIn("cost_usd=0.000153", log_output)

    def test_choose_ranked_actions_falls_back_when_model_request_fails(self) -> None:
        state = {
            "rule_variant": "berkeley_any",
            "your_color": "white",
            "turn": "white",
            "move_number": 3,
            "your_fen": "fen",
            "possible_actions": ["move"],
            "allowed_moves": ["e2e4"],
            "scoresheet": {"viewer_color": "white", "turns": []},
        }

        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_MODEL": "provider/model"}, clear=False):
            with mock.patch.object(bot, "call_llm", side_effect=bot.requests.RequestException("boom")):
                decisions, source, response_id = bot.choose_ranked_actions(state, game_id="gid1")

        self.assertEqual(decisions, [{"action": "move", "uci": "e2e4"}])
        self.assertEqual(source, "fallback")
        self.assertIsNone(response_id)

    def test_new_recent_items_returns_only_suffix_delta(self) -> None:
        previous = ["Turn 1 white: Move complete", "Turn 1 black: Illegal move"]
        current = [
            "Turn 1 white: Move complete",
            "Turn 1 black: Illegal move",
            "Turn 2 white: Has pawn captures",
        ]
        self.assertEqual(bot.new_recent_items(previous, current), ["Turn 2 white: Has pawn captures"])

    def test_scoresheet_digest_changes_with_content(self) -> None:
        digest_a = bot.scoresheet_digest({"viewer_color": "white", "turns": []})
        digest_b = bot.scoresheet_digest({"viewer_color": "white", "turns": [{"turn": 1, "white": [], "black": []}]})
        self.assertNotEqual(digest_a, digest_b)

    def test_summarize_scoresheet_turns_returns_lines(self) -> None:
        lines = bot.summarize_scoresheet_turns(
            {
                "viewer_color": "white",
                "turns": [{"turn": 1, "white": [{"message": "Move attempt — Move complete"}], "black": []}],
            },
            max_turns=12,
        )
        self.assertTrue(isinstance(lines, list))
        self.assertTrue(any("Move attempt" in line for line in lines))

    def test_should_create_lobby_game_when_under_cap_and_no_waiting_game(self) -> None:
        self.assertFalse(bot.should_create_lobby_game([]))
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME": "true"}):
            self.assertTrue(bot.should_create_lobby_game([]))
            self.assertFalse(bot.should_create_lobby_game([{"state": "active"}]))

    def test_should_not_create_lobby_game_when_waiting_game_exists(self) -> None:
        games = [{"state": "active"}, {"state": "waiting"}]
        self.assertFalse(bot.should_create_lobby_game(games))

    def test_should_not_create_lobby_game_when_active_cap_is_reached(self) -> None:
        games = [{"state": "active"} for _ in range(5)]
        self.assertFalse(bot.should_create_lobby_game(games))

    def test_lobby_create_cooldown_uses_tier_and_override(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_LLM_BOT_TIER": "T4"}, clear=False):
            self.assertEqual(bot.lobby_create_cooldown_seconds(), 21600)
        with mock.patch.dict(
            "os.environ",
            {"KRIEGSPIEL_LLM_BOT_TIER": "T4", "KRIEGSPIEL_AUTO_CREATE_COOLDOWN_SECONDS": "7200"},
            clear=False,
        ):
            self.assertEqual(bot.lobby_create_cooldown_seconds(), 7200)

    def test_can_create_lobby_game_uses_create_cooldown(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_LLM_BOT_TIER": "T3"}, clear=False):
            with mock.patch.object(bot, "load_state", return_value={"last_lobby_game_created_at": 100}):
                self.assertFalse(bot.can_create_lobby_game(now=10899))
                self.assertTrue(bot.can_create_lobby_game(now=10900))

    def test_maybe_create_lobby_game_records_creation(self) -> None:
        saved_states = []
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_AUTO_CREATE_LOBBY_GAME": "true"}, clear=False):
            with mock.patch.object(bot, "load_state", return_value={}):
                with mock.patch.object(bot, "save_state", side_effect=saved_states.append):
                    with mock.patch.object(bot.time, "time", return_value=4000.0):
                        with mock.patch.object(bot, "get_json", return_value={"games": []}):
                            with mock.patch.object(
                                bot,
                                "post_json",
                                return_value={"game_id": "g1", "game_code": "ABC123"},
                            ) as post_json:
                                self.assertTrue(bot.maybe_create_lobby_game([]))

        post_json.assert_called_once_with("/game/create", bot.create_payload())
        self.assertEqual(saved_states[-1]["last_lobby_game_created_at"], 4000.0)

    def test_open_bot_lobby_candidates_only_include_other_bot_waiting_games(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "openrouterbot"}):
            candidates = bot.open_bot_lobby_candidates(
                [
                    {
                        "game_code": "BOT123",
                        "created_by": "randobot",
                        "rule_variant": "berkeley_any",
                    },
                    {
                        "game_code": "SELF12",
                        "created_by": "openrouterbot",
                        "rule_variant": "berkeley_any",
                    },
                    {
                        "game_code": "HUM123",
                        "created_by": "fil",
                        "rule_variant": "berkeley_any",
                    },
                ],
                profile_lookup=lambda username: {"role": "bot" if username == "randobot" else "user"},
            )

        self.assertEqual([game["game_code"] for game in candidates], ["BOT123"])

    def test_open_bot_lobby_candidates_respect_supported_rule_variants(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "openrouterbot", "KRIEGSPIEL_SUPPORTED_RULE_VARIANTS": "berkeley,berkeley_any"}):
            candidates = bot.open_bot_lobby_candidates(
                [
                    {"game_code": "BER123", "created_by": "randobot", "rule_variant": "berkeley"},
                    {"game_code": "ANY123", "created_by": "randobot", "rule_variant": "berkeley_any"},
                    {"game_code": "CIN123", "created_by": "randobot", "rule_variant": "cincinnati"},
                ],
                profile_lookup=lambda username: {"role": "bot"},
            )

        self.assertEqual([game["game_code"] for game in candidates], ["BER123", "ANY123", "CIN123"])

    def test_choose_bot_game_to_join_returns_candidate(self) -> None:
        games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "openrouterbot"}):
            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    self.assertEqual(bot.choose_bot_game_to_join(games, rng=bot.random)["game_code"], "BOT123")

    def test_maybe_join_bot_lobby_game_samples_once_per_cooldown_window(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
            with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                    saved_states = []
                    with mock.patch.object(bot, "save_state", side_effect=saved_states.append):
                        with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                            with mock.patch.object(bot.random, "random", return_value=0.5):
                                with mock.patch.object(bot, "post_json") as post_json:
                                    self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                    post_json.assert_not_called()
                        self.assertEqual(len(saved_states), 1)

    def test_maybe_join_bot_lobby_game_records_sample_even_without_candidate(self) -> None:
        games = []

        with mock.patch.object(bot, "get_json", return_value={"games": []}):
            with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                saved_states = []
                with mock.patch.object(bot, "save_state", side_effect=saved_states.append):
                    with mock.patch.object(bot, "post_json") as post_json:
                        self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                        post_json.assert_not_called()

        self.assertEqual(len(saved_states), 1)

    def test_maybe_join_bot_lobby_game_skips_open_sample_during_cooldown(self) -> None:
        games = []

        with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 100}):
            with mock.patch.object(bot.time, "time", return_value=130):
                with mock.patch.object(bot, "get_json") as get_json:
                    self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                    get_json.assert_not_called()

    def test_can_attempt_bot_join_uses_ten_minute_cooldown(self) -> None:
        with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 100}):
            self.assertFalse(bot.can_attempt_bot_join(now=699))
            self.assertTrue(bot.can_attempt_bot_join(now=700))

    def test_maybe_join_bot_lobby_game_joins_when_probability_hits(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                        with mock.patch.object(bot, "save_state"):
                            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                                with mock.patch.object(bot.random, "random", return_value=0.0005):
                                    with mock.patch.object(bot, "llm_preflight_status", return_value=(True, "ok")):
                                        with mock.patch.object(bot, "post_json", return_value={"game_id": "g1", "game_code": "BOT123"}) as post_json:
                                            self.assertTrue(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                            post_json.assert_called_once_with("/game/join/BOT123")

    def test_should_not_join_bot_lobby_game_when_active_cap_reached(self) -> None:
        games = [{"state": "active"} for _ in range(5)]
        self.assertFalse(bot.should_join_bot_lobby_game(games))

    def test_has_own_waiting_game_detects_existing_lobby(self) -> None:
        with mock.patch.dict("os.environ", {"KRIEGSPIEL_BOT_USERNAME": "openrouterbot"}):
            self.assertTrue(bot.has_own_waiting_game([{"game_code": "ABC123", "created_by": "openrouterbot"}]))
            self.assertFalse(bot.has_own_waiting_game([{"game_code": "XYZ789", "created_by": "randobot"}]))

    def test_llm_preflight_status_caches_success(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_MODEL": "provider/model"}, clear=False):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.llm_preflight_status(), (True, "ok"))
                self.assertEqual(bot.llm_preflight_status(), (True, "ok"))
        self.assertEqual(post.call_count, 1)
        self.assertTrue(post.call_args.args[0].endswith("/chat/completions"))
        self.assertEqual(post.call_args.kwargs["json"]["messages"][0]["role"], "system")
        self.assertGreaterEqual(post.call_args.kwargs["json"]["max_tokens"], 16)

    def test_llm_preflight_can_use_max_completion_tokens_for_direct_openai(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.6-luna",
                "LLM_MAX_TOKENS_PARAMETER": "max_completion_tokens",
                "LLM_REASONING_EFFORT": "none",
            },
            clear=False,
        ):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.llm_preflight_status(), (True, "ok"))

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 16)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("max_tokens", payload)

    def test_call_llm_posts_chat_completion_with_json_schema(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "chatcmpl_1"}

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "provider/model",
                "LLM_API_BASE": "https://llm.example/v1",
                "LLM_MAX_OUTPUT_TOKENS": "321",
                "LLM_HTTP_REFERER": "https://kriegspiel.org",
                "LLM_X_OPENROUTER_TITLE": "Kriegspiel",
            },
            clear=False,
        ):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.call_llm(system_prompt="system", user_prompt="user"), {"id": "chatcmpl_1"})

        self.assertEqual(post.call_args.args[0], "https://llm.example/v1/chat/completions")
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["HTTP-Referer"], "https://kriegspiel.org")
        self.assertEqual(headers["X-OpenRouter-Title"], "Kriegspiel")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "provider/model")
        self.assertEqual(payload["messages"], [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}])
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertEqual(payload["response_format"]["json_schema"]["name"], bot.ACTION_SCHEMA_NAME)

    def test_call_llm_can_use_max_completion_tokens_for_direct_openai(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "chatcmpl_1"}

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.6-luna",
                "LLM_API_BASE": "https://api.openai.com/v1",
                "LLM_MAX_OUTPUT_TOKENS": "2048",
                "LLM_MAX_TOKENS_PARAMETER": "max_completion_tokens",
                "LLM_REASONING_EFFORT": "NONE",
            },
            clear=False,
        ):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.call_llm(system_prompt="system", user_prompt="user"), {"id": "chatcmpl_1"})

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_completion_tokens"], 2048)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("max_tokens", payload)

    def test_call_llm_can_opt_into_openai_tool_calls(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "chatcmpl_1"}

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "provider/model",
                "LLM_USE_TOOLS": "true",
                "LLM_REASONING_EFFORT": "",
            },
            clear=False,
        ):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.call_llm(system_prompt="system", user_prompt="user"), {"id": "chatcmpl_1"})

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["tool_choice"], {"type": "function", "function": {"name": bot.ACTION_SCHEMA_NAME}})
        self.assertEqual(payload["tools"][0]["type"], "function")
        self.assertEqual(payload["tools"][0]["function"]["name"], bot.ACTION_SCHEMA_NAME)
        self.assertEqual(payload["tools"][0]["function"]["parameters"], bot.action_schema()["schema"])
        self.assertTrue(payload["tools"][0]["function"]["strict"])

    def test_call_llm_can_set_reasoning_effort_for_tool_calls(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "chatcmpl_1"}

        with mock.patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_MODEL": "gpt-5.6-sol",
                "LLM_USE_TOOLS": "true",
                "LLM_REASONING_EFFORT": "none",
            },
            clear=False,
        ):
            with mock.patch.object(bot.requests, "post", return_value=response) as post:
                self.assertEqual(bot.call_llm(system_prompt="system", user_prompt="user"), {"id": "chatcmpl_1"})

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["tools"][0]["function"]["name"], bot.ACTION_SCHEMA_NAME)

    def test_model_call_semaphore_limits_concurrent_call_llm_execution(self) -> None:
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"id": "chatcmpl_1"}
        active_calls = 0
        max_active_calls = 0
        lock = threading.Lock()

        def slow_post(*args, **kwargs):  # noqa: ANN002, ANN003
            nonlocal active_calls, max_active_calls
            with lock:
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            time.sleep(0.03)
            with lock:
                active_calls -= 1
            return response

        bot.configure_model_call_semaphore(2)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                bot.call_llm(system_prompt="system", user_prompt="user")
            except BaseException as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key", "LLM_MODEL": "provider/model"}, clear=False):
            with mock.patch.object(bot.requests, "post", side_effect=slow_post):
                threads = [threading.Thread(target=worker) for _ in range(5)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=1)

        self.assertEqual(errors, [])
        self.assertLessEqual(max_active_calls, 2)
        self.assertEqual(max_active_calls, 2)

    def test_report_model_availability_posts_status_and_throttles_repeats(self) -> None:
        with mock.patch.object(bot, "post_json", return_value={"ok": True}) as post_json:
            self.assertTrue(bot.report_model_availability(False, "http_429: insufficient_quota"))
            self.assertFalse(bot.report_model_availability(False, "http_429: insufficient_quota"))
            self.assertTrue(bot.report_model_availability(True, "ok"))

        self.assertEqual(
            post_json.call_args_list,
            [
                mock.call("/bots/availability", {"provider": "openai", "ready": False, "reason": "http_429: insufficient_quota"}),
                mock.call("/bots/availability", {"provider": "openai", "ready": True, "reason": "ok"}),
            ],
        )

    def test_runner_scheduler_starts_one_runner_per_game_without_duplicates(self) -> None:
        class FakeRunner:
            def __init__(self, game_id: str) -> None:
                self.game_id = game_id
                self.started = 0
                self.stopped = 0
                self.joined = 0
                self.alive = False

            def start(self) -> None:
                self.started += 1
                self.alive = True

            def stop(self) -> None:
                self.stopped += 1
                self.alive = False

            def join(self, timeout: float | None = None) -> None:  # noqa: ARG002
                self.joined += 1

            def is_alive(self) -> bool:
                return self.alive

        created: dict[str, FakeRunner] = {}

        def runner_factory(game_id: str) -> FakeRunner:
            runner = FakeRunner(game_id)
            created[game_id] = runner
            return runner

        scheduler = bot.GameRunnerScheduler(poll_seconds=0.01, runner_factory=runner_factory)
        games = [
            {"state": "active", "game_id": "g1"},
            {"state": "active", "game_id": "g2"},
            {"state": "waiting", "game_id": "w1"},
        ]

        scheduler.reconcile(games)
        scheduler.reconcile(games)

        self.assertEqual(set(created), {"g1", "g2"})
        self.assertEqual(created["g1"].started, 1)
        self.assertEqual(created["g2"].started, 1)

        scheduler.reconcile([{"state": "active", "game_id": "g2"}])

        self.assertEqual(created["g1"].stopped, 0)
        self.assertIn("g1", scheduler.runners)
        self.assertIn("g2", scheduler.runners)

        created["g1"].alive = False
        scheduler.reconcile([{"state": "active", "game_id": "g2"}])

        self.assertNotIn("g1", scheduler.runners)
        self.assertIn("g2", scheduler.runners)

    def test_one_slow_game_runner_does_not_block_another_runner(self) -> None:
        slow_started = threading.Event()
        release_slow = threading.Event()
        fast_played = threading.Event()

        def fake_get_json(path: str) -> dict[str, str]:
            game_id = path.split("/")[2]
            return {"state": "active", "turn": "white", "your_color": "white", "game_id": game_id}

        def fake_maybe_play_game(game_id: str) -> bool:
            if game_id == "slow":
                slow_started.set()
                release_slow.wait(timeout=1)
                return True
            fast_played.set()
            return True

        slow_runner = bot.GameRunner("slow", poll_seconds=0.01)
        fast_runner = bot.GameRunner("fast", poll_seconds=0.01)

        with mock.patch.object(bot, "get_json", side_effect=fake_get_json):
            with mock.patch.object(bot, "maybe_play_game", side_effect=fake_maybe_play_game):
                slow_runner.start()
                self.assertTrue(slow_started.wait(timeout=0.5))
                fast_runner.start()
                self.assertTrue(fast_played.wait(timeout=0.5))
                slow_runner.stop()
                fast_runner.stop()
                release_slow.set()
                slow_runner.join(timeout=1)
                fast_runner.join(timeout=1)

        self.assertFalse(slow_runner.is_alive())
        self.assertFalse(fast_runner.is_alive())

    def test_maybe_join_bot_lobby_game_skips_join_when_llm_unavailable(self) -> None:
        games = []
        open_games = [{"game_code": "BOT123", "created_by": "randobot", "rule_variant": "berkeley_any"}]

        with mock.patch.dict("os.environ", {"LLM_API_KEY": "test-key"}, clear=False):
            with mock.patch.object(bot, "get_json", return_value={"games": open_games}):
                with mock.patch.object(bot, "get_public_user", return_value={"role": "bot"}):
                    with mock.patch.object(bot, "load_state", return_value={"last_bot_game_join_attempt_at": 0}):
                        with mock.patch.object(bot, "save_state"):
                            with mock.patch.object(bot.random, "choice", side_effect=lambda items: items[0]):
                                with mock.patch.object(bot.random, "random", return_value=0.0005):
                                    with mock.patch.object(bot, "llm_preflight_status", return_value=(False, "http_429: insufficient_quota")):
                                        with mock.patch.object(bot, "post_json") as post_json:
                                            self.assertFalse(bot.maybe_join_bot_lobby_game(games, rng=bot.random))
                                            post_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
