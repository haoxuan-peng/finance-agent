import unittest

from tavily.errors import BadRequestError, UsageLimitExceededError

from finance_agent.key_rotator import KeyRotator, NoAvailableAPIKeysError
from finance_agent.tools import TavilyWebSearch


class _FakeTavilyClient:
    def __init__(self, key, outcomes, calls):
        self._key = key
        self._outcomes = outcomes
        self._calls = calls

    async def search(self, **kwargs):
        self._calls.append((self._key, kwargs))
        outcome = self._outcomes[self._key]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TavilyKeyFailoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_switches_to_next_key_and_disables_exhausted_key(self):
        calls = []
        outcomes = {
            "exhausted": UsageLimitExceededError("usage limit exceeded"),
            "working": {"results": [{"title": "result"}]},
        }
        rotator = KeyRotator(["exhausted", "working"])
        tool = TavilyWebSearch(
            key_rotator=rotator,
            client_factory=lambda key: _FakeTavilyClient(key, outcomes, calls),
        )

        results = await tool._execute_search("test query")

        self.assertEqual(results, [{"title": "result"}])
        self.assertEqual([key for key, _ in calls], ["exhausted", "working"])
        self.assertEqual(rotator.active_key_count, 1)

        calls.clear()
        await tool._execute_search("second query")
        self.assertEqual([key for key, _ in calls], ["working"])

    async def test_does_not_rotate_for_bad_request(self):
        calls = []
        outcomes = {
            "first": BadRequestError("bad search parameters"),
            "second": {"results": []},
        }
        rotator = KeyRotator(["first", "second"])
        tool = TavilyWebSearch(
            key_rotator=rotator,
            client_factory=lambda key: _FakeTavilyClient(key, outcomes, calls),
        )

        with self.assertRaises(BadRequestError):
            await tool._execute_search("test query")

        self.assertEqual([key for key, _ in calls], ["first"])
        self.assertEqual(rotator.active_key_count, 2)

    async def test_raises_safe_error_after_all_keys_fail(self):
        outcomes = {
            "first": UsageLimitExceededError("first exhausted"),
            "second": UsageLimitExceededError("second exhausted"),
        }
        rotator = KeyRotator(["first", "second"])
        tool = TavilyWebSearch(
            key_rotator=rotator,
            client_factory=lambda key: _FakeTavilyClient(key, outcomes, []),
        )

        with self.assertRaisesRegex(
            NoAvailableAPIKeysError,
            "All configured Tavily API keys are unavailable",
        ):
            await tool._execute_search("test query")

        self.assertEqual(rotator.active_key_count, 0)


if __name__ == "__main__":
    unittest.main()
