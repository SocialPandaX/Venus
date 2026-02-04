import asyncio
import unittest

from agent.runtime import AgentRuntime


class EchoAgent:
    def __init__(self, name, runtime=None, extra_env=None):
        self.name = name
        self.runtime = runtime

    async def run_queue(self, input_queue, stop_event=None):
        while True:
            message = await asyncio.get_running_loop().run_in_executor(None, input_queue.get)
            if message.kind == "shutdown":
                break
            if message.kind == "call":
                if self.runtime:
                    self.runtime.complete_call(message.call_id, f"{self.name}:{message.content}")


class SilentAgent:
    def __init__(self, name, runtime=None, extra_env=None):
        self.name = name
        self.runtime = runtime

    async def run_queue(self, input_queue, stop_event=None):
        while True:
            message = await asyncio.get_running_loop().run_in_executor(None, input_queue.get)
            if message.kind == "shutdown":
                break


class RuntimeTests(unittest.TestCase):
    def test_call(self):
        runtime = AgentRuntime(["a"], call_timeout=1.0, agent_factory=EchoAgent)
        runtime.start()
        try:
            result = runtime.call("a", "ping", from_agent="b")
            self.assertEqual(result, "a:ping")
        finally:
            runtime.stop()
            runtime.join()

    def test_call_timeout(self):
        runtime = AgentRuntime(["a"], call_timeout=0.1, agent_factory=SilentAgent)
        runtime.start()
        try:
            result = runtime.call("a", "ping", from_agent="b", timeout=0.1)
            self.assertIn("timed out", result)
        finally:
            runtime.stop()
            runtime.join()

    def test_notify(self):
        runtime = AgentRuntime(["a"], call_timeout=1.0, agent_factory=EchoAgent)
        runtime.start()
        try:
            ok = runtime.notify("a", "hello", from_agent="b")
            self.assertTrue(ok)
        finally:
            runtime.stop()
            runtime.join()


if __name__ == "__main__":
    unittest.main()
