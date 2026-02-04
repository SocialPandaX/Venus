import unittest

from core.console_router import parse_console_input


class ConsoleRouterTests(unittest.TestCase):
    def test_at_agent(self):
        cmd = parse_console_input("@dev hello", "manager")
        self.assertEqual(cmd.action, "send")
        self.assertEqual(cmd.target, "dev")
        self.assertEqual(cmd.message, "hello")

    def test_broadcast(self):
        cmd = parse_console_input("/broadcast hi", "manager")
        self.assertEqual(cmd.action, "broadcast")
        self.assertEqual(cmd.message, "hi")

    def test_active(self):
        cmd = parse_console_input("/active dev", "manager")
        self.assertEqual(cmd.action, "active")
        self.assertEqual(cmd.target, "dev")

    def test_list(self):
        cmd = parse_console_input("/list", "manager")
        self.assertEqual(cmd.action, "list")

    def test_quit(self):
        cmd = parse_console_input("/quit", "manager")
        self.assertEqual(cmd.action, "quit")

    def test_default_send(self):
        cmd = parse_console_input("hello", "manager")
        self.assertEqual(cmd.action, "send")
        self.assertEqual(cmd.target, "manager")
        self.assertEqual(cmd.message, "hello")

    def test_invalid_at_agent(self):
        cmd = parse_console_input("@dev", "manager")
        self.assertEqual(cmd.action, "error")


if __name__ == "__main__":
    unittest.main()
