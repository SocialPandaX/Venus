import asyncio
import builtins
import queue
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class AgentMessage:
    kind: str
    content: str
    from_agent: Optional[str] = None
    call_id: Optional[str] = None
    call_chain: Optional[list[str]] = None


class AgentRuntime:
    def __init__(
        self,
        agent_names: list[str],
        extra_env: Optional[dict] = None,
        call_timeout: float = 300.0,
        agent_factory: Optional[Callable[..., object]] = None,
    ) -> None:
        self._agent_names = list(dict.fromkeys(agent_names))
        self.extra_env = extra_env or {}
        self.call_timeout = float(call_timeout)
        self._agent_factory = agent_factory
        self._queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._call_futures: dict[str, Future] = {}
        self._call_lock = threading.Lock()
        self._active_agent: Optional[str] = self._agent_names[0] if self._agent_names else None
        self._print_lock = threading.Lock()
        self._orig_print = None

    def list_agents(self) -> list[str]:
        return list(self._agent_names)

    def get_active(self) -> Optional[str]:
        return self._active_agent

    def set_active(self, agent_name: str) -> bool:
        if agent_name not in self._agent_names:
            return False
        self._active_agent = agent_name
        return True

    def _install_print_lock(self) -> None:
        if self._orig_print is not None:
            return

        self._orig_print = builtins.print

        def locked_print(*args, **kwargs):
            with self._print_lock:
                self._orig_print(*args, **kwargs)

        builtins.print = locked_print

    def _restore_print(self) -> None:
        if self._orig_print is None:
            return
        builtins.print = self._orig_print
        self._orig_print = None

    def start(self) -> None:
        if not self._agent_names:
            raise ValueError("No agents configured for runtime")

        self._install_print_lock()
        for name in self._agent_names:
            if name in self._threads:
                continue
            q: queue.Queue = queue.Queue()
            self._queues[name] = q
            thread = threading.Thread(
                target=self._agent_thread_main,
                name=f"AgentThread-{name}",
                args=(name, q),
                daemon=True,
            )
            self._threads[name] = thread
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        for name, q in self._queues.items():
            q.put(AgentMessage(kind="shutdown", content="", from_agent=None))

    def join(self, timeout: Optional[float] = None) -> None:
        for thread in self._threads.values():
            thread.join(timeout=timeout)
        self._restore_print()

    def _agent_thread_main(self, agent_name: str, q: queue.Queue) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if self._agent_factory is None:
                from agent.base_agent import BaseAgent

                agent = BaseAgent(agent_name, extra_env=self.extra_env, runtime=self)
            else:
                agent = self._agent_factory(agent_name, runtime=self, extra_env=self.extra_env)

            loop.run_until_complete(agent.run_queue(q, self._stop_event))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    def _enqueue(self, agent_name: str, message: AgentMessage) -> bool:
        q = self._queues.get(agent_name)
        if not q:
            return False
        q.put(message)
        return True

    def send_user_message(self, agent_name: str, content: str) -> bool:
        return self._enqueue(agent_name, AgentMessage(kind="user", content=content, from_agent=None))

    def broadcast(self, content: str) -> None:
        for name in self._agent_names:
            self._enqueue(name, AgentMessage(kind="user", content=content, from_agent=None))

    def notify(self, to_agent: str, content: str, from_agent: Optional[str] = None) -> bool:
        return self._enqueue(
            to_agent,
            AgentMessage(kind="notify", content=content, from_agent=from_agent),
        )

    def _create_call(
        self,
        to_agent: str,
        content: str,
        from_agent: Optional[str],
        call_chain: Optional[list[str]] = None,
    ) -> tuple[Optional[str], Optional[Future]]:
        if to_agent not in self._queues:
            return None, None
        chain = list(call_chain or [])
        if from_agent:
            chain.append(from_agent)
        call_id = uuid.uuid4().hex
        future: Future = Future()
        with self._call_lock:
            self._call_futures[call_id] = future
        self._enqueue(
            to_agent,
            AgentMessage(
                kind="call",
                content=content,
                from_agent=from_agent,
                call_id=call_id,
                call_chain=chain or None,
            ),
        )
        return call_id, future

    def call(
        self,
        to_agent: str,
        content: str,
        from_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        call_chain: Optional[list[str]] = None,
    ) -> str:
        call_id, future = self._create_call(to_agent, content, from_agent, call_chain=call_chain)
        if not future:
            return f"Error: agent '{to_agent}' not found."

        try:
            return future.result(timeout=timeout or self.call_timeout) or ""
        except Exception:
            return f"Error: call to '{to_agent}' timed out."
        finally:
            if call_id:
                with self._call_lock:
                    self._call_futures.pop(call_id, None)

    async def call_async(
        self,
        to_agent: str,
        content: str,
        from_agent: Optional[str] = None,
        timeout: Optional[float] = None,
        call_chain: Optional[list[str]] = None,
    ) -> str:
        call_id, future = self._create_call(to_agent, content, from_agent, call_chain=call_chain)
        if not future:
            return f"Error: agent '{to_agent}' not found."

        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=timeout or self.call_timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: call to '{to_agent}' timed out."
        finally:
            if call_id:
                with self._call_lock:
                    self._call_futures.pop(call_id, None)

    def complete_call(self, call_id: Optional[str], result: str) -> None:
        if not call_id:
            return
        with self._call_lock:
            future = self._call_futures.get(call_id)
        if future and not future.done():
            future.set_result(result)
