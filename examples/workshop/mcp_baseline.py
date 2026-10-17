"""Run the same RGB-D baseline through a real stdio MCP server/client pair.

This validates transport parity; the decisions are scripted, not LLM-generated.
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agilex_deploy.workshop_client import RobotClient, APIError
from code_policy import run_policy


class MCPRobotClient(RobotClient):
    def __init__(self, config):
        self.config = str(Path(config).resolve())
        self.claimed = False
        self.ready = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._thread, daemon=False)
        self.thread.start()
        if not self.ready.wait(20):
            raise TimeoutError("MCP startup timed out")
        if self.error:
            raise self.error

    def _thread(self):
        try:
            asyncio.run(self._main())
        except BaseException as error:
            self.error = error
            self.ready.set()

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        self.finished = asyncio.Event()
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agilex_deploy.workshop_mcp", "--config", self.config],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as self.session:
                await self.session.initialize()
                self.ready.set()
                await self.finished.wait()

    def tools(self):
        return (
            asyncio.run_coroutine_threadsafe(self.session.list_tools(), self.loop)
            .result(10)
            .tools
        )

    def call(self, method, **params):
        result = asyncio.run_coroutine_threadsafe(
            self.session.call_tool(method, params), self.loop
        ).result(35)
        if result.isError:
            raise APIError(
                "MCP_TOOL_ERROR",
                "\n".join(c.text for c in result.content if c.type == "text"),
            )
        if result.structuredContent is not None:
            return result.structuredContent
        return json.loads(next(c.text for c in result.content if c.type == "text"))

    def close(self):
        try:
            super().close()
        finally:
            self.loop.call_soon_threadsafe(self.finished.set)
            self.thread.join(10)
            if self.thread.is_alive():
                raise RuntimeError("MCP child did not exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="logs/workshop/client.json")
    args = parser.parse_args()
    with MCPRobotClient(args.config) as robot:
        run_policy(robot, robot.capabilities()["instruction"])


if __name__ == "__main__":
    main()
