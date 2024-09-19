import asyncio

from llama_index.core.workflow import Workflow


class HumanInTheLoopWorkflow(Workflow):
    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop()  # Store the event loop
        result = await super().run(*args, **kwargs)
        return result
