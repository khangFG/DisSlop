import asyncio
from typing import Any, Dict, List, Optional


class FakeUser:
    def __init__(self, user_id: int = 123456789, name: str = "TestUser", bot: bool = False):
        self.id = user_id
        self.name = name
        self.bot = bot
        self.mention = f"<@{user_id}>"

    def __str__(self) -> str:
        return f"{self.name}#{self.id}"


class FakeGuild:
    def __init__(self, guild_id: int = 987654321, name: str = "TestGuild"):
        self.id = guild_id
        self.name = name


class FakeChannel:
    def __init__(self, channel_id: int = 555555555, name: str = "general"):
        self.id = channel_id
        self.name = name
        self.sent_messages: List[str] = []
        self._typing_context = FakeTypingContext()

    def typing(self) -> "FakeTypingContext":
        return self._typing_context

    async def send(self, content: str, **kwargs: Any) -> "FakeMessage":
        self.sent_messages.append(content)
        return FakeMessage(content=content, channel=self, author=FakeUser(user_id=9999, name="Bot", bot=True))

    async def history(self, limit: int = 10, before: Any = None) -> Any:
        # Async generator of messages
        for msg in []:
            yield msg


class FakeTypingContext:
    async def __aenter__(self) -> "FakeTypingContext":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class FakeResponse:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        self.deferred: bool = False
        self._done: bool = False

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, content: str, ephemeral: bool = False, **kwargs: Any) -> None:
        self.messages.append({"content": content, "ephemeral": ephemeral, **kwargs})
        self._done = True

    async def defer(self, ephemeral: bool = False, **kwargs: Any) -> None:
        self.deferred = True
        self._done = True


class FakeFollowup:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    async def send(self, content: str, ephemeral: bool = False, **kwargs: Any) -> None:
        self.messages.append({"content": content, "ephemeral": ephemeral, **kwargs})


class FakeInteraction:
    def __init__(
        self,
        user: Optional[FakeUser] = None,
        guild: Optional[FakeGuild] = None,
        channel: Optional[FakeChannel] = None,
    ):
        self.user = user or FakeUser()
        self.guild = guild or FakeGuild()
        self.channel = channel or FakeChannel()
        self.response = FakeResponse()
        self.followup = FakeFollowup()


class FakeMessage:
    def __init__(
        self,
        content: str = "",
        author: Optional[FakeUser] = None,
        channel: Optional[FakeChannel] = None,
        mentions: Optional[List[FakeUser]] = None,
        message_id: int = 111111,
    ):
        self.id = message_id
        self.content = content
        self.author = author or FakeUser()
        self.channel = channel or FakeChannel()
        self.mentions = mentions or []
        self.replies: List[Dict[str, Any]] = []

    async def reply(self, content: str, mention_author: bool = False, **kwargs: Any) -> "FakeMessage":
        self.replies.append({"content": content, "mention_author": mention_author, **kwargs})
        return FakeMessage(content=content, channel=self.channel, author=FakeUser(user_id=9999, name="Bot", bot=True))


class FakeToolResult:
    def __init__(self, content: Any = None, is_error: bool = False):
        self.content = content if isinstance(content, list) else ([content] if content is not None else [])
        self.is_error = is_error


class FakeTextContent:
    def __init__(self, text: str):
        self.text = text


class FakeMCPServerWorker:
    """Fake worker that doesn't spawn real subprocesses."""

    def __init__(self, name: str, tools: Optional[List[Any]] = None, call_handler: Optional[Any] = None):
        self.name = name
        self.tools = tools or []
        self.call_handler = call_handler
        self.error: Optional[str] = None
        self.task: Optional[asyncio.Task] = None
        self.calls: List[tuple] = []
        self._stopped = False

    async def start(self, timeout: float = 5.0) -> bool:
        return True

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 5.0, authorize=None) -> Any:
        if authorize is not None and not authorize():
            raise PermissionError("Tool access revoked")
        self.calls.append((tool_name, arguments))
        if self.call_handler:
            return await self.call_handler(tool_name, arguments)
        return FakeToolResult(content=[FakeTextContent(f"Fake result for {tool_name}")])

    async def stop(self) -> None:
        self._stopped = True
