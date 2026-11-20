"""Crawler controls for this API-only origin, without buffering model streams."""
from starlette.datastructures import MutableHeaders
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send, Message


class CrawlerPolicyMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        async def send_with_policy(message: Message) -> None:
            if message['type'] == 'http.response.start':
                headers = MutableHeaders(scope=message)
                headers['X-Robots-Tag'] = 'noindex, nofollow, nosnippet'
            await send(message)

        if scope['path'] == '/robots.txt' and scope['method'] in {'GET', 'HEAD'}:
            response = PlainTextResponse('User-agent: *\nDisallow: /\n', headers={'Cache-Control': 'public, max-age=3600'})
            await response(scope, receive, send_with_policy)
            return
        await self.app(scope, receive, send_with_policy)
