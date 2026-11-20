import unittest
import httpx
from ai_gateway import app as gateway
from ai_gateway.crawler_policy import CrawlerPolicyMiddleware


class CrawlerPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_public_robots_auth_failures_and_missing_routes(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app), base_url='https://gateway') as client:
            robots = await client.get('/robots.txt')
            self.assertEqual(robots.status_code, 200)
            self.assertEqual(robots.text, 'User-agent: *\nDisallow: /\n')
            self.assertTrue(robots.headers['content-type'].startswith('text/plain'))
            head = await client.head('/robots.txt')
            self.assertEqual(head.status_code, 200)
            self.assertEqual(head.content, b'')
            for path, method, status in [('/v1/chat/completions','GET',401),('/v1/chat/completions','POST',401),('/missing','GET',404)]:
                result = await client.request(method,path)
                self.assertEqual(result.status_code,status)
                self.assertEqual(result.headers['x-robots-tag'],'noindex, nofollow, nosnippet')

    async def test_stream_frames_are_forwarded_immediately(self):
        messages = []
        async def downstream(scope, receive, send):
            await send({'type':'http.response.start','status':200,'headers':[(b'content-type',b'text/event-stream')]})
            self.assertEqual(len(messages),1)
            await send({'type':'http.response.body','body':b'data: first\n\n','more_body':True})
            self.assertEqual(len(messages),2)
            await send({'type':'http.response.body','body':b'data: [DONE]\n\n','more_body':False})
        async def receive(): return {'type':'http.disconnect'}
        async def send(message): messages.append(message)
        await CrawlerPolicyMiddleware(downstream)({'type':'http','path':'/v1/chat/completions','method':'POST'},receive,send)
        self.assertEqual(messages[1]['body'],b'data: first\n\n')
        self.assertTrue(messages[1]['more_body'])
        self.assertIn((b'x-robots-tag',b'noindex, nofollow, nosnippet'),messages[0]['headers'])
