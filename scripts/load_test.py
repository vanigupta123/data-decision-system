import asyncio
import httpx
import time

from metrics import timer

def load_test():
    async def send_request(session, url, payload):
        try:
            with timer() as t:
                resp = await session.post(url, json=payload)
                elapsed_ms = t.ms
            return resp.status_code, await resp.text(), elapsed_ms
        except Exception as e:
            return f"error {str(e)}"

    async def main():
        url = "http://localhost:8000/predict"
        payload = {
            "features": {"temperature": 273.3},
            "run_config": {"missing_rate": 0.2,
            "delay_steps": 1, "noise_std": 0.3, "seed": 1}
        }
        concurrent = 10
        requests_per_worker = 10

        async with httpx.AsyncClient(timeout=30) as session:
            tasks = []
            for _ in range(concurrent):
                for _ in range(requests_per_worker):
                    tasks.append(send_request(session, url, payload))
            with timer() as t:
                results = await asyncio.gather(*tasks)
                t1 = t.ms
            successes = 0
            times = []
            for result in results:
                code, _, elapsed = result
                if code == 200:
                    successes += 1
                if elapsed is not None:
                    times.append(elapsed)

            p50 = sum(times) / len(times) if times else 0
            print(f"sent {len(tasks)} requests in {t.ms}ms, {successes} succeeded")
            print(f"p50: {p50}ms")

    asyncio.run(main())
