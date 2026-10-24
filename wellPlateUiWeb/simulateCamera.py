"""Well Plate UI: a fake camera, to test the whole chain without the real one.

Standard library only. Start the server, plan a few wells, press Start run, then:

    python simulateCamera.py                     # walk the plan, sometimes into a wrong well
    python simulateCamera.py --errorRate 0 --delay 0.5

It sends what the real camera code (scripts/live_demo.py --ui) sends: a
heartbeat every 3 s and one POST /api/drop per dispense. If the server was
started with WELLPLATE_API_TOKEN set, set the same variable here (or --token).
"""

import argparse
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


class WellPlateClient:
    def __init__(self, baseUrl="http://localhost:7500", token=None, timeout=5):
        self.baseUrl = baseUrl.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, method, path, body=None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-Api-Token"] = self.token
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.baseUrl + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except Exception:
                message = e.reason
            raise ApiError(e.code, message) from None

    def getPlan(self):
        return self._request("GET", "/api/plan")

    def getState(self):
        return self._request("GET", "/api/state")

    def heartbeat(self):
        return self._request("POST", "/api/camera/heartbeat", {})

    def reportDrop(self, wells, compoundId=None):
        """A tip dispensed into these wells. Returns the server's verdict."""
        body = {"wells": wells}
        if compoundId is not None:
            body["compoundId"] = compoundId
        return self._request("POST", "/api/drop", body)

    def startHeartbeat(self, interval=3.0):
        """Send a heartbeat in the background. Returns a function that stops it."""
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                try:
                    self.heartbeat()
                except Exception as e:  # server restarting, network blip: keep trying
                    print(f"heartbeat failed: {e}")
                stop.wait(interval)

        threading.Thread(target=loop, daemon=True).start()
        return stop.set

    def waitForAck(self, timeout=120, pollSeconds=0.5):
        """Block until the user clicks Acknowledge in the UI."""
        end = time.time() + timeout
        while time.time() < end:
            if not self.getState()["run"]["alarm"]:
                return True
            time.sleep(pollSeconds)
        return False

    def waitForRun(self, pollSeconds=1.0):
        """Block until the user presses Start run in the UI."""
        announced = False
        while self.getPlan()["status"] != "active":
            if not announced:
                print("Press Start run in the Run tab. Waiting...")
                announced = True
            time.sleep(pollSeconds)


def printVerdict(res):
    if not res.get("accepted"):
        print("  not accepted:", res.get("reason"))
        return
    for r in res["results"]:
        print(f"  {r['well']:<4} {r['kind']:<10} (planned: {r['planned']})")


def simulate(client, delay=1.2, errorRate=0.2, pollSeconds=0.5, rng=None):
    """Walk the plan step by step (timers included), column by column. Sometimes
    dispense the wrong compound into a well, wait for the user to acknowledge
    the warning, then do it correctly."""
    rng = rng or random.Random()
    client.waitForRun(pollSeconds)
    plan = client.getPlan()
    allIds = [c["id"] for c in plan["compounds"]]
    for st in plan["steps"]:
        if st["kind"] == "wait":
            print(f"Timer: waiting {st['seconds']} s")
            time.sleep(st["seconds"] + 0.3)
            continue
        cid = st["compoundId"]
        for w in sorted(st["wells"], key=lambda w: (int(w[1:]), w[0])):
            time.sleep(delay)
            others = [c for c in allIds if c != cid]
            if others and rng.random() < errorRate:
                print(f"Simulated MISTAKE: {w} with the wrong compound")
                res = client.reportDrop([w], rng.choice(others))
                printVerdict(res)
                if not res.get("accepted"):
                    print("The run is not active any more. Stopping.")
                    return False
                print("  waiting for you to acknowledge the warning in the UI...")
                client.waitForAck(pollSeconds=pollSeconds)
                time.sleep(delay)
            print(f"Drop {w}")
            res = client.reportDrop([w], cid)
            printVerdict(res)
            if not res.get("accepted"):
                print("The run is not active any more. Stopping.")
                return False
    print("Simulation finished.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Well Plate UI: fake camera")
    ap.add_argument("--url", default=os.environ.get("WELLPLATE_URL", "http://localhost:7500"))
    ap.add_argument("--token", default=os.environ.get("WELLPLATE_API_TOKEN"),
                    help="only needed if the server was started with WELLPLATE_API_TOKEN")  # fmt: skip
    ap.add_argument("--delay", type=float, default=1.2, help="seconds between drops")
    ap.add_argument("--errorRate", type=float, default=0.2, help="chance of a mistake (0 to 1)")
    ap.add_argument("--seed", type=int, default=None, help="make the mistakes repeatable")
    args = ap.parse_args()

    client = WellPlateClient(args.url, args.token)
    try:
        plan = client.getPlan()
        client.heartbeat()
    except ApiError as e:
        raise SystemExit(f"The server refused this client ({e}). Check the token.") from None
    except OSError as e:
        raise SystemExit(f"Cannot reach the server at {args.url}: {e}") from None
    print(f"Connected to {args.url}. {len(plan['steps'])} steps planned.")
    client.startHeartbeat()
    simulate(client, args.delay, args.errorRate, rng=random.Random(args.seed))


if __name__ == "__main__":
    main()
