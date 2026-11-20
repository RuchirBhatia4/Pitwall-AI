"""Rain forecast over the circuit for the next two hours (Open-Meteo, no key).

Live sessions only: the pit call itself does not use it. It warns that rain may
arrive before the planned stop, which the lap-time maths cannot see coming.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request

# (latitude, longitude) of each 2026 venue, keyed by the season file's location
VENUES = {
    "Melbourne": (-37.8497, 144.9680), "Shanghai": (31.3389, 121.2197), "Suzuka": (34.8431, 136.5407),
    "Miami Gardens": (25.9581, -80.2389), "Montréal": (45.5005, -73.5228), "Monte Carlo": (43.7347, 7.4206),
    "Barcelona": (41.5700, 2.2611), "Spielberg": (47.2197, 14.7647), "Silverstone": (52.0786, -1.0169),
    "Spa-Francorchamps": (50.4372, 5.9714), "Budapest": (47.5789, 19.2486), "Zandvoort": (52.3888, 4.5409),
    "Monza": (45.6156, 9.2811), "Madrid": (40.4637, -3.6166), "Baku": (40.3725, 49.8533),
    "Kuala Lumpur": (2.7608, 101.7382), "Marina Bay": (1.2914, 103.8640), "Austin": (30.1328, -97.6411),
    "Mexico City": (19.4042, -99.0907), "São Paulo": (-23.7036, -46.6997), "Las Vegas": (36.1147, -115.1728),
    "Lusail": (25.4900, 51.4542), "Yas Marina": (24.4672, 54.6031),
}
URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&minutely_15=precipitation,precipitation_probability&forecast_minutely_15=8&timezone=UTC")
CACHE_S = 600  # forecasts update hourly; every viewer shares one fetch
RETRY_S = 120  # after a failed fetch
TIMEOUT_S = 3.0

_cache: dict[str, tuple[float, dict | None]] = {}
_lock = threading.Lock()


def _fetch(lat: float, lon: float) -> dict | None:
    with urllib.request.urlopen(URL.format(lat=lat, lon=lon), timeout=TIMEOUT_S) as r:
        m = json.load(r)["minutely_15"]
    prob = [p or 0 for p in m["precipitation_probability"]]
    mm = [x or 0.0 for x in m["precipitation"]]
    if len(prob) < 8:
        return None
    return {
        "next_hour": max(prob[:4]),
        "hour_after": max(prob[4:8]),
        "mm_2h": round(sum(mm), 1),
        "from": m["time"][0],
        "source": "Open-Meteo",
    }


def rain_forecast(location: str | None) -> dict | None:
    coords = VENUES.get(location or "")
    if coords is None:
        return None
    with _lock:
        hit = _cache.get(location)
        if hit and time.time() - hit[0] < (CACHE_S if hit[1] else RETRY_S):
            return hit[1]
        try:
            out = _fetch(*coords)
        except Exception:
            out = None
        _cache[location] = (time.time(), out)
        return out
