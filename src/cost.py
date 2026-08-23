"""
Illustrative cost model -- turns measured throughput into $/request from a
documented $/GPU-hr constant. Not looked up live; check current pricing
before treating these numbers as more than directional (same convention as
AnthropicClient's PRICE_IN/PRICE_OUT in the sibling agent-review-loop repo).
"""

DEFAULT_GPU_HOURLY_USD = 0.35  # illustrative Colab Pay-As-You-Go T4-class rate


def cost_usd_per_request(throughput_req_s: float, gpu_hourly_usd: float = DEFAULT_GPU_HOURLY_USD) -> float:
    if throughput_req_s <= 0:
        return float("inf")
    requests_per_hour = throughput_req_s * 3600
    return round(gpu_hourly_usd / requests_per_hour, 6)
