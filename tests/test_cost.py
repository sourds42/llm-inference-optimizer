import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cost import cost_usd_per_request


def test_higher_throughput_is_cheaper():
    cheap = cost_usd_per_request(throughput_req_s=10, gpu_hourly_usd=0.35)
    expensive = cost_usd_per_request(throughput_req_s=1, gpu_hourly_usd=0.35)
    assert cheap < expensive


def test_zero_throughput_is_infinite_cost():
    assert cost_usd_per_request(throughput_req_s=0) == float("inf")
