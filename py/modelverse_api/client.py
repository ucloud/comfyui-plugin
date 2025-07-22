import time
import requests
import asyncio
from .utils import BaseRequest


class ModelverseClient:
    """
    UCloud Modelverse API Client

    This class handles the core communication with the Modelverse API.
    """

    BASE_URL = "https://api.modelverse.cn"
    API_PATH = "/v1/images/generations"

    def __init__(self, api_key):
        """
        Initialize Modelverse API client

        Args:
            api_key (str): Modelverse API key
        """
        self.api_key = api_key

        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"}

    def post(self, payload, timeout=180):
        """
        Send POST request to Modelverse API

        Args:
            endpoint (str): API endpoint
            payload (dict): Request payload
            timeout (float, optional): Request timeout in seconds

        Returns:
            dict: API response
        """
        url = f"{self.BASE_URL}{self.API_PATH}"
        response = requests.post(
            url, headers=self.headers, json=payload, timeout=timeout)

        if response.status_code == 401:
            raise Exception("Unauthorized: Invalid API key")

        if response.status_code != 200:
            error_message = f"Error: {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_message = f"Error: {error_data['error']}"
            except:
                pass
            raise Exception(error_message)

        response_data = response.json()
        if isinstance(response_data, dict) and 'code' in response_data:
            if response_data['code'] == 401:
                raise Exception("Unauthorized: Invalid API key")
            if response_data['code'] != 200:
                raise Exception(
                    f"API Error: {response_data.get('message', 'Unknown error')}")
            return response_data.get('data', {})
        return response_data

    async def async_send_request(self, request: BaseRequest):
        """
        Sends an API request using a request object.

        Args:
            request (BaseRequest): The request object containing payload and endpoint logic.

        Returns:
            dict: API response or task result.
        """
        payload = request.build_payload()
        if "seed" in payload:
            payload["seed"] = payload["seed"] % 2147483647 if payload["seed"] != -1 else -1

        response = self.post(payload)
        return response.get("data", [])

    async def run_tasks(self, tasks):
        print("INFO:", f"Sending {len(tasks)} request(s) concurrently...")
        results = await asyncio.gather(*tasks)
        return results
