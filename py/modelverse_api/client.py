import time
import requests
import asyncio
from .utils import BaseRequest


import asyncio
from .utils import BaseRequest

class ModelverseClient:
    BASE_URL = "https://api.modelverse.cn"

    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def post(self, endpoint, payload, timeout=180):
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.post(url, headers=self.headers, json=payload, timeout=timeout)
        return self._handle_response(response)

    def post_multipart(self, endpoint, data=None, files=None, timeout=180):
        """POST with multipart/form-data. Content-Type is set by requests automatically."""
        url = f"{self.BASE_URL}{endpoint}"
        # Do not set Content-Type explicitly when using files; requests will handle it.
        headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        response = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)
        return self._handle_response(response)

    def get(self, endpoint, params=None, timeout=180):
        url = f"{self.BASE_URL}{endpoint}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        return self._handle_response(response)

    def _handle_response(self, response):
        if response.status_code == 401:
            raise Exception("Unauthorized: Invalid API key")
        
        # For backward compatibility with older error formats
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
                raise Exception(f"API Error: {response_data.get('message', 'Unknown error')}")
            return response_data.get('data', {})
        return response_data

    # --- New Methods for T2V ---
    def submit_task(self, model, task_input, parameters):
        endpoint = "/v1/tasks/submit"
        payload = {
            "model": model,
            "input": task_input,
            "parameters": parameters
        }
        return self.post(endpoint, payload)

    def get_task_status(self, task_id):
        endpoint = f"/v1/tasks/status"
        params = {"task_id": task_id}
        return self.get(endpoint, params=params)
        
    # --- Restored Async Methods for existing nodes ---
    async def async_send_request(self, request: BaseRequest):
        endpoint = request.API_PATH
        # Support multipart form requests when available
        if hasattr(request, "build_multipart") and callable(getattr(request, "build_multipart")):
            data, files = request.build_multipart()
            response = self.post_multipart(endpoint, data=data, files=files)
        else:
            payload = request.build_payload()
            if isinstance(payload, dict) and "seed" in payload:
                payload["seed"] = payload["seed"] % 2147483647 if payload["seed"] != -1 else -1
            response = self.post(endpoint, payload)
        return response.get("data", [])

    async def run_tasks(self, tasks):
        print("INFO:", f"Sending {len(tasks)} request(s) concurrently...")
        results = await asyncio.gather(*tasks)
        return results
