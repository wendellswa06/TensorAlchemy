import requests
import ecdsa
import base64
import time
from substrateinterface import Keypair, KeypairType


class SignedRequests:
    def __init__(self, hotkey: Keypair):
        self.hotkey=hotkey

    def sign_message(self, message: str) -> str:
        signature = self.hotkey.sign(message.encode())
        return base64.b64encode(signature).decode()

    def get(self, url: str, params: dict = None, **kwargs):
        return self._signed_request("GET", url, params=params, **kwargs)

    def post(self, url: str, data: dict = None, json: dict = None, **kwargs):
        return self._signed_request("POST", url, data=data, json=json, **kwargs)

    def put(self, url: str, data: dict = None, **kwargs):
        return self._signed_request("PUT", url, data=data, **kwargs)

    def delete(self, url: str, **kwargs):
        return self._signed_request("DELETE", url, **kwargs)

    def _signed_request(
        self,
        method: str,
        url: str,
        params: dict = None,
        data: dict = None,
        json: dict = None,
        **kwargs,
    ):
        timestamp = str(int(time.time()))
        message = f"{method} {url}?timestamp={timestamp}"

        signature = self.sign_message(message)

        headers = kwargs.get("headers", {})
        headers.update({"X-Signature": signature, "X-Timestamp": timestamp})
        kwargs["headers"] = headers

        return requests.request(
            method, url, params=params, data=data, json=json, **kwargs
        )
