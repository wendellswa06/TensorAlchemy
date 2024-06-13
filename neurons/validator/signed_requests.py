import time
import base64
from typing import Optional

import requests
import bittensor as bt

from substrateinterface import Keypair
from loguru import logger


class SignedRequests:
    def __init__(
        self,
        hotkey: Optional[Keypair] = None,
        wallet: Optional[bt.wallet] = None,
        validator: Optional["StableValidator"] = None,
    ):
        if validator is not None:
            self.hotkey = validator.wallet.hotkey

        elif wallet is not None:
            self.hotkey = wallet.hotkey

        elif hotkey is not None:
            self.hotkey = hotkey

        else:
            raise ValueError(
                #
                "SignedRequests requires one of "
                + "[hotkey, wallet, validator]"
            )

    def sign_message(self, message: str) -> str:
        signature = self.hotkey.sign(message.encode())
        return base64.b64encode(signature).decode()

    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        **kwargs,
    ):
        return self._signed_request(
            "GET",
            url,
            params=params,
            **kwargs,
        )

    def post(
        self,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        **kwargs,
    ):
        return self._signed_request(
            "POST",
            url,
            data=data,
            json=json,
            **kwargs,
        )

    def put(
        self,
        url: str,
        data: Optional[dict] = None,
        **kwargs,
    ):
        return self._signed_request(
            "PUT",
            url,
            data=data,
            **kwargs,
        )

    def delete(self, url: str, **kwargs):
        return self._signed_request("DELETE", url, **kwargs)

    def _signed_request(
        self,
        method: str,
        url: str,
        data: Optional[dict] = None,
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        **kwargs,
    ):
        try:
            timestamp = str(int(time.time()))
            message = f"{method} {url}?timestamp={timestamp}"

            signature = self.sign_message(message)

            headers = kwargs.get("headers", {})
            headers.update({"X-Signature": signature, "X-Timestamp": timestamp})
            kwargs["headers"] = headers
        except Exception as e:
            logger.error(
                "Exception raised while signing request; sending plain old request"
            )

        return requests.request(
            method, url, params=params, data=data, json=json, **kwargs
        )
