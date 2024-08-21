import httpx
import json


def download_validator_settings():
    url = "https://raw.githubusercontent.com/TensorAlchemy/validator-settings/main/settings.json"

    try:
        # Send a GET request to the URL using httpx
        with httpx.Client() as client:
            response = client.get(url)

        # Check if the request was successful
        response.raise_for_status()

        # Parse the JSON content
        data = response.json()

        # Print the downloaded JSON data
        print("Downloaded JSON data:")
        print(json.dumps(data, indent=2))

        return data

    except httpx.RequestError as e:
        print(f"An error occurred while downloading the JSON: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing the JSON: {e}")
        return None
