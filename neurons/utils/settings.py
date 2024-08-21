import httpx
import json
import asyncio


async def download_validator_settings():
    url = "https://raw.githubusercontent.com/TensorAlchemy/validator-settings/main/settings.json"
    try:
        # Send a GET request to the URL using httpx asynchronously
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
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


# Example of how to run the async function
async def main():
    result = await download_validator_settings()
    if result:
        print("Successfully downloaded and parsed the settings.")
    else:
        print("Failed to download or parse the settings.")


# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
