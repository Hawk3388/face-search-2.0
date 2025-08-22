
import requests
import time

def get_index(index):
    API_URL = "https://en.wikipedia.org/w/api.php"
    CATEGORY = "Living people"
    TARGET_INDEX = index
    limit = index // 10 if index > 1000 else index

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{CATEGORY}",
        "cmlimit": limit,
        "format": "json"
    }

    USER_AGENT = "FaceSearchBot/1.0 (Educational Research; Contact: github.com/Hawk3388/face-search-2.0)"

    headers = {
        "User-Agent": USER_AGENT
    }

    count = 0
    cmcontinue = None

    while True:
        if cmcontinue:
            params["cmcontinue"] = cmcontinue

        response = requests.get(API_URL, params=params, headers=headers)

        # Ensure the response is JSON
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
        else:
            print("⚠️ Response was not JSON!")
            print("HTTP Status:", response.status_code)
            print(response.text[:500])  # print first 500 characters
            return None

        members = data["query"]["categorymembers"]

        for m in members:
            count += 1
            if count == TARGET_INDEX:
                print(f"{TARGET_INDEX}th article: {m['title']}")
                print(f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}")
                target_url = f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}"
                return target_url

        if "continue" not in data:
            print("Index too large, category too short.")
            return None

        cmcontinue = data["continue"]["cmcontinue"]

if __name__ == "__main__":
    # 9 min for all articles
    start_time = time.time()
    index = int(input("Enter article index: "))
    url = get_index(index)
    if url:
        print(f"Found URL for article #{index}: {url}")
        print("Duration:", round(time.time() - start_time, 2))
    else:
        print(f"Article with index {index} not found.")