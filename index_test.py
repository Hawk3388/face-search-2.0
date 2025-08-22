import requests
import random

def get_index(index):
    API_URL = "https://en.wikipedia.org/w/api.php"
    CATEGORY = "Living people"
    TARGET_INDEX = index  # z.B. den 5468. Eintrag

    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{CATEGORY}",
        "cmlimit": "1000",   # max 500 pro Request
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

        # hier sicherstellen, dass JSON zurückkommt
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
        else:
            print("⚠️ Antwort war kein JSON!")
            print("HTTP Status:", response.status_code)
            print(response.text[:500])  # ersten 500 Zeichen ausgeben
            return None

        members = data["query"]["categorymembers"]

        for m in members:
            count += 1
            if count == TARGET_INDEX:
                print(f"{TARGET_INDEX}. Artikel: {m['title']}")
                print(f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}")
                target_url = f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}"
                return target_url

        if "continue" not in data:
            print("Index zu groß, Kategorie zu kurz.")
            return None

        cmcontinue = data["continue"]["cmcontinue"]

if __name__ == "__main__":
    index = random.randint(1, 1000000)
    url = get_index(index)
    if url:
        print(f"Gefundene URL für den {index}. Artikel: {url}")
    else:
        print(f"Artikel mit Index {index} nicht gefunden.")