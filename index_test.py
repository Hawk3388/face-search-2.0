import requests

API_URL = "https://en.wikipedia.org/w/api.php"
CATEGORY = "Living people"
TARGET_INDEX = 105468  # z.B. den 5468. Eintrag

params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": f"Category:{CATEGORY}",
    "cmlimit": "1000",   # max 500 pro Request
    "format": "json"
}

headers = {
    "User-Agent": "MyWikiScript/1.0 (https://github.com/deinname)"
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
        break

    members = data["query"]["categorymembers"]

    for m in members:
        count += 1
        if count == TARGET_INDEX:
            print(f"{TARGET_INDEX}. Artikel: {m['title']}")
            print(f"https://en.wikipedia.org/wiki/{m['title'].replace(' ', '_')}")
            exit()

    if "continue" not in data:
        print("Index zu groß, Kategorie zu kurz.")
        break

    cmcontinue = data["continue"]["cmcontinue"]