"""Rank Wikipedia person articles by support from distinct result images."""

from collections import defaultdict
import re
import ssl
from urllib.parse import unquote, urlsplit

import requests

from crawler.wiki_api import SystemTrustAdapter

from .search import normalize_image_url


WIKI_HOST = re.compile(r"([a-z][a-z0-9-]*)(?:\.m)?\.wikipedia\.org")
ITEM_ID = re.compile(r"Q[1-9][0-9]*")
USER_AGENT = "FaceSearchBot/2.0 (https://github.com/Hawk3388/face-search-2.0)"
MAX_ARTICLES = 50
MAX_WIKIS = 4


def wikipedia_article(url):
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
    except ValueError:
        return None
    host = WIKI_HOST.fullmatch(parsed.netloc.lower())
    if parsed.scheme not in ("http", "https") or not host or not parsed.path.startswith("/wiki/"):
        return None
    title = " ".join(unquote(parsed.path.removeprefix("/wiki/")).replace("_", " ").split())
    if title and "|" not in title and len(title.encode("utf-8")) <= 255:
        return host[1], title
    return None


def wiki_json(url, **params):
    # Only public article titles and item IDs are sent to Wikimedia.
    context = ssl.create_default_context()
    context.load_verify_locations(requests.certs.where())
    with requests.Session() as session:
        session.mount("https://", SystemTrustAdapter(context))
        with session.get(
            url,
            params={"format": "json", "formatversion": 2, **params},
            headers={"User-Agent": USER_AGENT},
            timeout=(2, 3),
            allow_redirects=False,
        ) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Unexpected Wikimedia response")
            data = response.json()
    if not isinstance(data, dict) or "error" in data:
        raise ValueError("Wikimedia lookup failed")
    return data


def wikipedia_people(articles):
    """Map (language, title) to (Wikidata item, name) for verified person pages."""
    by_wiki = defaultdict(list)
    for lang, title in articles:
        by_wiki[lang].append(title)
    if len(articles) > MAX_ARTICLES or len(by_wiki) > MAX_WIKIS:
        return {}

    article_items = {}
    for lang, titles in by_wiki.items():
        query = wiki_json(
            f"https://{lang}.wikipedia.org/w/api.php",
            action="query",
            prop="pageprops",
            ppprop="wikibase_item",
            redirects=1,
            titles="|".join(titles),
        )["query"]
        aliases = {row["from"]: row["to"] for key in ("normalized", "redirects") for row in query.get(key, [])}
        pages = {
            page["title"]: page.get("pageprops", {}).get("wikibase_item") for page in query["pages"] if page["ns"] == 0
        }
        for original in titles:
            title = original
            seen = set()
            while title in aliases and title not in seen:
                seen.add(title)
                title = aliases[title]
            item = pages.get(title)
            if item and ITEM_ID.fullmatch(item):
                article_items[lang, original] = (item, title)
    if not article_items:
        return {}

    labels = {}
    entities = wiki_json(
        "https://www.wikidata.org/w/api.php",
        action="wbgetentities",
        props="labels|claims",
        ids="|".join(sorted({item for item, _ in article_items.values()})),
        languages="|".join(dict.fromkeys(["en", "mul", *by_wiki])),
    )["entities"]
    for item, entity in entities.items():
        for statement in entity.get("claims", {}).get("P31", []):
            value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
            if statement.get("rank") != "deprecated" and isinstance(value, dict) and value.get("id") == "Q5":
                labels[item] = entity.get("labels", {})
                break

    people = {}
    for article, (item, title) in article_items.items():
        if item in labels:
            translations = labels[item]
            label = translations.get("en") or translations.get("mul") or translations.get(article[0])
            people[article] = (item, label["value"] if label else title)
    return people


def suggest_person(entries, lookup=wikipedia_people):
    article_images = defaultdict(set)
    for entry in entries:
        try:
            image_url = normalize_image_url(entry.get("image_url"))
        except ValueError:
            continue
        if not image_url:
            continue
        for page in entry.get("page_urls") or [entry.get("page_url")]:
            if article := wikipedia_article(page):
                article_images[article].add(image_url)
                if len(article_images) > MAX_ARTICLES:
                    return None
    if len(set().union(*article_images.values())) < 2:
        return None

    people = lookup(tuple(sorted(article_images)))
    person_images = defaultdict(set)
    names = {}
    for article, images in article_images.items():
        if article in people:
            item, name = people[article]
            person_images[item].update(images)
            names[item] = name
    ranks = sorted(person_images, key=lambda item: len(person_images[item]), reverse=True)
    if not ranks:
        return None
    first = ranks[0]
    support = len(person_images[first])
    if support < 2 or (len(ranks) > 1 and len(person_images[ranks[1]]) == support):
        return None
    return names[first], support
