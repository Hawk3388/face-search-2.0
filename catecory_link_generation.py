import urllib.parse

def wikipedia_url_to_category_link(article_url, category="Living people"):
    """
    Wandelt eine Wikipedia-Artikel-URL in die exakte Kategorie-Unterseiten-URL um.
    """
    # Titel aus URL extrahieren
    title = article_url.split("/")[-1]  # z.B. "Jane_Mary_Doe"
    title = title.replace("_", " ")     # "Jane Mary Doe"

    # Nachname = letztes Wort, alle anderen Wörter = Vorname(n)
    parts = title.split(" ")
    if len(parts) == 1:
        lastname = parts[0]
        firstname = ""
    else:
        lastname = parts[-1]
        firstname = " ".join(parts[:-1])

    # pagefrom = "Nachname, Vorname(n)\nOriginalname"
    pagefrom_raw = f"{lastname}, {firstname}\n{title}"

    # URL-encode
    pagefrom_encoded = urllib.parse.quote_plus(pagefrom_raw)

    # Kategorie-Link zusammenbauen
    category_url = f"https://en.wikipedia.org/w/index.php?title=Category:{category.replace(' ', '_')}&pagefrom={pagefrom_encoded}#mw-pages"
    return category_url

# Beispiel
article_url = "https://en.wikipedia.org/wiki/John_Doe"
link = wikipedia_url_to_category_link(article_url)
print(link)