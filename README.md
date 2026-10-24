# Face Search

Upload an image, select a face, and retrieve candidates from an existing face database.
The main objective is finding the **same person**. A search result is a candidate,
not proof of identity; similarity to other people can still produce false matches.

The Streamlit app supports JPG, JPEG, PNG and WebP, multiple-face selection, result
previews, source links, an API backend, and a local fallback. JSON and SQLite now
use the same search implementation and accept the same image/embedding requests.

## Installation

Use Python **3.10–3.12**. Python 3.10 is the locally tested configuration.

For this checkout, use the existing **`face-search2` Conda environment**:

```sh
conda activate face-search2
python -m streamlit run main.py
# In another terminal with the same environment:
python main_server_sql.py
```

Keep the existing working dlib/PyTorch installation. You do not need to replace
the environment or reinstall all dependencies to use the changes. The following
virtual-environment instructions are for a fresh installation only.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py
```

For smaller installations:

| File | Components |
| --- | --- |
| `requirements-ui.txt` | Remote-only Streamlit frontend |
| `requirements-server.txt` | API and dlib face encoder |
| `requirements-crawler.txt` | Wikipedia crawler and face encoder |
| `requirements-training.txt` | PyTorch training and dataset preparation |
| `requirements-dev.txt` | Tests, UI, server and crawler helpers; no model download |

The complete original package list is retained in `requirements-legacy.txt` for
reference. Dependency ranges are declared in `pyproject.toml`; this is not a
cross-platform lockfile. The face_recognition fork remains pinned to its original
commit. CUDA requires matching local PyTorch/dlib installations; CPU mode is supported.
The optional `encoder-wheel` extra provides the former Windows dlib-bin wheel for
fresh installations; do not install it over the working dlib 20.0.0 in `face-search2`.

## Configuration and startup

Use `config.example.env` as a template for `.env`. Preserve any existing credentials.
Streamlit also accepts `API_URL` and `TOKEN` from its secrets configuration.

```dotenv
TOKEN=your-long-random-token
API_URL=http://127.0.0.1:7403
SEARCH_METRIC=cosine
SEARCH_TOLERANCE=0.07
```

Start either backend:

```sh
python main_server.py                  # SQLite (face_embeddings.db)
python main_server_sql.py              # Same SQLite backend, legacy command
python main_server.py --backend json   # Optional JSON server (face_embeddings.json)
```

Both work with the current databases **without migration**. Paths default to the
project directory; `DATABASE_PATH` overrides the path. To use SQLite locally in
Streamlit, `LOCAL_BACKEND=sqlite` is now the default. If the API is unavailable the UI falls back
to that local database. Remote-only installations need the encoder extra for local fallback.

Both entry points now serve the same FastAPI application. For production, use an
ASGI server such as Uvicorn:

```sh
python -m uvicorn main_server:create_app --factory --host 127.0.0.1 --port 7403 --no-proxy-headers
```

Configure its listening address/port to match `API_URL`. ASGI references
`main_server:app` and `main_server_sql:app` also work. Waitress/WSGI commands must
be replaced by Uvicorn/ASGI commands; the HTTP endpoints remain compatible.
For a JSON ASGI server, set `SERVER_BACKEND=json` and use `main_server:create_app`.
If `DATABASE_PATH` is explicitly configured, point it at a file matching the selected
backend; otherwise the appropriate `.db` / `.json` default is used.
Each worker owns its own search snapshot; a reload affects the receiving worker.
Use one worker, restart all workers together, or coordinate reloads externally.
Terminate HTTPS at the deployment proxy. The development command does not enable HTTPS.

The API refuses to start without a token unless `ALLOW_UNAUTHENTICATED=1` is
explicitly set. The development runner then binds to localhost. Never expose an
unauthenticated ASGI deployment publicly.

The database loads during ASGI lifespan startup. Image encoding, search, reloads
and SQLite counters run in worker threads rather than blocking the event loop.
This does not make the face model itself faster. `/docs`, `/redoc` and `/openapi.json`
provide public API documentation without database contents; the interactive docs
accept the bearer token through **Authorize**. `/health` remains public. All data
endpoints retain token authentication and the existing JSON error envelope.
The default runner disables forwarded-header trust. Keep `--no-proxy-headers` when
running Uvicorn directly, or configure trusted proxies deliberately at deployment.

Default limits: 10 MiB image bytes, 20 million decoded pixels, 20 faces, 100 results,
and 120 analysis/search/reload requests per peer IP per minute per worker. Configure
these with the variables in `config.example.env` and `REQUESTS_PER_MINUTE` (0 disables
the request-rate limit). Face analysis admits one request per worker; a busy worker
returns 429. A proxy can provide deployment-wide limits.

## API compatibility

All endpoints except `/health` require `Authorization: Bearer <TOKEN>`.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Readiness, loaded face count, model, metric and capabilities; 503 if unavailable |
| `POST /analyze` | Multipart `image`; returns boxes and encodings for face selection |
| `POST /search` | Multipart `image` + optional `selected_face`, or JSON `encoding` |
| `POST /reload` | Build a new snapshot; retain the previous working snapshot on failure |
| `GET /stats` | Database counts and persistent request count |

```json
{
  "encoding": ["128 numeric values for the default encoder"],
  "model_id": "dlib-128-large-v1",
  "format": "objects",
  "metric": "euclidean",
  "tolerance": 0.5,
  "max_results": 20,
  "dedup": true
}
```

The example's encoding placeholder must be replaced with actual numbers.
`format=objects` returns dictionaries containing distance, image URL, source pages,
face ID and optional bounding box. No database embeddings are returned with matches.
For older clients, the JSON server defaults to `[entry, distance]` pairs; the SQL
server defaults to objects. Either server can return either format.
Legacy `debug`, `dedup`, and `full_scan` controls remain accepted. `full_scan` returns
up to the configured result cap, rather than an unbounded response.

The default metric is now **cosine distance with tolerance 0.07 in all modes**,
preserving the earlier JSON server's metric. To use the earlier local/SQLite
behavior, set `SEARCH_METRIC=euclidean` and `SEARCH_TOLERANCE=0.5`.
Neither threshold is claimed to be calibrated for your
deployment. The legacy `similarity` field remains for compatibility; it is **not a
probability**. The UI displays distance, with smaller values meaning closer matches.

Search still scans every embedding for the selected model. A partial selection
sorts only the leading candidates and expands when image duplicates consume slots;
distance ties retain database order. Norms and boundary calculations use float64.
Small candidate sets are reranked in float64; large sets refine near-zero distances
and threshold boundaries in bounded chunks. This improves numerical consistency,
including exact self-matches, without changing the configured matching threshold
or claiming improved identity accuracy from the face model itself.

## Data protection and migration

Existing databases and model weights are never automatically removed. Readers do
not repair or overwrite JSON files. A complete legacy open JSON array can be read
in memory; a truncated object fails explicitly. Large JSON arrays are streamed.

The new SQLite schema retains `images`, its legacy embedding column, and
`wiki_pages`. It adds `faces` (one row per face/model/vector) and transactional crawl
progress. Multiple people in one image no longer collapse into one embedding.

Stop writers before maintenance. Every schema migration first creates a uniquely
named `.backup-...` file using SQLite's online backup API, then verifies its integrity.
Committed WAL changes are included. Existing backups are never overwritten.

```sh
python -m tools.migrate_database face_embeddings.db
python -m tools.json_to_sql --source face_embeddings.json --output face_embeddings.db
```

The importer also backs up an already migrated destination before writing. It
preserves all distinct face embeddings for the same URL and is safe to repeat for
identical entries. Migration cannot reconstruct faces discarded by older importers;
import the original multi-face JSON or recrawl to recover those.

To restore a backup: stop readers and writers, keep the current database and its
sidecars for diagnosis, and point `DATABASE_PATH` at the verified backup or a new
copy of it. Do not overwrite a running SQLite database or mix old WAL files with a
restored file. Test restoration on a separate path before deployment.

### SQLite cleanup

```sh
# Read-only report (also the default when running tools/clean_db.py):
python -m tools.clean_db --database face_embeddings.db
# Stop database writers, then explicitly apply the reported URL-alias cleanup:
python -m tools.clean_db --database face_embeddings.db --apply
```

The cleaner groups original/thumbnail URL aliases in a linear pass instead of
comparing every image hash against every other hash. Identical pHashes are reported
for review only: even equal hashes do not justify deleting a face or merging distinct
image URLs. Ordinary URL query parameters remain significant, while recognized
Wikimedia still-image thumbnails on both upload/thumb hosts resolve to the original.
Document pages and video thumbnails are not collapsed to a shared file URL.

`--apply` first creates a unique integrity-checked backup, migrates a legacy schema
if needed (with its own backup), and merges URL aliases transactionally. Distinct
face embeddings and all source page links are retained. Only identical face rows
are deduplicated. Groups with conflicting face metadata stay separate. Extra schema
columns or references that the cleaner cannot preserve cause an explicit failure.
Existing crawl queues/progress are retained. No database file or backup is deleted,
and no automatic VACUUM rewrites the file. Without `--database`, `DATABASE_PATH`
or the project SQLite default is used. Import legacy JSON via `tools.json_to_sql`
before cleaning; the old JSON overwrite/pHash-deletion workflow is removed.

After applying cleanup, restart the server or call `/reload` to rebuild its snapshot.

## Crawling

```sh
conda activate face-search2
python crawler/crawler_wiki.py
# Bounded test run, using a NEW database:
python crawler/crawler_wiki.py --page "Albert Einstein" --database crawl-test.db
# One category pass, or a bounded number of article discoveries:
python crawler/crawler_wiki.py --once
python crawler/crawler_wiki.py --max-articles 20
# Other sources also write SQLite:
python crawler/crawler_url.py https://example.org/ --max-pages 100
python crawler/common_crawler.py https://example.org/ --index CC-MAIN-2025-30
```

`crawler_wiki.py`, `crawler_wiki_sql.py` and `crawler_wiki_thread.py` now start the
same SQLite implementation. Existing JSON files are retained and are never rewritten
by a crawler. Import them with the command above if their contents are missing from
your SQLite database. Set `CRAWLER_DATABASE_PATH` or `--database` for another destination.
Run only one crawler against a given database; a process lock prevents competing runs.

Wikipedia discovery uses [Categorymembers](https://www.mediawiki.org/wiki/API:Categorymembers)
and the [Images generator](https://www.mediawiki.org/wiki/API:Images), including all API
continuation pages, embedded template/gallery images, and Commons-hosted files. It
does not follow category HTML previous-page links. Image metadata is queried in
batches of up to 50. Category cursors and article jobs are committed together;
image jobs survive article completion and failed downloads receive persistent,
exponentially delayed retries (up to one day between attempts). `--once` processes
the category pass and currently due jobs; future retries stay queued for the next
run. Without `--once`, retries are checked every 30 seconds and discovery repeats
every 24 hours by default. `--page` runs finish after the selected articles.

Three download workers prefetch a bounded number of images with reusable HTTP
sessions. Completed downloads are processed in completion order, and free slots
are refilled before face analysis so one slow download does not hold up ready images.
At most twice the worker count is queued or downloaded ahead, plus the image being
analyzed. In-flight URLs are excluded from rescheduling until their result or retry
is committed; unfinished jobs remain pending after a stop. MediaWiki metadata
requests remain serial. Requests are paced per host;
HTTP 429/Retry-After and API maxlag cause backoff. Face extraction and SQLite writes
use one worker, avoiding duplicated model memory and competing database writes.
960px previews reduce transfer and detection costs. If no face or a small face is
detected, the crawler tries the original within configured limits, or a larger
preview. A face detected in the first pass is retained if the second finds fewer.
Both passes only locate faces. Descriptors are computed once, using the existing
face locations on the selected image; images without faces skip descriptor generation.
All detected faces, bounding boxes, pixel area and crop contrast are recorded.
Quality measurements are diagnostic values, not identity confidence scores.

Wikipedia SHA-1 plus the encoder/preprocessing configuration determines whether an
image needs analysis again; negative face results are cached too. Legacy rows have
no verified source SHA-1, so the first new crawl analyzes those images once. Changed
files are reprocessed; historical face rows are retained, not deleted. This means an
old embedding can remain searchable after a remote file changes at the same URL.
The article list itself is refreshed each pass; only unchanged image analysis is skipped.

The general URL crawler also discovers lazy `data-src` / `data-original`, `srcset`,
`picture`, `noscript`, and social preview images, including extensionless URLs. It
follows same-host links and observes robots.txt for live page/image requests. Common
Crawl reads archived HTML but still fetches image bytes from the live image URL.
Web image cache entries refresh daily, since these sources provide no Wikipedia
file SHA-1. JavaScript-only galleries and images shown only after login are not rendered.
Choose an available Common Crawl index explicitly; the example is not an automatic
latest-index selector. A source article is provenance, not a verified identity label.

Before the first write to an existing SQLite database, the crawler creates a unique,
integrity-checked backup, including WAL contents. New queue tables are additive.
Periodic backups require new face/provenance writes and elapsed time (24h by default),
not a count of reused images. Insufficient disk space stops the crawler; it never
deletes databases or old backups. Ctrl+C requests a graceful stop; unfinished jobs
remain pending. After a hard stop, committed transactions and queued work remain.

Tune `CRAWLER_DOWNLOAD_WORKERS`, `CRAWLER_PREVIEW_WIDTH`, `CRAWLER_REQUEST_INTERVAL`,
`CRAWLER_REFRESH_HOURS`, `CRAWLER_BACKUP_HOURS`, and `CRAWLER_MIN_FREE_MB` in `.env`.
The Wikipedia crawler retains `NUM_JITTERS` (10 by default). An optional
`CRAWLER_NUM_JITTERS=1` reduces encoding cost but changes descriptor sampling; measure
retrieval quality before using it. The cache profile includes this setting.
Use a descriptive `USER_AGENT` with your contact/project information.
The HTTP client trusts the operating system's CA roots as well as Requests' CA bundle,
including managed Windows roots, and keeps certificate/hostname verification enabled.
An explicit `REQUESTS_CA_BUNDLE` remains authoritative if your network needs a custom CA.
Server snapshots refresh via `/reload` or restart; the crawler does not reload them.

## Models and evaluation

The default encoder remains dlib/face_recognition with 128 dimensions. The older
TinyFaceNet is a face/non-face image classifier, not an identity embedding model.
The Siamese model in `model2` produces 512-dimensional normalized embeddings.

Prepare **new** face crops without modifying the original dataset:

```sh
python -m model2.prepare_dataset --input model2/dataset --output model2/prepared
python -m model2.train --dataset model2/prepared --epochs 20 --batch-size 32
```

Preparation rejects images with zero/multiple faces and removes exact decoded-image
duplicates. Labels still need manual quality control. Training uses disjoint people
for training, validation and test (at least six people with two usable images each).
Pairs are resampled each epoch; validation pairs remain fixed. Frozen BatchNorm
layers retain their statistics. Validation loss selects checkpoints, with learning
rate reduction and early stopping. No test identities select the checkpoint.

Each run writes to a new `runs/siamese-...` directory: `best.pth`, `last.pth`, and
`split.json`. Checkpoints include preprocessing, validation threshold, optimizer,
scheduler and random-generator states. Resume with `--resume runs/.../last.pth`;
use the same dataset and seed, and a larger `--epochs` value. Existing model files
are not overwritten. The older classifier's best-weight snapshot is also fixed.

Measure both distances on separate calibration/test people:

```sh
python -m tools.evaluate_search --dataset model2/prepared --split runs/siamese-RUN/split.json --output runs/dlib-evaluation.json
python -m tools.evaluate_search --dataset model2/prepared --checkpoint runs/siamese-RUN/best.pth --split runs/siamese-RUN/split.json --output runs/siamese-evaluation.json
python -m model2.test first.jpg second.jpg --checkpoint model2/siamese_model.pth
```

For a bounded pilot run on the existing dataset, add `--max-people 40
--images-per-person 5`. Sampling uses a fixed seed; the selected calibration and
test identities are recorded in the report. This is a pilot, not a substitute for
validation on deployment data.

The local pilot in `face-search2` used 40 identities and 170 usable images from
200 sampled images. On its 20 held-out test identities (67 query images), cosine
retrieved the correct identity first in 59 cases, Euclidean in 58. At separately
calibrated thresholds, both had 1.1% false acceptance and 11.1% false rejection on
the generated test pairs. This small difference supports cosine as a provisional
default, not a claim of general superiority. The pilot thresholds are **not**
automatically applied to the production database. The full local report is
`runs/dlib-pilot-face-search2.json` (excluded from Git because it contains dataset paths).

Reports include calibration/test false acceptance and false rejection rates,
Recall@1/10, skipped images and encoding time. Small sets cannot establish reliable
rare-error rates. Compare results on representative held-out data before changing
the production encoder or threshold; the software does not claim improved accuracy.

To use a Siamese checkpoint, build a separate index with matching preprocessing:

```sh
python -m tools.reindex_database --source face_embeddings.db --output face_embeddings_siamese.db --checkpoint runs/siamese-RUN/best.pth
```

This downloads source images and can take substantial time. It refuses an existing
output path and stops on failure, retaining partial output for diagnosis. Use the
output only after successful completion. Configure `FACE_ENCODER=siamese`,
`MODEL_PATH`, and `DATABASE_PATH` together. Model IDs include the checkpoint hash;
the API rejects queries from a different encoder. Existing plain state dictionaries
can be loaded, but their training preprocessing was not recorded: evaluate the
face-crop pipeline before using those old weights in production.

## Testing and privacy

```sh
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check face_search model2 tools crawler main.py main_server.py main_server_sql.py tests
```

Tests use temporary databases and mocked encoders, plus optional PyTorch tests.
They cover API contracts, input validation, WAL backups, migration rollback,
multiple faces per image, streaming JSON, calibration and Streamlit face selection.
Crawler tests additionally cover API pagination, Commons files, lazy HTML images,
persistent failures, negative-cache invalidation, atomic queue/face transactions,
bounded parallel downloads, stopping, disk-space checks and failed backups.

Uploads and query embeddings are not deliberately written to the face database.
They remain in the current Streamlit session until cleared/replaced or the session
ends; framework/proxy buffering and retention depend on deployment. `/analyze`
returns face embeddings to the frontend, so protect that connection. Result images
are loaded from their source hosts. Avoid logging uploaded images or encodings.
Use images you are authorized to process and respect source usage restrictions.

Licensed under Apache License 2.0; see `LICENSE`.
