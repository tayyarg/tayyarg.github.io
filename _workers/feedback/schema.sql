-- One row per (page, reaction): the running count shown on the page.
CREATE TABLE IF NOT EXISTS counts (
    page     TEXT NOT NULL,
    reaction TEXT NOT NULL,
    n        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (page, reaction)
);

-- One row per reader vote. voter is a salted hash of the reader's IP and the page,
-- so the same reader can't inflate a count and no IP address is stored.
CREATE TABLE IF NOT EXISTS votes (
    page     TEXT NOT NULL,
    reaction TEXT NOT NULL,
    voter    TEXT NOT NULL,
    at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (page, reaction, voter)
);

-- Reader comments. A comment is hidden until it is approved on the /admin page.
CREATE TABLE IF NOT EXISTS comments (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    page     TEXT NOT NULL,
    name     TEXT NOT NULL,
    body     TEXT NOT NULL,
    voter    TEXT NOT NULL,
    at       TEXT NOT NULL DEFAULT (datetime('now')),
    approved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS comments_page ON comments (page, approved, at);
