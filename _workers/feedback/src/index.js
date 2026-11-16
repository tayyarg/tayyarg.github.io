// Reactions and comments for tayyarg.github.io: no login for readers.
//
//   GET  /reactions?page=<key>                       -> { counts: {helpful: 3, ...}, mine: ["helpful"] }
//   POST /reactions {page, reaction, on: true|false}  -> same shape, after adding or removing the vote
//   GET  /comments?page=<key>                        -> { comments: [{id, name, body, at}] }  (approved only)
//   POST /comments {page, name, body, website}        -> { ok: true }  (held until approved)
//
//   GET  /admin                                       -> moderation page; open it as /admin#<ADMIN_TOKEN>
//   GET  /admin/api/comments                          -> pending and recent comments  (Bearer ADMIN_TOKEN)
//   POST /admin/api/comments {id, action}             -> approve, hide or delete one  (Bearer ADMIN_TOKEN)

const REACTIONS = ["helpful", "learned", "confusing"];
const PAGE_KEY = /^[a-z0-9][a-z0-9\-\/]{0,199}$/;
const MAX_NAME = 60;
const MAX_BODY = 4000;
const COMMENTS_PER_HOUR = 5;

export default {
    async fetch(request, env) {
        const url = new URL(request.url);
        const origin = request.headers.get("Origin") || "";
        const allowed = env.ALLOWED_ORIGINS.split(",");
        const cors = {
            "Access-Control-Allow-Origin": allowed.includes(origin) ? origin : allowed[0],
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Vary": "Origin",
        };
        const json = (body, status = 200, headers = cors) =>
            new Response(JSON.stringify(body), { status, headers: { ...headers, "Content-Type": "application/json" } });

        if (url.pathname === "/admin") {
            return new Response(ADMIN_PAGE, {
                headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "X-Robots-Tag": "noindex" },
            });
        }
        if (url.pathname === "/admin/api/comments") return admin(request, env, json);

        if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors });
        if (url.pathname !== "/reactions" && url.pathname !== "/comments") return json({ error: "not found" }, 404);

        let input = {};
        if (request.method === "GET") {
            input.page = url.searchParams.get("page");
        } else if (request.method === "POST") {
            if (!allowed.includes(origin)) return json({ error: "origin not allowed" }, 403);
            try { input = await request.json(); } catch { return json({ error: "bad json" }, 400); }
        } else {
            return json({ error: "method not allowed" }, 405);
        }
        if (typeof input.page !== "string" || !PAGE_KEY.test(input.page)) return json({ error: "bad page" }, 400);

        return url.pathname === "/reactions" ? reactions(request, env, input, json) : comments(request, env, input, json);
    },
};

async function reactions(request, env, { page, reaction, on }, json) {
    const voter = await readerId(request, env, page);

    if (request.method === "POST") {
        if (!REACTIONS.includes(reaction) || typeof on !== "boolean") return json({ error: "bad reaction" }, 400);
        // Change the vote, and the count only if the vote actually changed.
        const change = on
            ? await env.DB.prepare("INSERT OR IGNORE INTO votes (page, reaction, voter) VALUES (?, ?, ?)").bind(page, reaction, voter).run()
            : await env.DB.prepare("DELETE FROM votes WHERE page = ? AND reaction = ? AND voter = ?").bind(page, reaction, voter).run();
        if (change.meta.changes > 0) {
            await env.DB.prepare(
                "INSERT INTO counts (page, reaction, n) VALUES (?1, ?2, MAX(?3, 0)) " +
                "ON CONFLICT (page, reaction) DO UPDATE SET n = MAX(n + ?3, 0)"
            ).bind(page, reaction, on ? 1 : -1).run();
        }
    }

    const [counts, mine] = await env.DB.batch([
        env.DB.prepare("SELECT reaction, n FROM counts WHERE page = ?").bind(page),
        env.DB.prepare("SELECT reaction FROM votes WHERE page = ? AND voter = ?").bind(page, voter),
    ]);
    const result = Object.fromEntries(REACTIONS.map((r) => [r, 0]));
    for (const row of counts.results) if (row.reaction in result) result[row.reaction] = row.n;
    return json({ counts: result, mine: mine.results.map((row) => row.reaction) });
}

async function comments(request, env, { page, name, body, website }, json) {
    if (request.method === "GET") {
        const { results } = await env.DB.prepare(
            "SELECT id, name, body, at FROM comments WHERE page = ? AND approved = 1 ORDER BY at"
        ).bind(page).all();
        return json({ comments: results });
    }

    // website is a hidden field that people never see; bots fill it in. Pretend it worked.
    if (website) return json({ ok: true });
    name = typeof name === "string" ? name.trim().slice(0, MAX_NAME) : "";
    body = typeof body === "string" ? body.trim() : "";
    if (body.length < 2 || body.length > MAX_BODY) return json({ error: "bad body" }, 400);

    const voter = await readerId(request, env, "comments");
    const recent = await env.DB.prepare(
        "SELECT COUNT(*) AS n FROM comments WHERE voter = ? AND at > datetime('now', '-1 hour')"
    ).bind(voter).first();
    if (recent.n >= COMMENTS_PER_HOUR) return json({ error: "too many comments" }, 429);

    await env.DB.prepare("INSERT INTO comments (page, name, body, voter) VALUES (?, ?, ?, ?)")
        .bind(page, name || "Anonymous", body, voter).run();
    return json({ ok: true });
}

async function admin(request, env, json) {
    const headers = { "Cache-Control": "no-store" };
    const token = (request.headers.get("Authorization") || "").replace(/^Bearer /, "");
    if (!env.ADMIN_TOKEN || !(await sameSecret(token, env.ADMIN_TOKEN))) return json({ error: "unauthorized" }, 401, headers);

    if (request.method === "POST") {
        const { id, action } = await request.json();
        const statements = {
            approve: "UPDATE comments SET approved = 1 WHERE id = ?",
            hide: "UPDATE comments SET approved = 0 WHERE id = ?",
            delete: "DELETE FROM comments WHERE id = ?",
        };
        if (!(action in statements) || !Number.isInteger(id)) return json({ error: "bad action" }, 400, headers);
        await env.DB.prepare(statements[action]).bind(id).run();
    }
    const { results } = await env.DB.prepare(
        "SELECT id, page, name, body, at, approved FROM comments ORDER BY approved, at DESC LIMIT 200"
    ).all();
    return json({ comments: results }, 200, headers);
}

// A salted hash of the reader's IP address, so no IP is ever stored.
async function readerId(request, env, scope) {
    const ip = request.headers.get("CF-Connecting-IP") || "";
    return hex(await sha256(`${env.VOTER_SALT}|${ip}|${scope}`)).slice(0, 32);
}

async function sameSecret(a, b) {
    const [x, y] = await Promise.all([sha256(a), sha256(b)]);
    return crypto.subtle.timingSafeEqual(x, y);
}

async function sha256(text) {
    return crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
}

function hex(buffer) {
    return [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const ADMIN_PAGE = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comment moderation</title>
<style>
  body { font: 15px/1.5 -apple-system, system-ui, sans-serif; max-width: 860px; margin: 2em auto; padding: 0 16px; color: #222; background: #fafafa; }
  .c { background: #fff; border: 1px solid #ddd; border-left: 4px solid #d24d57; border-radius: 4px; padding: 10px 14px; margin: 12px 0; }
  .c.approved { border-left-color: #5cb85c; }
  .meta { color: #666; font-size: 13px; }
  .body { white-space: pre-wrap; margin: 8px 0; }
  button { margin-right: 6px; cursor: pointer; }
</style></head>
<body>
<h1>Comments</h1>
<p id="status">Loading…</p>
<div id="list"></div>
<script>
  var token = location.hash.slice(1);
  var status = document.getElementById('status');
  function call(method, body) {
    return fetch('/admin/api/comments', {
      method: method, body: body && JSON.stringify(body),
      headers: { 'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json' }
    }).then(function (r) { if (!r.ok) throw new Error(r.status === 401 ? 'Wrong or missing token: open this page as /admin#<token>.' : 'Error ' + r.status); return r.json(); });
  }
  function render(data) {
    var pending = data.comments.filter(function (c) { return !c.approved; }).length;
    status.textContent = pending + ' waiting for approval, ' + (data.comments.length - pending) + ' published.';
    var list = document.getElementById('list');
    list.innerHTML = '';
    data.comments.forEach(function (c) {
      var el = document.createElement('div');
      el.className = 'c' + (c.approved ? ' approved' : '');
      var meta = document.createElement('div'); meta.className = 'meta';
      meta.textContent = c.name + ' · ' + c.page + ' · ' + c.at + ' UTC · ' + (c.approved ? 'published' : 'pending');
      var body = document.createElement('div'); body.className = 'body'; body.textContent = c.body;
      el.append(meta, body);
      (c.approved ? ['hide', 'delete'] : ['approve', 'delete']).forEach(function (action) {
        var b = document.createElement('button'); b.textContent = action;
        b.onclick = function () { call('POST', { id: c.id, action: action }).then(render).catch(fail); };
        el.append(b);
      });
      list.append(el);
    });
  }
  function fail(e) { status.textContent = e.message; }
  call('GET').then(render).catch(fail);
</script>
</body></html>`;
