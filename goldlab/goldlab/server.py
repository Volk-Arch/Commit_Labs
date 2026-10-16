"""ПУЛЬТ. Сервер на стандартной библиотеке: python -m goldlab.server [--port 8765]
API: GET /api/catalog, GET /api/world?kind=&rho=&L=, POST /api/run {exp, params, region}, GET /api/jobs, GET /api/runs, GET /api/run/<id>.
Опыты считаются в фоновой очереди (по одному), результат сохраняется в runs/.
Экспорт просмотрщика: python -m goldlab.server --export viewer.html [--runs all|selftest|id1,id2] — страница с вшитыми прогонами."""
import json, os, sys, threading, queue, time, traceback
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from . import experiments as E, store, worlds as Wd

UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ui', 'index.html')
JOBS = {}; Q = queue.Queue(); LOCK = threading.Lock()

def worker():
    while True:
        jid = Q.get(); job = JOBS[jid]
        with LOCK: job['status'] = 'running'; job['started'] = time.time()
        try:
            res = E.run(job['exp'], job['params'], job['region']); rid = store.save(res)
            with LOCK: job.update(status='done', run=rid, seconds=res['seconds'])
        except Exception as ex:
            with LOCK: job.update(status='error', error=f'{type(ex).__name__}: {ex}', trace=traceback.format_exc()[-1500:])

def world_json(kind, rho, L):
    w = Wd.make(kind, rho=rho, L=L); return {'t': w.t.round(4).tolist(), 'x': w.x.round(4).tolist(), 'n': w.n, 'L': L, 'kind': kind, 'rho': rho}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, obj, code=200, ctype='application/json'):
        body = obj if isinstance(obj, (bytes, bytearray)) else json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header('Content-Type', ctype + ('; charset=utf-8' if 'json' in ctype or 'html' in ctype else ''))
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                with open(UI, 'rb') as f: return self.send(f.read().replace(b'/*BOOT*/', b'window.GOLDLAB_MODE="live";'), ctype='text/html')
            if u.path == '/api/catalog': return self.send(E.catalog())
            if u.path == '/api/world': return self.send(world_json(q.get('kind', 'gold11'), float(q.get('rho', 4)), float(q.get('L', 12))))
            if u.path == '/api/jobs':
                with LOCK: return self.send([{k: v for k, v in j.items() if k != 'trace'} | {'id': i} for i, j in sorted(JOBS.items(), reverse=True)][:30])
            if u.path == '/api/runs': return self.send(store.list_runs())
            if u.path.startswith('/api/run/'): return self.send(store.load(u.path.split('/')[-1]))
            return self.send({'error': 'нет такого пути'}, 404)
        except Exception as ex:
            return self.send({'error': str(ex)}, 500)
    def do_POST(self):
        u = urlparse(self.path); n = int(self.headers.get('Content-Length', 0)); body = json.loads(self.rfile.read(n) or b'{}')
        if u.path == '/api/run':
            if body.get('exp') not in E.REG: return self.send({'error': 'нет такого опыта'}, 400)
            jid = time.strftime('%H%M%S') + f'-{len(JOBS)}'
            with LOCK: JOBS[jid] = {'exp': body['exp'], 'params': body.get('params', {}), 'region': body.get('region'), 'status': 'queued', 'queued': time.time()}
            Q.put(jid); return self.send({'job': jid})
        return self.send({'error': 'нет такого пути'}, 404)

def export(out, which='all'):
    runs = store.list_runs()
    if which == 'selftest': runs = [r for r in runs if r['id'].endswith('selftest')]
    elif which != 'all': ids = set(which.split(',')); runs = [r for r in runs if r['id'] in ids]
    seen, full = set(), []
    for r in runs:                                                     # по одному последнему прогону каждого опыта
        if r['exp'] in seen: continue
        seen.add(r['exp']); full.append(store.load(r['id']))
    data = {'catalog': E.catalog(), 'runs': full, 'world': world_json('gold11', 4.0, 12.0)}
    with open(UI, encoding='utf-8') as f: html = f.read()
    boot = 'window.GOLDLAB_MODE="static";window.GOLDLAB_DATA=' + json.dumps(data, ensure_ascii=False).replace('</', '<\\/') + ';'
    html = html.replace('/*BOOT*/', boot)
    if out.endswith('.artifact.html'):                                 # для публикации: обёртку документа даёт хостинг
        import re
        for pat in (r'<!doctype html>\s*', r'<html[^>]*>\s*', r'</?head>\s*', r'<body>\s*', r'</body>\s*', r'</html>\s*', r'<meta charset[^>]*>\s*', r'<meta name="viewport"[^>]*>\s*'):
            html = re.sub(pat, '', html, flags=re.I)
    with open(out, 'w', encoding='utf-8') as f: f.write(html)
    print(f'просмотрщик: {out} — опытов {len(full)}')

def main():
    if '--export' in sys.argv:
        out = sys.argv[sys.argv.index('--export') + 1]; which = sys.argv[sys.argv.index('--runs') + 1] if '--runs' in sys.argv else 'all'
        return export(out, which)
    port = int(sys.argv[sys.argv.index('--port') + 1]) if '--port' in sys.argv else 8765
    threading.Thread(target=worker, daemon=True).start()
    print(f'пульт: http://localhost:{port}', flush=True); ThreadingHTTPServer(('0.0.0.0', port), H).serve_forever()

if __name__ == '__main__':
    main()
