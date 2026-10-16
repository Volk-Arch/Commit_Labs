"""какие кадры ещё не залиты: печатает JSON-список путей (не больше 12 за раз), помечает их залитыми."""
import json, os, sys, glob
run = sys.argv[1]; mark = sys.argv[2] == 'mark' if len(sys.argv) > 2 else False
D = f'/home/claude/data/live/{run}'; S = f'{D}/.pushed'
done = set(open(S).read().split()) if os.path.exists(S) else set()
todo = [f for f in sorted(glob.glob(f'{D}/frame_*.json')) if os.path.basename(f) not in done][:12]
if mark:
    with open(S, 'a') as f: f.write('\n'.join(os.path.basename(x) for x in todo) + '\n')
print(json.dumps([dict(op='set', collection='frames', doc_id='f' + os.path.basename(x)[6:10], file_path=x) for x in todo], ensure_ascii=False))
