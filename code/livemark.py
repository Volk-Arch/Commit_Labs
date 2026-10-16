import sys, os
run, upto = sys.argv[1], int(sys.argv[2]); D = f'/home/claude/data/live/{run}'
with open(f'{D}/.pushed', 'w') as f: f.write('\n'.join(f'frame_{i:04d}.json' for i in range(upto + 1)) + '\n')
