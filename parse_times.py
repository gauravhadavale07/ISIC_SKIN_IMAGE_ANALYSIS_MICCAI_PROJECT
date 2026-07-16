import re
import glob

for f_name in glob.glob('*.log'):
    log = open(f_name).read()
    blocks = log.split('COMMENCING RUN WITH SEED: ')
    if len(blocks) > 1:
        print(f"File: {f_name}")
        for i, block in enumerate(blocks[1:]):
            seed = block.split('\n')[0].strip()
            matches = re.findall(r'100%\|.*?\[(\d+):(\d+)<00:00', block)
            total_time = sum(int(m)*60 + int(s) for m,s in matches)
            print(f"  Block {i}, Seed: {seed}, Time: {total_time}s, Matches: {len(matches)}")
