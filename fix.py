with open('figures/fig_biopsy_leak.py', 'r') as f:
    c = f.read()

c = c.replace('for part in ["imgs_part_1", "imgs_part_2", "imgs_part_3"]:',
              'for part in ["imgs_part_1/imgs_part_1", "imgs_part_2/imgs_part_2", "imgs_part_3/imgs_part_3", "imgs_part_1", "imgs_part_2", "imgs_part_3"]:')

with open('figures/fig_biopsy_leak.py', 'w') as f:
    f.write(c)

