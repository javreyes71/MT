with open('scripts/plot_metrics.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('f"tabla_{level}.png"', 'f"tabla_{level}_v2.png"')
c = c.replace('"tabla_global.png"', '"tabla_global_v2.png"')

with open('scripts/plot_metrics.py', 'w', encoding='utf-8') as f:
    f.write(c)
