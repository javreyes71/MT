import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os

net_path = "sumo/network.net.xml"
out_path = r"C:\Users\Ress\.gemini\antigravity\brain\143539f9-bc7c-4d8d-8a84-dce094d84ea5\large_map_visual.png"

tree = ET.parse(net_path)
root = tree.getroot()

fig, ax = plt.subplots(figsize=(10, 10))

for edge in root.findall('edge'):
    if 'function' in edge.attrib and edge.attrib['function'] == 'internal':
        continue
    for lane in edge.findall('lane'):
        shape = lane.attrib.get('shape')
        if shape:
            points = [tuple(map(float, p.split(','))) for p in shape.split()]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, color='#2c3e50', linewidth=1)

ax.set_aspect('equal')
ax.axis('off')
plt.title("Topología del Mapa: Osorno (Large Map)", fontweight='bold', fontsize=14)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print("Mapa guardado en", out_path)
