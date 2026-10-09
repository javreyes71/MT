import sumolib
import yaml
import sys

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')

net = sumolib.net.readNet('sumo/caso_estudio.net.xml', withPrograms=True)

target_streets = ["mackenna", "rodriguez", "freire", "republic", "portales", "suarez", "concepcion", "victoria"]
target_streets.extend(["repblica", "surez", "concepcin"])

controlled_tls = set()

for tls in net.getTrafficLights():
    tls_id = tls.getID()
    is_target = False
    
    for links in tls.getLinks().values():
        for link in links:
            from_name = (link[0].getEdge().getName() or "").lower()
            to_name = (link[1].getEdge().getName() or "").lower()
            
            for street in target_streets:
                if street in from_name or street in to_name:
                    is_target = True
                    break
            if is_target: break
        if is_target: break
            
    if is_target:
        progs = tls.getPrograms()
        if progs:
            prog = progs.get("0", next(iter(progs.values())))
            n_greens = sum(1 for p in prog.getPhases() if ("G" in p.state or "g" in p.state) and "y" not in p.state)
            if n_greens >= 2:
                controlled_tls.add(tls_id)

tls_list = sorted(list(controlled_tls))
print(f"Found {len(tls_list)} traffic lights on the target avenues.")

with open('config/controlled_tls.yaml', 'w', encoding='utf-8') as f:
    yaml.dump({"simulation": {"controlled_tls": tls_list}}, f)

print("Saved to config/controlled_tls.yaml")
