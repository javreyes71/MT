import sumolib
net = sumolib.net.readNet('sumo/caso_estudio.net.xml', withPrograms=True)
count = 0
for tls_obj in net.getTrafficLights():
    progs = tls_obj.getPrograms()
    if not progs: continue
    prog = progs.get("0", next(iter(progs.values())))
    n_greens = sum(1 for p in prog.getPhases() if ("G" in p.state or "g" in p.state) and "y" not in p.state)
    if n_greens < 2: continue
    count += 1
print("Valid TLS:", count)
