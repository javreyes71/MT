import sumolib
net=sumolib.net.readNet('sumo/caso_estudio.net.xml')
out=[]
for tls in net.getTrafficLights():
    names=set()
    for links in tls.getLinks().values():
        for link in links:
            if link[0].getEdge().getName(): names.add(link[0].getEdge().getName())
            if link[1].getEdge().getName(): names.add(link[1].getEdge().getName())
    out.append((tls.getID(), names))

for o in out:
    s = str(o[1])
    if any(n in s for n in ['Mackenna', 'Rodriguez', 'Freire', 'Republica', 'República', 'Portales', 'Suarez', 'Suárez', 'Concepcion', 'Concepción', 'Victoria']):
        print(o[0], o[1])
