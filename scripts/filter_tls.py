import sumolib
import argparse
import yaml

def analyze_tls(net_file, output_yaml):
    net = sumolib.net.readNet(net_file, withPrograms=True)
    all_tls = net.getTrafficLights()
    
    # 5 corredores núcleo de Osorno
    corridors = [
        "Avenida Juan Mackenna",
        "Los Carrera",
        "Manuel Rodr\u00edguez",  # Rodríguez
        "Avenida Alcalde Ren\u00e9 Soriano B\u00f3rquez", # René Soriano
        "Avenida Rep\u00fablica",
        "Avenida Julio Buschmann"
    ]
    
    controlled_tls = []
    
    for tls in all_tls:
        tls_id = tls.getID()
        progs = tls.getPrograms()
        if not progs:
            continue
            
        prog = progs.get('0', next(iter(progs.values())))
        phases = prog.getPhases()
        
        green_phases = sum(1 for p in phases if ('G' in p.state or 'g' in p.state) and 'y' not in p.state)
        
        if green_phases < 2:
            continue
            
        # Check if connected to any of the corridors
        is_in_corridor = False
        for conn in tls.getConnections():
            incoming_lane = conn[0]
            edge_name = incoming_lane.getEdge().getName()
            if edge_name and any(c in edge_name for c in corridors):
                is_in_corridor = True
                break
                
        if is_in_corridor:
            controlled_tls.append(tls_id)
            
    print(f"Seleccionados {len(controlled_tls)} semáforos controlables en corredores.")
    
    with open(output_yaml, 'w') as f:
        yaml.dump({'simulation': {'controlled_tls': controlled_tls}}, f, default_flow_style=False)
    print(f"Guardado en {output_yaml}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', default='sumo/network.net.xml')
    parser.add_argument('--out', default='config/controlled_tls.yaml')
    args = parser.parse_args()
    analyze_tls(args.net, args.out)
