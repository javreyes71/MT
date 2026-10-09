c = open('config/large_map.yaml').read()
c = c.replace('delta_time: 5.0', 'route_files: 'sumo/large_routes.rou.xml'\n  delta_time: 5.0')
open('config/large_map.yaml', 'w').write(c)
