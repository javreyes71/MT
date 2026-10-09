import sumolib
from src.environment.traffic_signal import TrafficSignal
net = sumolib.net.readNet('sumo/caso_estudio.net.xml', withPrograms=True)
for tls_obj in net.getTrafficLights():
    try:
        ts = TrafficSignal(tls_obj.getID(), tls_obj, net)
    except Exception as e:
        print(tls_obj.getID(), repr(e))
