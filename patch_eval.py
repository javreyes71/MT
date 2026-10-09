
import re
with open('scripts/evaluate.py', 'r', encoding='utf-8') as f:
    c = f.read()

new_func = '''def run_episode(env: TrafficSumoEnv, policy, seed: int) -> dict:
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    
    total_teleports = 0
    total_inference_time = 0.0

    is_native = getattr(policy, \'type_id\', None) is not None
    if is_native:
        for tid in env.tls_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tid)
                if logics:
                    logic = logics[0]
                    logic.type = policy.type_id
                    traci.trafficlight.setProgramLogic(tid, logic)
            except Exception:
                pass

    while True:
        t0 = time.perf_counter()
        if hasattr(policy, \'model\'):
            actions_dict = policy.get_actions(env.tls_ids, env.signals, obs, env)
        else:
            actions_dict = policy.get_actions(env.tls_ids, env.signals)
        
        inference_time = time.perf_counter() - t0
        total_inference_time += inference_time

        if actions_dict is None:
            action_list = None
        else:
            action_list = [actions_dict.get(tid, 0) for tid in env.tls_ids]
            
        obs, reward, terminated, truncated, info = env.step(action_list)
        
        total_teleports += traci.simulation.getStartingTeleportNumber()
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    return {
        \'total_reward\': total_reward, 
        \'steps\': steps, 
        \'seed\': seed,
        \'teleports\': total_teleports,
        \'avg_inference_ms\': (total_inference_time / steps) * 1000 if steps > 0 else 0.0
    }'''

c = re.sub(r'def run_episode.*?return \{.*?\n    \}', new_func, c, flags=re.DOTALL)

with open('scripts/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(c)

