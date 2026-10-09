with open('scripts/evaluate.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('def __init__(self, model_path: str):', 'def __init__(self, model_path: str, config_path: str = "config/default.yaml"):\n        self.config_path = config_path')
c = c.replace('config = load_config("config/default.yaml")', 'config = load_config(self.config_path)')
c = c.replace('RLPolicy(args.model_path)', 'RLPolicy(args.model_path, args.config)')

with open('scripts/evaluate.py', 'w', encoding='utf-8') as f:
    f.write(c)
