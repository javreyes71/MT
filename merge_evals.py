import pandas as pd
import os

f1 = "results_large/evaluation.csv"
f2 = "results_large/evaluation_rl.csv"

if os.path.exists(f1) and os.path.exists(f2):
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv(f1, index=False)
    print("Merged successfully!")
    
    # Generate new plots
    import sys
    sys.path.append(os.getcwd())
    from scripts.plot_metrics import generate_all_plots
    generate_all_plots(f1, "results_large/plots")
