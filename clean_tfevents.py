import os
import tensorflow as tf

def filter_tfevents(input_path, output_dir, max_step):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    writer = tf.compat.v1.summary.FileWriter(output_dir)
    
    for event in tf.compat.v1.train.summary_iterator(input_path):
        if event.step <= max_step:
            writer.add_event(event)
            
    writer.close()
    print(f"Filtrado {input_path} hasta el paso {max_step}. Guardado en {output_dir}")

filter_tfevents("tensorboard/PPO_2/events.out.tfevents.1791410847.DESKTOP-OR9CQ4H.11504.0", "tensorboard/PPO_2/cleaned", 15600000)
