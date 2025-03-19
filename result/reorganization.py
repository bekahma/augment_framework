import os
import shutil

# Navigate to the result folder
base_dir = "result"
os.chdir(base_dir)

for subfolder in os.listdir():
    subfolder_path = os.path.join(base_dir, subfolder)
    print("exploring "+subfolder)

    # Ensure it's a directory
    if subfolder.startswith("eval_prompt"):

        # Loop through all result files
        for filename in os.listdir(subfolder):
            
            if filename.startswith("result_") and "_eval_prompt" in filename:
                print("moving")
                # Extract model name
                model_name = filename.split("result_")[1].split("_eval_prompt")[0]
                
                # Create model directory if it doesn't exist
                #model_dir = os.path.join(base_dir, model_name)
                os.makedirs(model_name, exist_ok=True)
                
                # Move file to model directory
                shutil.move(os.path.join(subfolder, filename), model_name)
                
