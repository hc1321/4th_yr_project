import os
import json


json_folder = "Videos_for_overlaying/overlays"


conversion_list = []


for file in sorted(os.listdir(json_folder)):
    if file.endswith("_lines.json"):  
        json_path = os.path.join(json_folder, file)
    
        with open(json_path, "r") as f:
            data = json.load(f)

        if "conversion_factor" in data:
            current_val = data["conversion_factor"]
        
            if current_val != 0:
                new_val = 1 / current_val
            else:
                new_val = None  

            conversion_list.append({"overlay": file.replace("_lines.json", ""), "new_conversion_factor": new_val})

            data["conversion_factor"] = new_val

            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            
            print(f"[INFO] Updated {file}: New conversion factor = {new_val:.13f}")
        else:
            print(f"[WARNING] No 'conversion_factor' found in {file}")


print("\nUpdated Conversion Factors")
for item in conversion_list:
    print(f"{item['overlay']}: {item['new_conversion_factor']:.6f}")


