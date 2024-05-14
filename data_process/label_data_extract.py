import subprocess
import json
print("help")
# Opening JSON file
f = open('project-9-at-2024-05-10-14-08-8629092e.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
print(data.keys())
# for i in data['emp_details']:
#     print(i)
 
# Closing file
f.close()

# subprocess.run(["scp", FILE, "USER@SERVER:PATH"])
#e.g. subprocess.run(["scp", "foo.bar", "joe@srvr.net:/path/to/foo.bar"])