
import json
import os 

dir = '/home/y3229wan/scratch/KateData_coco/images/'
# print(len(os.listdir(dir)))

# Opening JSON file
f = open('/home/y3229wan/scratch/KateData_coco/result.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
# print(data[0].keys())
# dict_keys(['id', 'annotations', 'drafts', 'predictions', 'data', 'meta', 'created_at', 'updated_at', 'inner_id', 'total_annotations', 'cancelled_annotations', 
#           'total_predictions', 'comment_count', 'unresolved_comment_count', 'last_comment_updated_at', 'project', 'updated_by', 'comment_authors'])
categories = data['categories']
print(categories)

print(f"total images labeled are: {len(os.listdir(dir))}, total labels are: {len(data['annotations'])}")

mn_cnt = 0
apop_cnt = 0
nuc_cnt = 0
div_nuc_cnt = 0
for i in data['annotations']:
    if i["category_id"] == 7:
        mn_cnt += 1
    elif i["category_id"] == 6:
        apop_cnt += 1
    elif i["category_id"] == 5:
        nuc_cnt += 1
    elif i["category_id"] == 4:
        div_nuc_cnt += 1
    # if not i["segmentation"]:
    #     print("no segmentation")
print(f"nuc # is {nuc_cnt}, mn # is {mn_cnt}, apop # is {apop_cnt}, dividing cell # is {div_nuc_cnt}")
 
# Closing file
f.close()

# subprocess.run(["scp", FILE, "USER@SERVER:PATH"])
#e.g. subprocess.run(["scp", "foo.bar", "joe@srvr.net:/path/to/foo.bar"])