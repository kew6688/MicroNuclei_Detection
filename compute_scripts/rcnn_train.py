from mn_segmentation.models.mask_rcnn import maskrcnn_mobile, maskrcnn_resnet, maskrcnn_resnet50_fpn

print("===========================================")
print("resnet50fpn")
print("===========================================")
model1 = maskrcnn_resnet50_fpn(weights="DEFAULT")
model1.load_state_dict(torch.load("/content/checkpoints/maskrcnn-resnet50-jitter-v2-60.pt"))
trainer_mrcnn('maskrcnn-resnet50-jitter-v2-60', model1, epoch, transform=get_transform_jitter, dataset_path="/content/mnMask_v2/", cp_path="/content/checkpoints/")