class Evaluator:
  def __init__(self):
      self.pred_list = []
      self.objects = 0
      self.predictions = 0
      self.TP = 0
      self.FP = 0
      self.FN = 0
      self.precision = 0
      self.recall = 0
      self.f1 = 0
      self.map = 0

  def update(self, pred_masks, pred_scores, gt_masks, ap_iou):
    self.objects += gt_masks.shape[0]-1
    self.predictions += pred_masks.shape[0]
    t_cnt = 0
    for i in range(pred_masks.shape[0]):
      conf = pred_scores[i]
      res = False
      overlap = 0
      for j in range(gt_masks.shape[0]):
        if gt_masks[j].sum()>1000 or gt_masks[j].sum()<5: continue
        intersection = np.logical_and(pred_masks[i], gt_masks[j]).sum()
        union = np.logical_or(pred_masks[i], gt_masks[j]).sum()
        iou = intersection / union
        if intersection > overlap:
          overlap = intersection
          res = True if iou > ap_iou else False
      if res:
        self.TP += 1
        t_cnt += 1
      else:
        self.FP += 1
      self.pred_list.append((conf, res))
    return t_cnt/(gt_masks.shape[0]-1) if (gt_masks.shape[0]-1) > 0 else 1

  def finalize(self):
    self.FN = self.objects - self.TP
    self.pred_list.sort(key=lambda x: x[0], reverse=True)
    correct = 0
    for i in range(len(self.pred_list)):
      if self.pred_list[i][1]:
        correct += 1
      self.map += correct / (i + 1)
    self.map /= self.predictions
    print(f"mAP: {self.map}")

    self.precision = self.TP / (self.TP + self.FP)
    self.recall = self.TP / (self.TP + self.FN)
    self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
    print(f"precision: {self.precision}")
    print(f"recall: {self.recall}")
    print(f"f1: {self.f1}")

    return self.precision, self.recall, self.f1

  def draw_pr_curve(self):
    p = []
    r = []
    correct = 0
    for i in range(len(self.pred_list)):
      if self.pred_list[i][1]:
        correct += 1
      p.append(correct / (i + 1))
      r.append(correct / self.objects)
    plt.plot(r, p)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    print(f"Average Precision: {sum(p)/len(p)}")

def evaluate_nuc_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'labels'))):
    # get final pred, may need customize model predict function
    image = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
    image = np.array(image.convert("RGB"))
    masks = mask_generator.generate(image)
    # print(len(masks))
    # print(masks[0]['segmentation'].shape)
    pred_masks = []
    pred_scores = []
    for mask in masks:
      if mask['bbox'][3] > 900 and mask['bbox'][2] > 900:
        continue
      pred_masks.append(mask['segmentation'])
      pred_scores.append(mask['predicted_iou'])

    # get GT mask
    points = load_polys(os.path.join(dataset_path,'labels',file), target_clas = 5, xy_length=(1408,1040))
    gt_masks = []
    for p in points:
      im = np.zeros((1040,1408),dtype=np.uint8)
      cv2.fillPoly( im, points, 255 )
      gt_masks.append(im)

    # compare and update
    evaluator.update(np.array(pred_masks), np.array(pred_scores), np.array(gt_masks), ap_iou)

  evaluator.finalize()
  evaluator.draw_pr_curve()
  return

def evaluate_mn_dataset(model, dataset_path, evaluator, nms_iou=0.2, conf=0.4, ap_iou=0.5):
  for file in tqdm(os.listdir(os.path.join(dataset_path,'label_masks'))[-200:]):
    # get final pred, may need customize model predict function
    im = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
    image = pil_to_tensor(im)
    pred = model._predict(image)
    pred_boxes,pred_masks,pred_scores = model._post_process(pred, conf)
    pred_boxes = pred_boxes.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    pred_masks = (pred_masks > conf).squeeze(1)       # shape [n,w,h]
    pred_scores = pred_scores.cpu().numpy()           # shape [n]
    if pred_masks.ndim == 2:
      pred_masks = np.expand_dims(pred_masks, axis=0)
    # print(pred_masks.shape)
    # print(pred_scores.shape)

    # get GT mask
    gt_masks = np.load(os.path.join(dataset_path,'label_masks',file))  # shape [n,w,h]
    obj_ids = np.unique(gt_masks)
    gt_masks = (gt_masks == obj_ids[:, None, None])
    # print(gt_masks.shape)

    # compare and update
    recall = evaluator.update(pred_masks, pred_scores, gt_masks, ap_iou)

    # uncomment this can give bad cases that cause recall drop, missing mn
    # if recall < 0.5:
    #   print(recall,file)

  evaluator.finalize()
  # evaluator.draw_pr_curve()
  return

def display_yolo_format_points(file):
    points = load_polys(file, target_clas = 5, xy_length=(1408,1040))
    im = np.zeros((1040,1408),dtype=np.uint8)
    plt.imshow(cv2.fillPoly( im, points, 255 ))

def display_sam_prediction(file):
    img = Image.open(file)
    img = np.array(img.convert("RGB"))
    masks = mask_generator.generate(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.show()

file = "GFP-H2B_A1_1_2023y07m03d_01h02m-1.npy"
model = Application("/content/RCNN.pt")

im = Image.open(os.path.join(dataset_path,'images',file[:-4]+".png"))
image = pil_to_tensor(im)
pred = model._predict(image)
pred_boxes,pred_masks,pred_scores = model._post_process(pred, 0.7)
pred_boxes = pred_boxes.cpu().numpy()
pred_masks = np.squeeze(pred_masks.cpu().numpy())       # shape [n,w,h]
pred_scores = pred_scores.cpu().numpy()      # shape [n]
if pred_masks.ndim == 2:
  pred_masks = np.expand_dims(pred_masks, axis=0)
print(pred_masks.shape)
print(pred_scores.shape)

def get_GT_mask(dataset_path, file):
    # get GT mask
    gt_masks = np.load(os.path.join(dataset_path,'label_masks',file))  # shape [n,w,h]
    obj_ids = np.unique(gt_masks)
    gt_masks = (gt_masks == obj_ids[:, None, None])
    print(gt_masks.shape)
