import tensorflow  as tf


def vis_image_with_bboxes_gt_det(image_batch, bbox_gt=None, bbox_det=None):
    pass


def image_batch_with_gt(image_batch, label_batch):
    vis_batch = image_batch + 1.0
    vis_batch /= 2.0
    vis_batch *= 255.0

    numbboxes = label_batch['numbboxes']
    batchsize = tf.shape(image_batch)[0]
    bboxes_gt = label_batch['object_bboxes']
    bboxes_gt = tf.unstack(bboxes_gt, axis=0)
    pass