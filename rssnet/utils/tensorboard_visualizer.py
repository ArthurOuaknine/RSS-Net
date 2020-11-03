"""Class to create Tensorboard Visualization during training"""
from torch.utils.tensorboard import SummaryWriter

class TensorboardVisualizer:

    def __init__(self, writer):
        self.writer = writer

    def update_train_loss(self, loss, iteration):
        self.writer.add_scalar('losses/train', loss,
                               iteration)

    def update_multi_train_loss(self, global_loss, rd_loss, ra_loss, iteration):
        self.writer.add_scalar('train_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_doppler', rd_loss,
                               iteration)
        self.writer.add_scalar('train_losses/range_angle', ra_loss,
                               iteration)

    def update_val_loss(self, loss, iteration):
        self.writer.add_scalar('losses/validation', loss,
                               iteration)

    def update_multi_val_loss(self, global_loss, rd_loss, ra_loss, iteration):
        self.writer.add_scalar('validation_losses/global', global_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler', rd_loss,
                               iteration)
        self.writer.add_scalar('validation_losses/range_angle', ra_loss,
                               iteration)

    def update_learning_rate(self, lr, iteration):
        self.writer.add_scalar('parameters/learning_rate', lr, iteration)

    def update_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('losses/validation', metrics['loss'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Mean', metrics['acc'],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Background',
                               metrics['acc_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Pedestrian',
                               metrics['acc_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Cyclist',
                               metrics['acc_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelAccuracy/Car',
                               metrics['acc_by_class'][3],
                               iteration)
        self.writer.add_scalar('PixelRecall/Mean', metrics['recall'],
                               iteration)
        self.writer.add_scalar('PixelRecall/Background',
                               metrics['recall_by_class'][0],
                               iteration)
        self.writer.add_scalar('PixelRecall/Pedestrian',
                               metrics['recall_by_class'][1],
                               iteration)
        self.writer.add_scalar('PixelRecall/Cyclist',
                               metrics['recall_by_class'][2],
                               iteration)
        self.writer.add_scalar('PixelRecall/Car',
                               metrics['recall_by_class'][3],
                               iteration)
        self.writer.add_scalar('MIoU/Mean', metrics['miou'],
                               iteration)
        self.writer.add_scalar('MIoU/Background',
                               metrics['miou_by_class'][0],
                               iteration)
        self.writer.add_scalar('MIoU/Pedestrian',
                               metrics['miou_by_class'][1],
                               iteration)
        self.writer.add_scalar('MIoU/Cyclist',
                               metrics['miou_by_class'][2],
                               iteration)
        self.writer.add_scalar('MIoU/Car',
                               metrics['miou_by_class'][3],
                               iteration)

    def update_multi_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('validation_losses/global',
                               (1/2)*(metrics['range_doppler']['loss']+metrics['range_angle']['loss']),
                               iteration)
        self.writer.add_scalar('validation_losses/range_doppler', metrics['range_doppler']['loss'],
                               iteration)
        self.writer.add_scalar('validation_losses/range_angle', metrics['range_angle']['loss'],
                               iteration)

        self.writer.add_scalar('Range_Doppler_metrics/PixelAccuracy',
                               metrics['range_doppler']['acc'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/PixelRecall',
                               metrics['range_doppler']['recall'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/MIoU',
                               metrics['range_doppler']['miou'],
                               iteration)
        self.writer.add_scalar('Range_Doppler_metrics/Dice',
                               metrics['range_doppler']['dice'],
                               iteration)

        self.writer.add_scalar('Range_angle_metrics/PixelAccuracy',
                               metrics['range_angle']['acc'],
                               iteration)
        self.writer.add_scalar('Range_angle_metrics/PixelRecall',
                               metrics['range_angle']['recall'],
                               iteration)
        self.writer.add_scalar('Range_angle_metrics/MIoU',
                               metrics['range_angle']['miou'],
                               iteration)
        self.writer.add_scalar('Range_angle_metrics/Dice',
                               metrics['range_angle']['dice'],
                               iteration)

    def update_detection_val_metrics(self, metrics, iteration):
        self.writer.add_scalar('AveragePrecision/Mean', metrics['map'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Pedestrian',
                               metrics['map_by_class']['pedestrian'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Cyclist',
                               metrics['map_by_class']['cyclist'],
                               iteration)
        self.writer.add_scalar('AveragePrecision/Car',
                               metrics['map_by_class']['car'],
                               iteration)

    def update_img_masks(self, pred_grid, gt_grid, iteration):
        self.writer.add_image('Predicted_masks', pred_grid, iteration)
        self.writer.add_image('Ground_truth_masks', gt_grid, iteration)

    def update_multi_img_masks(self, rd_pred_grid, rd_gt_grid,
                               ra_pred_grid, ra_gt_grid, iteration):
        self.writer.add_image('Range_Doppler/Predicted_masks', rd_pred_grid, iteration)
        self.writer.add_image('Range_Doppler/Ground_truth_masks', rd_gt_grid, iteration)
        self.writer.add_image('Range_angle/Predicted_masks', ra_pred_grid, iteration)
        self.writer.add_image('Range_angle/Ground_truth_masks', ra_gt_grid, iteration)
