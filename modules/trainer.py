import os
import yaml
import torch
import pickle
import shutil
import itertools
from tqdm import tqdm
from easydict import EasyDict

from data.dataloader import InfiniteDataloader
from .evaluator import Evaluator
from .common import ModelMode, teacher_normalization

class Trainer:

    def __init__(self, config, teacher, student, autoencoder, train_loader, validation_loader, test_set, default_transforms, device, tboard):
        self.config = config
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.device = device
        self.tboard = tboard

        self.default_transforms = default_transforms
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_set = test_set

        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        self._setup_save_dir()
        self._setup_evaluator()

        self._move_models_to_device()

    def train(self):
        self.best_score = 0

        self._pretrain()

        train_loader_infinite = InfiniteDataloader(self.train_loader)

        tqdm_obj = tqdm(range(self.config.trainer.steps))
        for iteration, (image_st, image_ae) in zip(tqdm_obj, train_loader_infinite):
            self.train_loader.dataset.dataset.step_transforms.increment_step()

            image_st = image_st.to(self.device)
            image_ae = image_ae.to(self.device)

            with torch.no_grad():
                teacher_output_st = self.teacher(image_st)
                teacher_output_st = (teacher_output_st - self.teacher_mean) / self.teacher_std
            student_output_st = self.student(image_st)[:, :self.config.net.out_channels]

            ae_output = self.autoencoder(image_ae)
            with torch.no_grad():
                teacher_output_ae = self.teacher(image_ae)
                teacher_output_ae = (teacher_output_ae - self.teacher_mean) / self.teacher_std
            student_output_ae = self.student(image_ae)[:, self.config.net.out_channels:]

            self._compute_teacher_student_loss(teacher_output_st, student_output_st)
            self._compute_ae_student_loss(ae_output, student_output_ae)
            self._compute_teacher_ae_loss(teacher_output_ae, ae_output)
            self._compute_total_loss()

            self.loss_total.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            metric_dict = {
                "Total_Loss": self.loss_total.item(),
                "Loss_ST": self.loss_st.item(),
                "Loss_AE": self.loss_ae.item(),
                "Loss_STAE": self.loss_stae.item(),
                "LR": self.scheduler.get_last_lr()[0]
            }
            self.tboard.add_multiple_scalars(maintag='train', scalars_dict=metric_dict)

            if iteration % 10 == 0:
                tqdm_obj.set_description(
                    "[Train] Loss: {:.4f}  ".format(self.loss_total.item()))
                
            if iteration and iteration % self.config.trainer.test_freq_steps == 0:

                self._test(iteration=iteration)

        self._test(iteration=self.config.trainer.steps)


    def _set_mode(self, mode: ModelMode):
        if mode == ModelMode.TRAIN:
            self.teacher.eval()
            self.student.train()
            self.autoencoder.train()

        elif mode == ModelMode.EVAL:
            self.teacher.eval()
            self.student.eval()
            self.autoencoder.eval()

    def _move_models_to_device(self):
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        self.autoencoder = self.autoencoder.to(self.device)

    def _setup_save_dir(self):
        output_dir = self.config.saver.exp_path
        os.makedirs(output_dir, exist_ok=True)

        def to_dict(d):
            if isinstance(d, EasyDict):
                return {k: to_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [to_dict(v) for v in d]
            return d

        # Save config save
        with open(os.path.join(output_dir, 'config.yml'), "w") as f:
            config_dict = to_dict(self.config)
            yaml.dump(config_dict, f)

        self.output_dir_train = os.path.join(output_dir, 'train')
        os.makedirs(self.output_dir_train, exist_ok=True)

        self.output_dir_train_metrics = os.path.join(self.output_dir_train, 'metrics')
        os.makedirs(self.output_dir_train_metrics, exist_ok=True)

        self.output_dir_test = os.path.join(output_dir, 'test')
        os.makedirs(self.output_dir_test, exist_ok=True)
    
    def _setup_optimizer(self):
        return torch.optim.AdamW(
            itertools.chain(self.student.parameters(), self.autoencoder.parameters()),
            lr=self.config.trainer.lr,
            weight_decay=self.config.trainer.wd
        )
    
    def _setup_scheduler(self):
        scheduler_type = self.config.trainer.lr_scheduler.type
        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, **self.config.trainer.lr_scheduler.kwargs)

        elif scheduler_type == "ExponentialLR":
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, **self.config.trainer.lr_scheduler.kwargs)

        elif scheduler_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config.trainer.lr_scheduler.kwargs)
        
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, **self.config.trainer.lr_scheduler.kwargs)
        
    def _setup_evaluator(self):
        self.evaluator = Evaluator(config=self.config, test_set=self.test_set, default_transform=self.default_transforms, device=self.device, output_dir=self.output_dir_test)

    def _pretrain(self):
        teacher_mean, teacher_std = teacher_normalization(self.teacher, self.train_loader, device=self.device)
        with open(os.path.join(self.output_dir_train, 'teacher_normalization.pkl'), 'wb') as file:
            pickle.dump((teacher_mean, teacher_std), file)

        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std

        self.evaluator.set_teacher_norm(teacher_mean, teacher_std)

        self._test(iteration=0)

    def _compute_teacher_student_loss(self, teacher_output_st, student_output_st):
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        self.loss_st = torch.mean(distance_st[distance_st >= d_hard])

        return self.loss_st

    def _compute_ae_student_loss(self, ae_output, student_output_ae):
        distance_stae = (ae_output - student_output_ae)**2
        self.loss_stae = torch.mean(distance_stae)

        return self.loss_stae

    def _compute_teacher_ae_loss(self, teacher_output_ae, ae_output):
        distance_ae = (teacher_output_ae - ae_output)**2
        self.loss_ae = torch.mean(distance_ae)
        
        return self.loss_ae
    
    def _compute_total_loss(self):
        self.loss_total = self.loss_st + self.loss_ae + self.loss_stae

        return self.loss_total
    
    def _test(self, iteration):
        self._set_mode(ModelMode.EVAL)

        self.evaluator._write_to_log(f"Iteration: #{iteration}", with_print=True)
        au_pro, _, au_roc, _ = self.evaluator.eval(teacher=self.teacher, student=self.student, autoencoder=self.autoencoder, map_norm_loader=self.validation_loader, desc='[Test]')
        self._process_eval_step(iteration=iteration, au_pro=au_pro, au_roc=au_roc, teacher=self.teacher, student=self.student, autoencoder=self.autoencoder)

        self._set_mode(ModelMode.TRAIN)
        
    def _process_eval_step(self, iteration, au_pro, au_roc, teacher, student, autoencoder):
        test_dict = {}
        test_dict['AU-ROC'] = au_roc
        if au_pro:
            test_dict['AU-PRO'] = au_pro

        if test_dict:
            self.tboard.add_multiple_scalars(maintag='val', scalars_dict=test_dict)

        torch.save(student, os.path.join(self.output_dir_train, 'student_last.pth'))
        torch.save(autoencoder, os.path.join(self.output_dir_train, 'autoencoder_last.pth'))

        # current_metric = au_pro if au_pro else au_roc
        current_metric = au_roc

        if current_metric > self.best_score:
            self._save_best(iteration, au_pro, au_roc, teacher, student, autoencoder)

            self.best_score = current_metric
            
    def _save_best(self, iteration, au_pro, au_roc, teacher, student, autoencoder):

        # Save models
        torch.save(teacher, os.path.join(self.output_dir_train, 'teacher.pth'))
        torch.save(student, os.path.join(self.output_dir_train, 'student_best.pth'))
        torch.save(autoencoder, os.path.join(self.output_dir_train, 'autoencoder_best.pth'))

        # Save map norm
        self.evaluator.save_map_norm(path=os.path.join(self.output_dir_train, 'map_normalization_best.pkl'))

        # Save ROC and PR curves
        shutil.copy(os.path.join(self.output_dir_test, 'roc_curve.jpg'), os.path.join(self.output_dir_train_metrics, 'roc_curve_best.jpg'))
        if au_pro:
            shutil.copy(os.path.join(self.output_dir_test, 'pro_curve.jpg'), os.path.join(self.output_dir_train_metrics, 'pro_curve_best.jpg'))

        # Save metrics
        with open(os.path.join(self.output_dir_train_metrics, 'best_iter.txt'), 'w') as f:
            results_str = self.evaluator.to_str()
            f.write(f'Iteration: {iteration}\n')
            f.write(results_str)