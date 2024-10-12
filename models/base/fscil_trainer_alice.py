import torch
from utils import *
import torch.nn as nn
from tqdm import tqdm
from .base import Trainer
from dataloader.data_utils import *
from dataloader.mixed_ups import puzzle_mix
from models.backbones.network import Encoder
from functorch import make_functional_with_buffers
from utils import ensure_path, restart_from_checkpoint
from sklearn.metrics.pairwise import pairwise_distances

class NCMValidation(object):
    def __init__(self, args, model):
        self.model = model
        self.args = args
        self.train_set, self.train_dataloader, self.val_set, self.val_dataloader = get_validation_dataloader(args)

    def _retrieval(self):
        """Extract features from validation split and search on train split features."""
        cls_wise_feature_prototype = []
        avg_cls = []
        embedding_list = []
        label_list = []
        validation_embedding_list = []
        validation_label_list = []

        self.model.eval()
        torch.cuda.empty_cache()

        # --- using training data to generate average feature embedding for each class ---
        print('acquiring class-wise feature prototype from training data ...')
        with torch.no_grad():
            tqdm_gen = tqdm(self.train_dataloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = self.model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        # generate the average feature with all data
        for index in range(self.args.base_class):
            class_index = (label_list == index).nonzero()
            embedding_this = embedding_list[class_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0, keepdims=True).cuda()
            cls_wise_feature_prototype.append(embedding_this)
            avg_cls.append(index)
        for i in range(len(cls_wise_feature_prototype)):
            cls_wise_feature_prototype[i] = cls_wise_feature_prototype[i].view(-1)
        proto_list = torch.stack(cls_wise_feature_prototype, dim=0).cpu()
        proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=-1)

        # --- acquire feature for each validation data ---
        print('acquiring feature prototype for testing data ...')
        with torch.no_grad():
            tqdm_gen = tqdm(self.val_dataloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = self.model(data)
                validation_embedding_list.append(embedding.cpu())
                validation_label_list.append(label.cpu())
        validation_embedding_list = torch.cat(validation_embedding_list, dim=0).cpu()
        validation_embedding_list = torch.nn.functional.normalize(validation_embedding_list, p=2, dim=-1)
        validation_label_list = torch.cat(validation_label_list, dim=0).cpu()

        # --- calculate the cosine similarity for each validation data ---
        # metric: euclidean, cosine, l2, l1
        pairwise_distance = pairwise_distances(self.args.temperature * np.asarray(validation_embedding_list), self.args.temperature * np.asarray(proto_list), metric='cosine')
        prediction_result = np.argmin(pairwise_distance, axis=1)
        validation_label_list = np.asarray(validation_label_list)
        top1 = np.sum(prediction_result == validation_label_list) / float(len(validation_label_list))
        return top1

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        self.validation_ncm = NCMValidation(self.args, self.model.encoder)
        self.min_eig, self.max_eig = None, None
        self.min_eigs, self.max_eigs = [], []
        self.acc = []

    def train(self):
        load_self_pretrain_weights(self.args, self.model)
        for session in range(self.args.start_session, self.args.sessions):
            train_set, trainloader = get_train_dataloader(self.args, session)
            if session == 0:  # load base class train img label
                print('The classes contained in this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                if os.path.exists(self.args.resume):
                    to_restore = {"epoch": 0}
                    restart_from_checkpoint(
                        self.args.resume,
                        run_variables=to_restore,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler
                    )
                    start_epoch = to_restore["epoch"] + 1
                else:
                    start_epoch = 0
                for epoch in range(start_epoch, self.args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl = self.base_train(self.model, trainloader, optimizer, scheduler, session, epoch, self.args)
                    # test model with all seen class
                    val_ncm_acc = self.validation_ncm._retrieval()
                    # save better model
                    if (val_ncm_acc * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (val_ncm_acc * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
                        self.save_checkpoint(epoch, self.model, optimizer, scheduler, val_ncm_acc, save_model_dir, 'Saving the best model!')
                    lrc = scheduler.get_last_lr()[0]
                    print('epoch:%03d, lr:%.4f, training_loss:%.5f, val_acc:%.5f' % (epoch, lrc, tl, val_ncm_acc))
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % ((time.time() - start_time) * (self.args.epochs_base - epoch) / 60))
                    scheduler.step()
                    if self.args.save_ntk:
                        self.calculate_and_save_linear_NTK(train_set, val_ncm_acc)

                # test session 0
                self.args.num_cls = self.args.base_class + self.args.way * session
                self.model = self.get_cls_backbone()
                # freeze all layers but the last fc
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
                best_path = os.path.join(self.args.save_path, 'session0_max_acc.pth')
                checkpoint = torch.load(best_path, map_location="cpu")
                state_dict = checkpoint['model']
                new_state_dict = dict()
                for old_key, value in state_dict.items():
                    if old_key.startswith('backbone') and 'fc' not in old_key:
                        new_key = old_key.replace('backbone.', '')
                        new_state_dict[new_key] = value
                self.model.load_state_dict(new_state_dict, strict=False)
                print("=> loaded pre-trained model '{}'".format(best_path))
                self.model = self.model.cuda()
                trainset, train_loader, testset, testloader = get_incremental_dataset_fs(self.args, session=session)
                print('length of the trainset: {0}'.format(len(trainset)))
                print('----------------------------- calculate and store average class-wise feature embedding -----------------------------')
                cls_wise_feature_prototype = []
                cls_label = []
                transform = testloader.dataset.transform
                cls_avg_feature, cls_avg_feature_index = self.calculate_avg_feature_for_each_cls(trainset, transform, self.model, self.args)
                for i in range(len(cls_avg_feature)):
                    cls_wise_feature_prototype.append(cls_avg_feature[i])
                    cls_label.append(cls_avg_feature_index[i])
                feature_save_dir = os.path.join(self.args.save_path, 'cls_wise_avg_feature_{0}.pth'.format(session))
                torch.save(dict(class_feature=cls_wise_feature_prototype, class_id=cls_label), feature_save_dir)
                print('----------------------------- do interference -----------------------------')
                save_path = os.path.join(self.args.save_path, 'result_{0}.txt'.format(session))
                prediction_result, label_list = self.test_NCM(self.model, testloader, self.args, cls_wise_feature_prototype, save_path)
            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                trainset, train_loader, testset, testloader = get_incremental_dataset_fs(self.args, session=session)
                print('length of the trainset: {0}'.format(len(trainset)))
                print('----------------------------- calculate and store average class-wise feature embedding -----------------------------')
                cls_wise_feature_prototype = []
                cls_label = []
                transform = testloader.dataset.transform
                self.args.num_cls = self.args.base_class + self.args.way * session
                cls_avg_feature, cls_avg_feature_index = self.calculate_avg_feature_for_each_cls(trainset, transform, self.model, self.args)
                for i in range(len(cls_avg_feature)):
                    cls_wise_feature_prototype.append(cls_avg_feature[i])
                    cls_label.append(cls_avg_feature_index[i])
                feature_save_dir = os.path.join(self.args.save_path, 'cls_wise_avg_feature_{0}.pth'.format(session))
                torch.save(dict(class_feature=cls_wise_feature_prototype, class_id=cls_label), feature_save_dir)
                save_path = os.path.join(self.args.save_path, 'result_{0}.txt'.format(session))
                prediction_result, label_list = self.test_NCM(self.model, testloader, self.args, cls_wise_feature_prototype, save_path)

    def save_checkpoint(self, epoch, model, optimizer, scheduler, ncm_acc, filename, msg):
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if ncm_acc is not None:
            state['ncm_acc'] = ncm_acc
        torch.save(state, filename)
        print(msg)

    def calculate_and_save_linear_NTK(self, train_set, val_ncm_acc, sample_num=3):
        self.model.eval()
        fnet, params, buffers = make_functional_with_buffers(self.model.projector.fc)

        def fnet_single(params, buffers, x):
            return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

        def empirical_ntk_jacobian_contraction(params, buffers, x1, x2, compute='trace'):
            einsum_expr = {'full': 'Naf,Mbf->NMab', 'trace': 'Naf,Maf->NM', 'diagonal': 'Naf,Maf->NMa'}.get(compute)
            jac1, jac2 = (torch.vmap(torch.func.jacrev(fnet_single), (None, None, 0))(params, buffers, x) for x in [x1, x2])
            result = torch.stack([torch.einsum(einsum_expr, j1.flatten(2), j2.flatten(2)) for j1, j2 in zip(jac1, jac2)]).sum(0)
            return result

        def get_min_max_eigvals(new_set, sample_num):
            _, new_loader = get_new_dataloader(self.args, new_set)
            new_samples, _ = next(iter(new_loader))
            old_samples, _ = next(iter(old_loader))
            old_embeddings = self.model.backbone(old_samples.cuda())[:sample_num]
            new_embeddings = self.model.backbone(new_samples.cuda())[:sample_num]
            ntk_matrix = empirical_ntk_jacobian_contraction(params, buffers, old_embeddings, new_embeddings)
            eigvals = torch.linalg.eigh(ntk_matrix)[0]
            return eigvals.min().item(), eigvals.max().item()

        old_sampler = CategoriesSampler(train_set.targets, 1, self.args.base_class, 1)
        old_loader = torch.utils.data.DataLoader(dataset=train_set, batch_sampler=old_sampler, num_workers=8, pin_memory=True)
        min_eigvals, max_eigvals = zip(*[get_min_max_eigvals(i, sample_num) for i in range(1, self.args.sessions)])
        self.min_eigs.append(np.mean(min_eigvals))
        self.max_eigs.append(np.mean(max_eigvals))
        self.acc.append(val_ncm_acc)
        for attr, name in zip([self.min_eigs, self.max_eigs, self.acc], ['min', 'max', 'acc']):
            torch.save(attr, os.path.join(self.args.save_path, f"{name}.pth"))
        self.model.train()

    def get_cls_backbone(self):
        backbone_mapping = {
            'resnet18': {
                'cifar100': ('models.backbones.resnet18_width', 'ResNet18', {'width': self.args.resnet_width}),
                'miniimagenet': ('models.backbones.resnet18_width', 'ResNet18', {'width': self.args.resnet_width}),
                'cub200': ('models.backbones.resnet_cub', 'ResNet18', {}),
                'imagenet100': ('models.backbones.resnet18_width', 'ResNet18', {'width': self.args.resnet_width}),
            },
            'resnet12': {
                'cifar100': ('models.backbones.resnet12_width', 'ResNet12', {'width': self.args.resnet_width}),
                'miniimagenet': ('models.backbones.resnet12_width', 'ResNet12', {'width': self.args.resnet_width}),
                'cub200': ('models.backbones.resnet12_width', 'ResNet12', {'width': self.args.resnet_width}),
                'imagenet100': ('models.backbones.resnet12_width', 'ResNet12', {'width': self.args.resnet_width}),
            },
            'vit_tiny': {
                'cifar100': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'miniimagenet': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'cub200': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
                'imagenet100': ('models.backbones.vision_transformer', 'vit_tiny', {'patch_size': 2}),
            },
            'vit_small': {
                'cifar100': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'miniimagenet': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'cub200': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
                'imagenet100': ('models.backbones.vision_transformer', 'vit_small', {'patch_size': 2}),
            }
        }
        if self.args.network_type in backbone_mapping:
            if self.args.dataset in backbone_mapping[self.args.network_type]:
                module_name, class_name, kwargs = backbone_mapping[self.args.network_type][self.args.dataset]
                ModuleClass = getattr(__import__(module_name, fromlist=[class_name]), class_name)
                return ModuleClass(**kwargs)
        raise RuntimeError('Something is wrong.')

    def test_NCM(self, model, testloader, args, cls_wise_feature_prototype, save_path):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            tqdm_gen = tqdm(testloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0).cpu()
        embedding_list = torch.nn.functional.normalize(embedding_list, p=2, dim=-1)
        label_list = torch.cat(label_list, dim=0).cpu()
        for i in range(len(cls_wise_feature_prototype)):
            cls_wise_feature_prototype[i] = cls_wise_feature_prototype[i].view(-1)
        proto_list = torch.stack(cls_wise_feature_prototype, dim=0).cpu()
        proto_list = torch.nn.functional.normalize(proto_list, p=2, dim=-1)
        # metric: euclidean, cosine, l2, l1
        pairwise_distance = pairwise_distances(self.args.temperature * np.asarray(embedding_list), self.args.temperature * np.asarray(proto_list), metric='cosine')
        prediction_result = np.argmin(pairwise_distance, axis=1)
        label_list = np.asarray(label_list)
        total_acc = np.sum(prediction_result == label_list) / float(len(label_list))
        num_of_img_per_task = [0] * args.sessions
        correct_prediction_per_task = [0] * args.sessions
        acc_list = [0.0] * args.sessions
        for i in range(args.sessions):
            if i == 0:
                start_class = 0
                end_class = args.base_class
            else:
                start_class = args.base_class + (i - 1) * args.way
                end_class = args.base_class + i * args.way
            for k in range(len(label_list)):
                if start_class <= label_list[k] < end_class:
                    num_of_img_per_task[i] = num_of_img_per_task[i] + 1
                    if label_list[k] == prediction_result[k]:
                        correct_prediction_per_task[i] = correct_prediction_per_task[i] + 1
            if num_of_img_per_task[i] != 0:
                acc_list[i] = correct_prediction_per_task[i] / num_of_img_per_task[i]
        print('TEST, total average accuracy={:.4f}'.format(total_acc))
        print('TEST, task-wise correct prediction: {0}'.format(correct_prediction_per_task))
        print('TEST, task-wise number of images: {0}'.format(num_of_img_per_task))
        print('TEST, task-wise accuracy: {0}'.format(acc_list))
        if save_path != None:
            txt_file = open(save_path, mode='w')
            # txt_file.write('---------------- session: {0} --------------------------------------\n'.format(session))
            txt_file.write('TEST, total average accuracy={:.4f}\n'.format(total_acc))
            txt_file.write('TEST, task-wise correct prediction: {0}\n'.format(correct_prediction_per_task))
            txt_file.write('TEST, task-wise number of images: {0}\n'.format(num_of_img_per_task))
            txt_file.write('TEST, task-wise accuracy: {0}\n'.format(acc_list))
        return prediction_result, label_list

    def calculate_avg_feature_for_each_cls(self, trainset, transform, model, args):
        model = model.eval()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, num_workers=8, pin_memory=True,shuffle=False)
        trainloader.dataset.transform = transform
        overall_avg_feature = []
        overall_avg_cls = []
        final_avg_feature = []
        final_avg_cls = []
        embedding_list = []
        label_list = []
        with torch.no_grad():
            tqdm_gen = tqdm(trainloader)
            for _, batch in enumerate(tqdm_gen, 1):
                data, label = [_.cuda() for _ in batch]
                embedding = model(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        # generate the average feature with all data
        for index in range(args.num_cls):
            class_index = (label_list == index).nonzero()
            embedding_this = embedding_list[class_index.squeeze(-1)]
            embedding_this = embedding_this.mean(0, keepdims=True).cuda()
            overall_avg_feature.append(embedding_this)
            overall_avg_cls.append(index)

        for index in range(args.num_cls):
            final_avg_feature.append(overall_avg_feature[index])
            final_avg_cls.append(overall_avg_cls[index])
        return final_avg_feature, final_avg_cls

    def fusion_aug_generate_label(self, y_a, y_b, session, args):
        current_total_cls_num = args.base_class + session * args.way
        if session == 0:  # base session -> increasing: [(args.base_class) * (args.base_class - 1)]/2
            y_a, y_b = y_a, y_b
            assert y_a != y_b
            if y_a > y_b:  # make label y_a smaller than y_b
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = ((2 * current_total_cls_num - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1
        else:  # incremental session -> increasing: [(args.way) * (args.way - 1)]/2
            y_a = y_a - (current_total_cls_num - args.way)
            y_b = y_b - (current_total_cls_num - args.way)
            assert y_a != y_b
            if y_a > y_b:  # make label y_a smaller than y_b
                tmp = y_a
                y_a = y_b
                y_b = tmp
            label_index = int(((2 * args.way - y_a - 1) * y_a) / 2 + (y_b - y_a) - 1)
        return label_index + current_total_cls_num

    def fusion_aug_one_image(self, x, y, session, args, alpha=20.0, mix_times=4):  # mixup based
        batch_size = x.size()[0]
        mix_data = []
        mix_target = []
        for _ in range(mix_times):
            index = torch.randperm(batch_size).cuda()
            for i in range(batch_size):
                if y[i] != y[index][i]:
                    new_label = self.fusion_aug_generate_label(y[i].item(), y[index][i].item(), session, args)
                    if args.dataset == 'cub200':
                        lam = np.random.beta(alpha, alpha)
                        if lam < 0.4 or lam > 0.6:
                            lam = 0.5
                        mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                    else:
                        # lam = np.random.beta(alpha, alpha)
                        # if lam < 0.4 or lam > 0.6:
                        #     lam = 0.5
                        # mix_data.append(lam * x[i] + (1 - lam) * x[index, :][i])
                        mix_data.append(puzzle_mix(x[i], x[index, :][i], block_num=4))
                    mix_target.append(new_label)
        new_target = torch.Tensor(mix_target)
        y = torch.cat((y, new_target.cuda().long()), 0)
        for item in mix_data:
            x = torch.cat((x, item.unsqueeze(0)), 0)
        return x, y

    def base_train(self, model, trainloader, optimizer, scheduler, session, epoch, args):
        tl = Averager()
        model = model.train()
        # standard classification for pretrain
        tqdm_gen = tqdm(trainloader)
        for i, batch in enumerate(tqdm_gen, 1):
            images = batch[0].cuda()
            target = batch[1].cuda()
            # print(images.shape, target.shape) # torch.Size([100, 3, 32, 32]) torch.Size([100])
            if args.data_fusion and self.args.dataset != 'imagenet100':
                fusion_images, fusion_target = self.fusion_aug_one_image(images[:target.shape[0]], target, session, args, alpha=20.0, mix_times=2)
            # print(fusion_images.shape, fusion_target.shape) # torch.Size([421, 3, 32, 32]) torch.Size([421])
            if self.args.dataset != 'imagenet100':
                output, pesudo_label = model.meta_forward(images)
                meta_loss = model.loss_fn(output, pesudo_label)
                cls_logits = model(fusion_images)
                cls_loss = model.cls_loss_fn(cls_logits, fusion_target)
                loss = cls_loss + meta_loss * self.args.meta_temp
            else:
                meta_loss = torch.Tensor([0.0]).cuda()
                cls_logits = model(images)
                cls_loss = model.cls_loss_fn(cls_logits, target)
                loss = cls_loss + meta_loss * self.args.meta_temp

            sn = 0.0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    w = module.weight
                    u, s, v = torch.svd(w.view(w.size(0), -1))
                    sn += torch.max(s) / torch.min(s)
            spectral_regularization_loss = sn * self.args.regularization_temp
            total_loss = loss + spectral_regularization_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description('Session 0, epo {}, lrc={:.4f}, cls_losses={:.4f}, meta_losses={:.4f}, spectral_losses2={:.4f}, total_losses={:.4f}'
                                     .format(epoch, lrc, cls_loss.item(), meta_loss.item(), spectral_regularization_loss.item(), total_loss.item()))
            tl.add(total_loss.item())
        tl = tl.item()
        return tl

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones, gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)
        return optimizer, scheduler

    def set_up_model(self):
        self.model = Encoder(self.args)
        self.model = self.model.cuda()

    def set_save_path(self):
        self.args.save_path = self.args.save_path + '/%s/' % self.args.dataset
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-MS_%s-Gam_%.2f' % (self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Step_%d-Gam_%.2f' % (self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr1_%.6f-Cosine-Gam_%.2f' % (self.args.epochs_base, self.args.lr_base, self.args.gamma)
        ensure_path(self.args.save_path)
        return None