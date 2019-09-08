import argparse
import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
import timm
from apex import amp
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.optimize import minimize
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.nn import MSELoss
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from timm.models import TestTimePoolHead
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    ShiftScaleRotate,
    RandomRotate90,
    Normalize,
    Compose,
    OneOf,
    RandomBrightness,
    RandomContrast,
    RandomGamma,
    Resize
)
from albumentations.torch import ToTensor


# https://www.kaggle.com/ratthachat/aptos-updatedv14-preprocessing-ben-s-cropping


parser = argparse.ArgumentParser(description='argument for program')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--model', type=str, default='b3')
parser.add_argument('--sample_weight', type=float, default=.2)
parser.add_argument('--use_pb_weights', type=int, default=0)
parser.add_argument('--image_size', type=int, default=300)
parser.add_argument('--circle_crop', type=int, default=0)
parser.add_argument('--sigmax', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--finetune', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--grad_accum', type=int, default=1)
parser.add_argument('--mixup', type=float, default=0)
parser.add_argument('--lr_scheduler', type=str, default='step')
parser.add_argument('--early_stopping', type=float, default=10)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
IMG_READ_SIZE = 1024
TTA = 2
FOLDS = 5
TRAIN = True
EXP = False
PB_WEIGHT = dict({
    2: 1100,
    1: 270,
    3: 247,
    0: 231,
    4: 80,
})


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_df():
    train_images = '../data/train_images/'
    test_images = '../data/test_images/'
    old_train_images = '../data/old_train_images'

    train_meta = pd.read_csv('../data/train.csv')
    train_meta['image_path'] = train_meta['id_code'].map(lambda x: os.path.join(train_images, x + '.png'))
    train_meta['pb_weight'] = train_meta['diagnosis'].map(PB_WEIGHT)
    train_meta.reset_index(inplace=True)

    test_meta = pd.read_csv('../data/test.csv')
    test_meta['image_path'] = test_meta['id_code'].map(lambda x: os.path.join(test_images, x + '.png'))
    test_meta.reset_index(inplace=True)

    old_train_meta = pd.read_csv('../data/old_train.csv')
    old_train_meta.columns = ['image', 'diagnosis']
    old_train_meta['image_path'] = old_train_meta['image'].map(lambda x: os.path.join(old_train_images, x + '.jpeg'))
    old_train_meta['pb_weight'] = old_train_meta['diagnosis'].map(PB_WEIGHT)
    old_train_meta.reset_index(inplace=True)

    return train_meta, test_meta, old_train_meta


def get_cv_splits(train_meta, old_train_meta, seed=42):
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    train_splits = list(splitter.split(train_meta, y=train_meta['diagnosis']))
    old_train_splits = list(splitter.split(old_train_meta, y=old_train_meta['diagnosis']))
    return train_splits, old_train_splits


def crop_image_from_gray(image, tol=7):
    if image.ndim == 2:
        mask = image > tol
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray_image > tol
        check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape != 0:
            r = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            g = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            b = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            image = np.stack([r, g, b], axis=-1)
    return image


def circle_crop(image):
    height, width, depth = image.shape
    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))
    circle_image = np.zeros((height, width), np.uint8)
    cv2.circle(circle_image, (x, y), int(r), 1, thickness=-1)
    image = cv2.bitwise_and(image, image, mask=circle_image)
    return image


def read_image_and_crop(image_path, sigmax=args.sigmax, use_circle_crop=args.circle_crop):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if use_circle_crop:
        image = circle_crop(image)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_READ_SIZE, IMG_READ_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmax), -4, 128)
    return image


def mixup_data(images, labels, alpha=args.mixup):
    lambd = np.random.beta(alpha, alpha)
    lambd = max(lambd, 1 - lambd)
    permutation = torch.randperm(labels.shape[0]).cuda()
    mixed_images = lambd * images + (1 - lambd) * images[permutation, :]
    alt_labels = labels[permutation, :]
    return mixed_images, alt_labels, lambd


def qwk(logits, labels, num_classes=5, epsilon=1e-10):
    probas = torch.nn.functional.softmax(logits, 0).float()
    labels = torch.nn.functional.one_hot(labels, num_classes).float()
    repeat_op = torch.arange(0, num_classes).view(num_classes, 1).repeat(1, num_classes).float().cuda()
    repeat_op_sq = torch.pow((repeat_op - repeat_op.transpose(0, 1)), 2)
    weights = repeat_op_sq / 4 ** 2

    pred_ = probas ** 2
    pred_norm = pred_ / (epsilon + pred_.sum(1).view(-1, 1))

    hist_rater_a = pred_norm.sum(0)
    hist_rater_b = labels.sum(0)
    conf_mat = torch.matmul(pred_norm.transpose(0, 1), labels)

    nom = (weights * conf_mat).sum()
    denom = (weights * torch.matmul(
        hist_rater_a.view(num_classes, 1),
        hist_rater_b.view(1, num_classes)
    ) / labels.shape[0]).sum()
    return nom / (denom + epsilon)


augment_transform = Compose([
    HorizontalFlip(p=.5),
    VerticalFlip(p=.5),
    RandomRotate90(p=.5),
    ShiftScaleRotate(
        shift_limit=0,
        scale_limit=.1,
        rotate_limit=0,
        p=.5
    ),
    OneOf([RandomContrast(limit=.2), RandomGamma(gamma_limit=(80, 120)), RandomBrightness(limit=.2)], p=.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    # CenterCrop(IMG_CROP_SIZE, IMG_CROP_SIZE, p=.5),
    Resize(args.image_size, args.image_size),
    ToTensor()
])


base_transform = Compose([
    HorizontalFlip(p=.5),
    VerticalFlip(p=.5),
    RandomRotate90(p=.5),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1),
    Resize(args.image_size, args.image_size),
    ToTensor()
])


class APTOSDataset(Dataset):
    def __init__(self, df, mode='train', alt_df=None, sample_weight=args.sample_weight, use_pb_weights=False,
                 transform=None):
        self.mode = mode
        self.labels = None
        self.transform = transform

        if alt_df is not None:
            self.image_paths = np.array(df['image_path'].tolist() + alt_df['image_path'].tolist())
            if use_pb_weights:
                self.sample_weights = np.array(df['pb_weight'].tolist() + (sample_weight * alt_df['pb_weight']).tolist())
            else:
                self.sample_weights = np.array([1] * len(df) + [sample_weight] * len(alt_df))
            self.n_ = int(len(df) + sample_weight * len(alt_df))
            if mode == 'train':
                self.labels = np.array(df['diagnosis'].tolist() + alt_df['diagnosis'].tolist())
        else:
            self.image_paths = np.array(df['image_path'].tolist())
            if use_pb_weights == 1 and mode == 'train':
                self.sample_weights = np.array(df['pb_weight'].tolist())
            else:
                self.sample_weights = None
            self.n_ = len(df)
            if mode == 'train':
                self.labels = np.array(df['diagnosis'].tolist())

    def __len__(self):
        return self.n_

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image_and_crop(image_path)
        if self.transform:
            image = self.transform(image=image)['image']

        if self.mode == 'train':
            label = self.labels[idx]
            return image, label
        else:
            return image


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']


def get_model():
    if args.model == 'b3':
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=1)
    elif args.model == 'b4':
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)
    elif args.model == 'b5':
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1)
    elif args.model == 'resnet':
        model = timm.create_model('ig_resnext101_32x8d', pretrained=True)
        model.fc = torch.nn.Linear(2048, 1)
    return model


set_seed()
train_meta, test_meta, old_train_meta = get_df()
train_splits, old_train_splits = get_cv_splits(train_meta, old_train_meta)
oof_predictions = []
oof_labels = []


for fold in range(FOLDS):
    print('---start on fold {}'.format(fold))
    train_idx, valid_idx = train_splits[fold]
    old_train_idx, _ = old_train_splits[fold]

    if args.sample_weight > 0:
        train_dataset = APTOSDataset(
            df=train_meta.iloc[train_idx],
            alt_df=old_train_meta.iloc[old_train_idx],
            transform=augment_transform,
            use_pb_weights=args.use_pb_weights
        )
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=args.use_pb_weights == 1
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=12,
            pin_memory=True
        )
    else:
        train_dataset = APTOSDataset(
            df=train_meta.iloc[train_idx],
            alt_df=None,
            transform=augment_transform,
            use_pb_weights=args.use_pb_weights
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=12,
            pin_memory=True
        )

    valid_dataset = APTOSDataset(
        df=train_meta.iloc[valid_idx],
        transform=base_transform,
        use_pb_weights=args.use_pb_weights
    )
    if args.use_pb_weights:
        valid_sampler = WeightedRandomSampler(
            weights=valid_dataset.sample_weights,
            num_samples=len(valid_dataset),
            replacement=True
        )
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=12,
            pin_memory=True
        )
    else:
        valid_dataloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=12,
            pin_memory=True
        )

    # test_dataset = APTOSDataset(
    #     df=test_meta,
    #     mode='test',
    #     transform=base_transform
    # )
    #
    # test_dataloader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=6,
    #     pin_memory=True
    # )

    checkpoint = '../model/{}_fold_{}_gpu_{}.pth'.format(args.model, fold, args.gpu)
    model = get_model()
    if args.finetune:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        for param in list(model.parameters())[:-109]:
            param.requires_grad = False

    model.cuda()
    if TRAIN:
        criterion = MSELoss()
        if args.lr_scheduler == 'step':
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.lr/100)
            scheduler = StepLR(optimizer, step_size=7, gamma=.1)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=1, cooldown=1,
                                          min_lr=args.lr / 1000, factor=.1)

        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)
        best_epoch = 0
        best_loss = 100
        for epoch in range(1, args.max_epoch + 1):
            t0 = time.time()
            train_loss = 0.
            model.train()
            optimizer.zero_grad()
            for idx, (images, labels) in enumerate(train_dataloader, 1):
                images, labels = images.cuda(), labels.float().view(-1, 1).cuda()
                if args.mixup > 0:
                    mixed_images, alt_labels, lambd = mixup_data(images, labels)
                    out = model(mixed_images)
                    loss = lambd * criterion(out, labels) + (1 - lambd) * criterion(out, alt_labels)
                else:
                    out = model(images)
                    loss = criterion(out, labels)
                train_loss += loss.item() / len(train_dataloader)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if idx % args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            valid_loss = 0.
            valid_preds = []
            valid_labels = []
            model.eval()
            with torch.no_grad():
                for idx, (images, labels) in enumerate(valid_dataloader, 1):
                    images, labels = images.cuda(), labels.float().view(-1, 1).cuda()
                    out = model(images)
                    loss = criterion(out, labels)
                    valid_loss += loss.item() / len(valid_dataloader)
                    score = out.detach().cpu().squeeze().numpy().reshape(-1, 1)
                    valid_preds.append(score)
                    valid_labels.append(labels.detach().cpu().numpy())

            if args.lr_scheduler == 'step':
                scheduler.step()
            else:
                scheduler.step(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                if not EXP:
                    if args.finetune:
                        torch.save(model.state_dict(), checkpoint.split('.')[0] + '_finetune.pth')
                    else:
                        torch.save(model.state_dict(), checkpoint)

            valid_preds = np.vstack(valid_preds)
            valid_labels = np.vstack(valid_labels)
            optR = OptimizedRounder()
            optR.fit(valid_preds, valid_labels)
            coefficients = optR.coefficients()
            valid_preds = optR.predict(valid_preds, coefficients)
            kappa = cohen_kappa_score(valid_labels, valid_preds, weights='quadratic')
            acc = accuracy_score(valid_labels, valid_preds)
            t1 = time.time()
            print('--epoch {} time {:.4f} train loss {:.4f} valid loss {:.4f} kappa {:.4f} acc {:.4f}'.format(
                epoch, t1 - t0, train_loss, valid_loss, kappa, acc))

            if epoch - best_epoch >= args.early_stopping:
                print('---early stopping triggered best loss {} happened at epoch {}'.format(best_loss, best_epoch))
                break

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    # if args.model == 'resnet' and args.image_size != 224:
    #     model = TestTimePoolHead(model, original_pool=model.default_cfg['pool_size'])
    model.cuda()
    model.eval()
    print('---generating oof predictions:')
    valid_predictions = []
    valid_labels = []
    for tta in range(TTA):
        with torch.no_grad():
            for idx, (images, labels) in tqdm(enumerate(valid_dataloader), desc='tta {}'.format(tta),
                                              total=len(valid_dataloader)):
                images = images.cuda()
                out = model(images)
                score = out.detach().cpu().squeeze().numpy().reshape(-1, 1)
                valid_predictions.append(score)
                valid_labels.append(labels.detach().cpu().squeeze().numpy().reshape(-1, 1))
    valid_predictions = np.vstack(valid_predictions)
    valid_labels = np.vstack(valid_labels)
    oof_predictions.append(valid_predictions)
    oof_labels.append(valid_labels)


oof_predictions = np.vstack(oof_predictions)
oof_labels = np.vstack(oof_labels)
optR = OptimizedRounder()
optR.fit(oof_predictions, oof_labels)
coefficients = optR.coefficients()
print('--coefficients found based on oof predictions:')
print(coefficients)
print('---cohen kappa score on oof predictions:')
print(cohen_kappa_score(oof_labels, optR.predict(oof_predictions, coefficients), weights='quadratic'))
