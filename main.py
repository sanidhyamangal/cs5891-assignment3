#!/usr/local/bin/python3
"""
author:Sanidhya Mangal
github:sanidhyamangal
"""

import argparse  # for argument parsing

import tensorflow as tf

from datapipeline.load_imageds import (  # model pipeline for loading image datasets
    LoadData, PredictionDataLoader)
from models import EfficientNetB0Model  # models for training
from models import (DenseNetModel, MobileNetModel, MobileNetV1Model,
                    ResnetV2Model, VGG16Model, XceptionNetModel)
from trainer import ModelManager  # model manager for handing all the ops

MODEL_ARCH = {
    "xception": XceptionNetModel,
    "densenet": DenseNetModel,
    "vgg": VGG16Model,
    "efficientnet": EfficientNetB0Model,
    "resnet": ResnetV2Model,
    "mobilenetv2": MobileNetModel,
    "mobilenetv1": MobileNetV1Model
}


def train_model(args) -> None:
    """
    Helper function for train arg subparser to train the entire network
    """
    # define data loader for the validation and trainer set
    train_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_train_dir
    ]

    val_dataset_loader = [
        LoadData(path=_path,
                 image_shape=(args.height, args.width),
                 channel=args.channel) for _path in args.path_to_eval_dir
    ]

    # retrieve and define the model for the interconnection
    model = MODEL_ARCH.get(args.model_arch, XceptionNetModel)(
        img_shape=(args.height, args.width, args.channel),
        num_classes=len(train_dataset_loader[0].root_labels),
        fine_tune_at=args.fine_tune_at,
        train_from_scratch=args.train_from_scratch,
        custom_input_preprocessing=args.custom_input_preprocessing)

    # print the model arch name for the logs
    print(f"{'='*30}{args.model_arch}{'='*30}")

    # init a model manager to start the training process
    model_manager = ModelManager(name=args.model_arch)

    # prepare the training dataset for ingesting it into the model
    train_dataset = train_dataset_loader[0].create_dataset(
        batch_size=args.batch_size,
        autotune=AUTOTUNE,
        drop_remainder=True,
        prefetch=True)

    # prepare validation dataset for the ingestion process
    validation_dataset = val_dataset_loader[0].create_dataset(
        batch_size=args.batch_size,
        autotune=AUTOTUNE,
        drop_remainder=True,
        prefetch=True)

    if len(train_dataset_loader) > 1:
        for i in train_dataset_loader[1:]:
            train_dataset.concatenate(
                i.create_dataset(batch_size=args.batch_size,
                                 autotune=AUTOTUNE,
                                 drop_remainder=True,
                                 prefetch=True))

    if len(val_dataset_loader) > 1:
        for i in val_dataset_loader[1:]:
            validation_dataset.concatenate(
                i.create_dataset(batch_size=args.batch_size,
                                 autotune=AUTOTUNE,
                                 drop_remainder=True,
                                 prefetch=True))

    # call train function for the training ops
    model_manager.train(
        model,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        learning_rate=args.lr,
        trainer_dataset=train_dataset,
        validation_dataset=validation_dataset,
        check_point_dir=args.checkpoint_dir,
        tensorboard_log=args.tensorboard_log_dir,
        epochs=args.epoch)


def predict_run(args) -> None:
    """Runner function for runing the prediction ops for generating prediction files for sv data"""

    prediction_data_loader = PredictionDataLoader(path=args.path_to_images,
                                                  image_shape=(args.height,
                                                               args.width),
                                                  channel=args.channel)
    prediction_ds = prediction_data_loader.create_dataset(args.batch_size,
                                                          autotune=AUTOTUNE,
                                                          cache=True,
                                                          prefetch=True)

    # retrieve and define the model for the interconnection
    model = MODEL_ARCH.get(args.model_arch, XceptionNetModel)(
        img_shape=(args.height, args.width, args.channel),
        num_classes=args.num_classes,
        custom_input_preprocessing=args.custom_input_preprocessing)

    # print the model arch name for the logs
    print(f"{'='*30}{args.model_arch}{'='*30}")

    # init a model manager to start the training process
    model_manager = ModelManager(name=args.model_arch)
    model_manager.predict(
        model,
        checkpoint_dir=args.checkpoint_dir,
        prediction_dataset=prediction_ds,
        output_file=args.output_file,
        all_file_paths=prediction_data_loader.all_images_path)


if __name__ == "__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    parser = argparse.ArgumentParser(
        description=
        "Script to train and predict the sv images using Xception models")
    subparsers = parser.add_subparsers(
        help='preprocess, augmentate, train or predict')

    parser_train = subparsers.add_parser('train',
                                         help='train the classification model')
    parser_predict = subparsers.add_parser(
        'predict', help='make predications for candidate SVs')

    parser_train.add_argument("--model_arch",
                              choices=[
                                  "xception", "densenet", "efficientnet",
                                  "vgg", "resnet", "mobilenetv2", "mobilenetv1"
                              ],
                              default="xception",
                              dest="model_arch")
    parser_train.add_argument('--epoch',
                              type=int,
                              default=2,
                              help='number of total epoches',
                              dest="epoch")
    parser_train.add_argument('--batch_size',
                              type=int,
                              default=64,
                              help='number of samples in one batch',
                              dest="batch_size")
    parser_train.add_argument('--lr',
                              type=float,
                              default=2e-2,
                              help='initial learning rate for adam',
                              dest="lr")
    parser_train.add_argument('--path_to_train_dir',
                              required=True,
                              help='path to training dataset directory',
                              dest="path_to_train_dir",
                              action="append")
    parser_train.add_argument('--path_to_eval_dir',
                              required=True,
                              help='path to evaluation dataset directory',
                              dest="path_to_eval_dir",
                              action="append")
    parser_train.add_argument('--train_from_scratch',
                              type=bool,
                              default=False,
                              help='path to evaluation dataset directory',
                              dest="train_from_scratch")
    parser_train.add_argument('--custom_input_preprocessing',
                              type=bool,
                              default=False,
                              help='to use custom input processing units',
                              dest="custom_input_preprocessing")

    parser_train.add_argument(
        "--fine_tune_at",
        type=int,
        default=0,
        help="fine tune network from the layer, default to none",
        dest="fine_tune_at")
    parser_train.add_argument(
        '--checkpoint_dir',
        required=True,
        help='path to directory where checkpoints needs to be saved',
        dest="checkpoint_dir")
    parser_train.add_argument('--tensorboard_log_dir',
                              required=True,
                              help='tensorboard summary',
                              dest="tensorboard_log_dir")

    parser_train.add_argument(
        "--height",
        default=224,
        type=int,
        help="height of input images, default value is 224",
        dest="height")
    parser_train.add_argument(
        "--width",
        default=224,
        type=int,
        help="width of input images, default value is 224",
        dest="width")
    parser_train.add_argument(
        "--channel",
        default=3,
        type=int,
        help="channel of input images, default value is 3",
        dest="channel")

    parser_train.set_defaults(func=train_model)

    # arguments for predict section
    parser_predict.add_argument("--model_arch",
                                choices=[
                                    "xception", "densenet", "efficientnet",
                                    "vgg", "resnet", "mobilenetv2",
                                    "mobilenetv1"
                                ],
                                default="xception")
    parser_predict.add_argument('--batch_size',
                                type=int,
                                default=64,
                                help='number of samples in one batch')
    parser_predict.add_argument('--path_to_images',
                                required=True,
                                help='path to prediction images directory')
    parser_predict.add_argument(
        '--output_file',
        required=True,
        help='path where output file needs to be saved')

    parser_predict.add_argument(
        '--checkpoint_dir',
        required=True,
        help='path to directory from which checkpoints needs to be loaded')

    parser_predict.add_argument(
        '--num_classes',
        help='number of classes to pick prediction from',
        default=2,
        type=int)

    parser_predict.add_argument(
        "--height",
        default=224,
        type=int,
        help="height of input images, default value is 224",
        dest="height")
    parser_predict.add_argument(
        "--width",
        default=224,
        type=int,
        help="width of input images, default value is 224",
        dest="width")
    parser_predict.add_argument(
        "--channel",
        default=3,
        type=int,
        help="channel of input images, default value is 3",
        dest="channel")

    parser_predict.add_argument('--custom_input_preprocessing',
                                type=bool,
                                default=False,
                                help='to use custom input processing units',
                                dest="custom_input_preprocessing")

    parser_predict.set_defaults(func=predict_run)

    args = parser.parse_args()
    args.func(args)
