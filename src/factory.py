from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


def get_model(cfg):
    """Create model from Keras applications based on config.

    cfg.model.model_name: name of keras application (e.g. EfficientNetB3)
    cfg.model.img_size: input image size
    cfg.model.pretrained: whether to load imagenet weights (bool)
    """
    weights = "imagenet" if getattr(cfg.model, "pretrained", False) else None
    # Keras will attempt to download imagenet weights automatically if not present locally.
    base_model = getattr(applications, cfg.model.model_name)(
        include_top=False,
        weights=weights,
        input_shape=(int(cfg.model.img_size), int(cfg.model.img_size), 3),
        pooling="avg",
    )
    features = base_model.output
    pred_gender = Dense(units=2, activation="softmax", name="pred_gender")(features)
        # age head as distribution over 0..100 (101 classes) — better performance in many works
    pred_age = Dense(units=101, activation="softmax", name="pred_age")(features)
    model = Model(inputs=base_model.input, outputs=[pred_gender, pred_age])

    # optionally unfreeze top N layers for fine-tuning
    # optionally unfreeze top N layers for fine-tuning (cfg.train may be missing in some calls)
    finetune_layers = 0
    try:
        finetune_layers = int(getattr(cfg.train, 'finetune_layers', 0))
    except Exception:
        finetune_layers = 0
    if finetune_layers > 0:
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[-finetune_layers:]:
            layer.trainable = True

    return model


def get_preprocess_fn(cfg):
    """Return preprocessing function corresponding to the chosen backbone if available.

    This allows generators to apply the same preprocessing that the backbone expects.
    """
    preprocess = None
    model_name = cfg.model.model_name
    # map common keras applications to their preprocess_input
    try:
        prep_mod = getattr(applications, model_name)
        if hasattr(prep_mod, "preprocess_input"):
            preprocess = prep_mod.preprocess_input
    except Exception:
        preprocess = None
    return preprocess


def get_optimizer(cfg):
    lr = float(cfg.train.lr)
    if cfg.train.optimizer_name == "sgd":
        return SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif cfg.train.optimizer_name == "adam":
        return Adam(learning_rate=lr)
    else:
        raise ValueError("optimizer name should be 'sgd' or 'adam'")


def get_scheduler(cfg):
    class Schedule:
        def __init__(self, nb_epochs, initial_lr):
            self.epochs = nb_epochs
            self.initial_lr = float(initial_lr)

        def __call__(self, epoch_idx):
            # piecewise constant schedule
            if epoch_idx < self.epochs * 0.25:
                return self.initial_lr
            elif epoch_idx < self.epochs * 0.50:
                return self.initial_lr * 0.2
            elif epoch_idx < self.epochs * 0.75:
                return self.initial_lr * 0.04
            return self.initial_lr * 0.008

    return Schedule(cfg.train.epochs, cfg.train.lr)
