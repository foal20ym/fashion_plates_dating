def train_model(train_ds_batched, val_ds_batched, config, num_classes, input_shape):
    model = build_model(config, num_classes=num_classes, input_shape=input_shape)
    optimizer = Adam(learning_rate=config["training"]["learning_rate"])
    regression = config["task"] == "regression"
    loss = "mean_squared_error" if regression else "sparse_categorical_crossentropy"
    metrics = ["mae", "mse"] if regression else ["accuracy"]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["training"]["early_stopping_patience"],
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        TensorBoard(log_dir=config.get("log_dir", "logs"), histogram_freq=1),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=config["training"]["reduce_lr_patience"],
            min_delta=1e-4,
            verbose=1,
            min_lr=1e-6,
        ),
    ]
    history = model.fit(
        train_ds_batched,
        validation_data=val_ds_batched,
        epochs=config["training"]["epochs"],
        callbacks=callbacks,
        verbose=2,
    )
    return model, history
