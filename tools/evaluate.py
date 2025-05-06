def evaluate_model(model, val_ds_batched, config, class_to_idx, min_year, max_year):
    regression = config["task"] == "regression"
    results = model.evaluate(val_ds_batched, verbose=0)
    if regression:
        preds = model.predict(val_ds_batched, verbose=0)
        y_true = np.array([label.numpy() for _, label in val_ds_batched.unbatch()])
        return results, preds, y_true
    else:
        y_score, y_true, y_pred = [], [], []
        for images, labels in val_ds_batched:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))
            y_score.extend(preds)
        return results, np.array(y_score), np.array(y_true), np.array(y_pred)
