import app


def test_load_zoo_model_missing_file_returns_none(monkeypatch):
    app.load_zoo_model.clear()
    errors = []

    monkeypatch.setattr(app.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(
        app.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )

    model, class_names = app.load_zoo_model()

    assert model is None
    assert class_names is None
    assert any("not found" in message.lower() for message in errors)


def test_load_zoo_model_corrupted_bundle_returns_none(monkeypatch):
    app.load_zoo_model.clear()
    errors = []

    monkeypatch.setattr(app.st, "error", lambda message: errors.append(str(message)))
    monkeypatch.setattr(
        app.torch,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("corrupted bundle")),
    )

    model, class_names = app.load_zoo_model()

    assert model is None
    assert class_names is None
    assert any("error loading model" in message.lower() for message in errors)
