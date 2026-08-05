# Object Search Pipeline

## Standalone Service

Run the standalone object-search API from the `object-search` directory:

```bash
PYTHONPATH=. python -m pipeline.online.app \
  --config_file_path /path/to/config.json \
  --host 0.0.0.0 \
  --port 45678 \
  --cors
```

The config file must contain a `maps` array. Each map uses its `id` as the UI display name.

## GUI

A standalone Streamlit app for experimenting with the service lives in `../object-search-gui/`.
See [object-search-gui/README.md](../object-search-gui/README.md) for setup and usage.
