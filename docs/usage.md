# Usage Notes

## CLI

Run the demo pipeline and print a JSON summary:

```bash
./.venv/bin/python main.py --demo
```

## API

```bash
python main.py --serve   # Swagger UI at http://localhost:8000/docs
```

## Outputs

The pipeline writes these artifacts to the local SQLite store:

- drift scores per batch
- forecast trajectories
- alert history
- integration action log

## Documentation Guides

- **[Developer Guide](developer_guide.md)**: Modular component structure and extensibility notes.
- **[v2 protocol](v2_protocol.md), [v2 results](v2_results.md), [real-data study](v2_realdata.md)**: the maintained research documentation.
