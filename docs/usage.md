# Usage Notes

## CLI

Run the demo pipeline and print a JSON summary:

```bash
./.venv/bin/python main.py --demo
```

## Dashboard (Grafana)

Launch the Grafana monitoring dashboard:

```bash
docker compose up -d grafana
```

Then navigate to **http://localhost:3000** in your browser (default credentials: `admin` / `admin`).

## Outputs

The pipeline writes these artifacts to the local SQLite store:

- drift scores per batch
- forecast trajectories
- alert history
- integration action log

## Documentation Guides

- **[PyTorch Deployment & Testing Guide](pytorch_deployment_and_testing_guide.md)**: Full guide for deploying PyTorch MLPs, extracting penultimate layer embeddings, dev testing, and production deployment.
- **[Developer Guide](developer_guide.md)**: Modular component structure and extensibility notes.
- **[Architecture Overview](architecture.md)**: System design and multi-layer drift detection methodology.