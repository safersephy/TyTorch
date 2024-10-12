# `show_logs` target: Run the MLflow server to visualize experiment logs
# Start the MLflow server with the specified configuration
# Set the URI for the backend store (where MLflow metadata is stored)
# Set the default root directory for storing artifacts (e.g., models, plots)
# Set the host for the MLflow server to bind to (localhost in this case)

format:
	ruff format tytorch
lint:
	ruff check tytorch --fix
	mypy tytorch  --ignore-missing-imports --disallow-untyped-defs