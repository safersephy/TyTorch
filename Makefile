# `show_logs` target: Run the MLflow server to visualize experiment logs
# Start the MLflow server with the specified configuration
# Set the URI for the backend store (where MLflow metadata is stored)
# Set the default root directory for storing artifacts (e.g., models, plots)
# Set the host for the MLflow server to bind to (localhost in this case)

format:
	isort -v tytorch
	black tytorch

lint:
	ruff check tytorch
	mypy tytorch  --ignore-missing-imports