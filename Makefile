.PHONY: help install test lint docker-build docker-run \
        compose-up compose-down streamlit aws-setup push-ecr deploy-k8s

help:
	@echo ""
	@echo "  make install        Install dependencies"
	@echo "  make test           Run pytest"
	@echo "  make save-artifacts Save model artifacts"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-run     Run API in Docker"
	@echo "  make compose-up     Start full local stack"
	@echo "  make compose-down   Stop local stack"
	@echo "  make streamlit      Start Streamlit dashboard"
	@echo "  make aws-setup      One-time AWS setup"
	@echo "  make push-ecr       Build and push to ECR"
	@echo "  make deploy-k8s     Deploy to Kubernetes"
	@echo ""

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov httpx ruff

test:
	pytest tests/ -v --tb=short --cov=app --cov-report=term-missing

save-artifacts:
	python scripts/save_artifacts.py

run:
	uvicorn app.main:app --reload --port 8000

IMAGE_NAME ?= fraud-detection
IMAGE_TAG  ?= 1.0.0

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm -p 8000:8000 \
		-v $(PWD)/artifacts:/home/appuser/artifacts:ro \
		-v $(PWD)/configs:/home/appuser/configs:ro \
		$(IMAGE_NAME):$(IMAGE_TAG)

compose-up:
	docker compose up --build -d
	@echo "API:       http://localhost:8000/docs"
	@echo "Streamlit: http://localhost:8501"

compose-monitoring:
	docker compose --profile monitoring up -d
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana:    http://localhost:3000"

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f api

streamlit:
	cd streamlit && streamlit run app.py --server.port 8501

AWS_REGION     ?= eu-west-1
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text)
ECR_URI        ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/fraud-detection
GIT_SHA        ?= $(shell git rev-parse --short HEAD)

aws-setup:
	chmod +x scripts/aws_setup.sh
	./scripts/aws_setup.sh

push-ecr: docker-build
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin \
		$(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_URI):$(GIT_SHA)
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) $(ECR_URI):latest
	docker push $(ECR_URI):$(GIT_SHA)
	docker push $(ECR_URI):latest

deploy-k8s:
	kubectl apply -f k8s/
	kubectl rollout status deployment/fraud-detection -n fraud-detection

rollback-k8s:
	kubectl rollout undo deployment/fraud-detection -n fraud-detection

k8s-logs:
	kubectl logs -f deployment/fraud-detection -n fraud-detection

k8s-status:
	kubectl get pods,svc,ingress,hpa -n fraud-detection