# Makefile for Docker management
include .env

#check env variables
test-var:
	@echo "Building image with the following parameters:"
	@echo "Region: $(REGION)"
	@echo "Project ID: $(PROJECT_ID)"
	@echo "Artifact Registry: $(ARTIFACT_REGISTRY)"
	@echo "Image Name: $(IMAGE_NAME)"
	@echo "Tag: $(TAG)"

# Build the Docker image
build:
	docker build -t $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(ARTIFACT_REGISTRY)/$(IMAGE_NAME):$(TAG)  --platform=linux/amd64 -f Dockerfile .

#push docker image to registry
push:
	docker push $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(ARTIFACT_REGISTRY)/$(IMAGE_NAME):$(TAG)


deploy:
	gcloud run deploy predict-api \
	--image=northamerica-northeast2-docker.pkg.dev/aircheck-398319/ml-predict-api/predict-api:$(TAG) \
	--port=8000 \
	--service-account=153945772792-compute@developer.gserviceaccount.com \
	--min-instances=1 \
	--region=northamerica-northeast2 \
	--project=aircheck-398319

api-run:
	docker build -t  api-server .
	docker run -d  -p 8000:8000 api-server



# Start services in detached mode and follow logs
up-local:
	docker-compose up -d
	docker-compose logs -f

# Alternative approach using & for background process
up-background:
	docker-compose up -d
	@echo "Services started in detached mode. Use 'make logs' to view logs."



# Stop all running containers
stop:
	@docker stop $$(docker ps -q) 2>/dev/null || true

# Remove all stopped containers and unused images
clean: stop rm-containers rm-images

# Docker Compose down with removing images
compose-down:
	docker-compose down --rmi all

# Full cleanup - stops containers, removes containers and images, and prunes docker system
prune: clean
	docker system prune -af --volumes

# Rebuild and run the container
rebuild: 
	docker-compose build


copy_prediction:
	docker cp ml-scripts-mlflow-app-1:/app/data/processed/sample_enamin_subset.smi ./data/processed/

.PHONY: build run stop rm-containers rm-images clean compose-down prune rebuild


