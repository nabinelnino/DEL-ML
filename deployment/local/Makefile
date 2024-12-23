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


# Start services in detached mode and follow logs
up-local:
	docker-compose up -d
	docker-compose logs -f

# Alternative approach using & for background process
up-background:
	docker-compose up -d
	@echo "Services started in detached mode. Use 'make logs' to view logs."

logs:
	docker-compose logs -f

down:
	docker compose down

# Stop all running containers
stop:
	@docker stop $$(docker ps -q) 2>/dev/null || true

# Remove all containers
rm-containers:
	@docker rm $$(docker ps -aq) 2>/dev/null || true

# Remove all images
rm-images:
	@docker rmi $$(docker images -q) -f 2>/dev/null || true

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
	docker cp local-mlflow-app-1:/app/data/processed/sample_enamin_subset.smi ../../data/processed/

.PHONY: build run stop rm-containers rm-images clean compose-down prune rebuild


#  docker cp ml-scripts-mlflow-app-1:/app/data/processed/sample_enamin_subset.smi .