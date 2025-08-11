.PHONY: help
help:
	@echo "Available Makefile commands:"
	@echo
	@echo "  Development:"
	@echo "    build           Build the Image"
	@echo "    up              Start the Container"
	@echo "    exec            Open a Bash inside Container"
	@echo "    down            Stop the Container"
	@echo "    destroy         Stop and remove the Container and Volumes"
	@echo
	@echo "  Production:"
	@echo "    pull            Pull the Image from Docker Hub"
	@echo
	@echo "Usage:"
	@echo "  make <command>    Run the specified command"


define FIND_AND_SET_CUDA
	CUDA_VERSION=$$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+'); \
	export CUDA_VERSION=$$CUDA_VERSION
endef

.PHONY: build
build:
	@$(FIND_AND_SET_CUDA); \
	echo "Using CUDA_VERSION=$$CUDA_VERSION"; \
	docker compose build

.PHONY: up
up:
	@echo "Allowing X11 connections from local root..."
	@xhost +local:root
	@$(FIND_AND_SET_CUDA); \
	docker compose up -d

.PHONY: exec
exec:
	@docker exec -it efficientad bash

.PHONY: down
down:
	@xhost -local:root
	@docker compose stop;

.PHONY: destroy
destroy:
	@xhost -local:root
	@docker compose down --volumes

.PHONY: pull
pull:
	@$(FIND_AND_SET_CUDA); \
	TAG=$$CUDA_VERSION-cudnn-devel-ubuntu22.04; \
	echo "Pulling 'efficientad' image from Docker Hub (TAG=$$TAG)..."; \
	docker pull danielcarreira/efficientad:$$TAG; \
	docker tag danielcarreira/efficientad:$$TAG efficientad:$$TAG; \
	echo "Done!"