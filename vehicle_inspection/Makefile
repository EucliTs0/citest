IMAGE_REVISION ?= latest
MYPY_TARGET ?= generator
LINT_TARGET ?= generator tests/unit_test
TEST_TARGET ?= tests/unit_test
SERVICE_IMAGE := vehicle-inspection-api:$(IMAGE_REVISION)
TEST_IMAGE := service-test:$(IMAGE_REVISION)
VOLUME_SRC := "${PWD}/service:/service:ro"
VOLUME_TEST := "${PWD}/tests:/tests:ro"
VOLUME_TEST_ARGS := -v $(VOLUME_SRC)

UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: service_image service_run lint mypy test_image unit_test

service_image:
	docker build --pull --no-cache -f docker/service/Dockerfile \
		-t $(SERVICE_IMAGE) .

service_run:
	docker run -p 8080:8080 \
		$(SERVICE_IMAGE)

lint:
	docker run --rm $(VOLUME_TEST_ARGS) $(TEST_IMAGE) pylint $(LINT_TARGET)

mypy:
	docker run --rm $(VOLUME_TEST_ARGS) $(TEST_IMAGE) \
		mypy --ignore-missing-imports --allow-untyped-decorators $(MYPY_TARGET)

test_image:
	docker build --pull -f docker/test/Dockerfile \
		-t $(TEST_IMAGE) ${DOCKER_BUILD_ARGS} .

unit_test:
	docker run --rm --user $(UID):$(GID) \
		 -v "${PWD}:/htmlcov" -w /htmlcov $(VOLUME_TEST_ARGS) $(TEST_IMAGE) \
		python3 -m pytest --disable-warnings \
			--cov=generator --cov-report term --cov-report html $(TEST_TARGET)
