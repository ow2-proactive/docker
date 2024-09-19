.PHONY: build clean run test

build:
	docker build -t activeeon/tritonserver:22.10-py3-ddd .

clean:
	docker rm -f model-server-ddd
	docker system prune

run:
	docker rm -f model-server-ddd
	docker run -d --rm --shm-size 16384m --name model-server-ddd \
		-v /etc/timezone:/etc/timezone \
		-v /etc/localtime:/etc/localtime \
		-p 19999:9999 -p 19000:9000 -p 18000:8000 -p 18001:8001 -p 18002:8002 -p 18080:8080 -p 19090:9090 -p 16379:6379 \
		-v /opt/models:/models \
		activeeon/tritonserver:22.10-py3-ddd \
		bash -c "tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=5"
	docker logs -f model-server-ddd

test:
	docker exec -it model-server-ddd bash
