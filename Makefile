build_bot:
	docker build -t doc.smartparking.kz/lp_recognizer_bot_onnx:app telegram/

run:
	docker-compose up -d --build

stop:
	docker-compose down

restart:
	make stop
	make run

build_and_run:
	make build_bot
	make run