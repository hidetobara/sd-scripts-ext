# Volume Mount
/models: モデル
/train: 教師データ

# Build
docker compose build

# Run
docker compose run --service-ports --rm develop /bin/bash
