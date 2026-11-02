docker buildx build --platform linux/arm64 --load -t verekia/upscaler .
docker save verekia/upscaler | gzip > /tmp/upscaler.tar.gz
scp /tmp/upscaler.tar.gz midgar:/tmp/
ssh midgar docker load --input /tmp/upscaler.tar.gz
ssh midgar docker compose up -d upscaler
